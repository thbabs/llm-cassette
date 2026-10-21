"""Cassette sessions and the cassette() helper."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import partial, wraps
from inspect import iscoroutinefunction
from pathlib import Path
from types import TracebackType
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload

from llm_cassette.errors import (
    CassetteError,
    CassetteMismatchError,
    CassetteNotFoundError,
)
from llm_cassette.format import read_events, sorted_json, write_cassette
from llm_cassette.normalise import (
    canonical_json,
    clone_json,
    ensure_json,
    normalise_request,
    redact_secrets,
    request_hash,
)
from llm_cassette.types import EventKind, JSONValue, Matcher, Mode

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class _NativeCoroutine(Protocol):
    @property
    def cr_running(self) -> bool: ...

    def close(self) -> None: ...


_MODES = frozenset({"once", "replay", "record", "none"})
_LLM_KEYS = frozenset({"provider", "request"})
_TOOL_KEYS = frozenset({"name", "args"})
_current: ContextVar[Cassette | None] = ContextVar("llm_cassette_current", default=None)


def current_cassette() -> Cassette | None:
    """Return the session active in this context, if any."""

    return _current.get()


class Cassette:
    """One open recording or replay session.

    Create sessions with ``cassette()``. A session is not safe for concurrent
    use.
    """

    def __init__(
        self,
        path: Path,
        mode: Mode,
        *,
        strict: bool,
        matcher: Matcher | None,
    ) -> None:
        self._path = path
        self._mode = mode
        self._strict = strict
        self._matcher = matcher
        self._passthrough = False
        self._recording = False
        self._events: list[_Interaction] = []
        self._cursor = 0
        self._closed = False
        self._inflight = 0
        self._orphans: list[Coroutine[Any, Any, Any]] = []
        self._prepare()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def recording(self) -> bool:
        """True when calls are executed and stored."""

        return self._recording

    @overload
    def record_or_replay(
        self,
        kind: EventKind,
        request: Mapping[str, Any],
        call_fn: Callable[[], Awaitable[T]],
    ) -> Awaitable[T]: ...

    @overload
    def record_or_replay(
        self,
        kind: EventKind,
        request: Mapping[str, Any],
        call_fn: Callable[[], T],
    ) -> T: ...

    def record_or_replay(
        self,
        kind: EventKind,
        request: Mapping[str, Any],
        call_fn: Callable[..., Any],
    ) -> Any:
        """Record ``call_fn`` or replay the next matching interaction.

        ``call_fn`` runs only while recording or in ``none`` mode. Async
        functions are detected and must be awaited. Recorded results are
        returned as JSON with secrets redacted, so replay matches recording.
        Tuples in recorded values become lists.
        """

        self._check_open()
        checked = _check_kind(kind)
        identity = _identity(checked, request)
        if _is_async_callable(call_fn):
            async_fn = cast(Callable[[], Awaitable[Any]], call_fn)
            self._inflight += 1
            coroutine = self._run_async(checked, identity, async_fn)
            self._orphans.append(coroutine)
            return coroutine
        return self._run_sync(checked, identity, call_fn)

    def _prepare(self) -> None:
        if self._mode != "none" and self._path.exists() and not self._path.is_file():
            raise CassetteError(f"cassette path is not a file: {self._path}")
        if self._mode == "none":
            self._passthrough = True
            return
        if self._mode == "record" or (
            self._mode == "once" and not self._path.is_file()
        ):
            self._recording = True
            return
        if not self._path.is_file():
            raise CassetteNotFoundError(self._path)
        self._events = _load_interactions(self._path)

    def _check_open(self) -> None:
        if self._closed:
            raise CassetteError("cassette is closed")

    def _run_sync(
        self,
        kind: EventKind,
        identity: dict[str, JSONValue],
        call_fn: Callable[[], Any],
    ) -> Any:
        if self._passthrough:
            return call_fn()
        if self._recording:
            return self._store(kind, identity, call_fn())
        return self._take(identity)

    async def _run_async(
        self,
        kind: EventKind,
        identity: dict[str, JSONValue],
        call_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        try:
            self._check_open()
            if self._passthrough:
                return await call_fn()
            if self._recording:
                return self._store(kind, identity, await call_fn())
            return self._take(identity)
        finally:
            self._inflight -= 1

    def _store(
        self,
        kind: EventKind,
        identity: dict[str, JSONValue],
        response: object,
    ) -> JSONValue:
        payload = sorted_json(redact_secrets(ensure_json(response)))
        self._events.append(_Interaction(kind, identity, payload))
        return clone_json(payload)

    def _take(self, identity: dict[str, JSONValue]) -> JSONValue:
        if self._cursor >= len(self._events):
            raise self._mismatch(
                index=self._cursor,
                expected=None,
                received=identity,
                summary="no further recorded interactions",
            )
        expected = self._events[self._cursor].identity
        if not self._matches(expected, identity):
            raise self._mismatch(
                index=self._cursor,
                expected=expected,
                received=identity,
                summary="request does not match",
            )
        payload = self._events[self._cursor].payload
        self._cursor += 1
        return clone_json(payload)

    def _matches(
        self,
        expected: dict[str, JSONValue],
        received: dict[str, JSONValue],
    ) -> bool:
        if self._matcher is not None:
            matcher = self._matcher
        elif self._strict:
            matcher = _canonical_equal
        else:
            matcher = _hash_equal
        return bool(matcher(clone_json(expected), clone_json(received)))

    def _mismatch(
        self,
        *,
        index: int,
        expected: JSONValue | None,
        received: JSONValue | None,
        summary: str,
    ) -> CassetteMismatchError:
        expected_value = None if expected is None else clone_json(expected)
        received_value = None if received is None else clone_json(received)
        message = (
            f"{self._path}: {summary} at interaction {index}\n"
            f"expected:\n{_format_json(expected_value)}\n"
            f"received:\n{_format_json(received_value)}"
        )
        return CassetteMismatchError(
            message,
            expected=expected_value,
            received=received_value,
            index=index,
            path=self._path,
        )

    def _ensure_consumed(self) -> None:
        remaining = len(self._events) - self._cursor
        if remaining == 0:
            return
        noun = "interaction" if remaining == 1 else "interactions"
        raise self._mismatch(
            index=self._cursor,
            expected=self._events[self._cursor].identity,
            received=None,
            summary=f"{remaining} unconsumed {noun} starting",
        )

    def _close_orphans(self) -> None:
        orphans = self._orphans
        self._orphans = []
        for coroutine in orphans:
            native = cast(_NativeCoroutine, coroutine)
            if native.cr_running:
                continue
            native.close()

    def _finish(self, exc: BaseException | None) -> None:
        if self._closed:
            return
        try:
            leaked = exc is None and self._inflight > 0
            self._close_orphans()
            if leaked:
                raise CassetteError("async record_or_replay call was not awaited")
            if (
                exc is None
                and self._strict
                and not self._recording
                and not self._passthrough
            ):
                self._ensure_consumed()
            if exc is None and self._recording:
                write_cassette(
                    self._path,
                    [event.as_event() for event in self._events],
                )
        finally:
            self._closed = True


class _CassetteContext:
    def __init__(
        self,
        path: Path,
        mode: Mode,
        *,
        strict: bool,
        matcher: Matcher | None,
    ) -> None:
        self._path = path
        self._mode = mode
        self._strict = strict
        self._matcher = matcher
        self._stack: list[tuple[Cassette, Token[Cassette | None]]] = []

    def _open(self) -> Cassette:
        session = Cassette(
            self._path,
            self._mode,
            strict=self._strict,
            matcher=self._matcher,
        )
        token = _current.set(session)
        self._stack.append((session, token))
        return session

    def __enter__(self) -> Cassette:
        return self._open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        session, token = self._stack.pop()
        try:
            session._finish(exc)
        finally:
            _current.reset(token)

    async def __aenter__(self) -> Cassette:
        return self._open()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, traceback)

    def __call__(self, fn: Callable[P, R]) -> Callable[P, R]:
        if _is_async_callable(fn):
            async_fn = cast(Callable[P, Awaitable[Any]], fn)

            @wraps(fn)
            async def async_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
                async with self:
                    return await async_fn(*args, **kwargs)

            return cast(Callable[P, R], async_wrapped)

        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with self:
                return fn(*args, **kwargs)

        return wrapped


def cassette(
    path: str | os.PathLike[str],
    mode: Mode = "once",
    *,
    strict: bool = False,
    matcher: Matcher | None = None,
) -> _CassetteContext:
    """Open a cassette as a context manager or decorator.

    ``once`` records when ``path`` is missing and replays otherwise.
    ``replay`` replays an existing file. ``record`` calls through and
    overwrites ``path`` when the block finishes successfully. ``none``
    calls through and leaves the file untouched.

    Interactions match in order. The default matcher compares hashes of
    the normalised requests. ``strict=True`` compares the normalised
    requests themselves and raises if any recorded interaction was not
    replayed. ``matcher`` replaces that comparison. It is called with the
    expected and received normalised requests.
    """

    return _CassetteContext(
        _as_path(path),
        _check_mode(mode),
        strict=strict,
        matcher=_check_matcher(matcher),
    )


@dataclass(frozen=True)
class _Interaction:
    kind: EventKind
    identity: dict[str, JSONValue]
    payload: JSONValue

    def as_event(self) -> dict[str, JSONValue]:
        if self.kind == "llm_call":
            return {
                "type": "llm_call",
                "provider": self.identity["provider"],
                "request": self.identity["request"],
                "response": self.payload,
            }
        return {
            "type": "tool_call",
            "name": self.identity["name"],
            "args": self.identity["args"],
            "result": self.payload,
        }


def _load_interactions(path: Path) -> list[_Interaction]:
    return [_interaction_from_event(path, event) for event in read_events(path)]


def _interaction_from_event(
    path: Path,
    event: dict[str, JSONValue],
) -> _Interaction:
    kind = event.get("type")
    if kind == "llm_call":
        identity = _identity(
            "llm_call",
            {"provider": event["provider"], "request": event["request"]},
        )
        payload = sorted_json(redact_secrets(event["response"]))
        return _Interaction("llm_call", identity, payload)
    if kind == "tool_call":
        identity = _identity(
            "tool_call",
            {"name": event["name"], "args": event["args"]},
        )
        payload = sorted_json(redact_secrets(event["result"]))
        return _Interaction("tool_call", identity, payload)
    raise CassetteError(f"{path}: unknown event type {kind!r}")


def _identity(kind: EventKind, request: Mapping[str, Any]) -> dict[str, JSONValue]:
    if not isinstance(request, Mapping) or any(
        not isinstance(key, str) for key in request
    ):
        raise CassetteError("request must be a mapping with string keys")
    if kind == "llm_call":
        _reject_unexpected(kind, request, _LLM_KEYS)
        if "provider" not in request or "request" not in request:
            raise CassetteError(
                "llm_call request must include 'provider' and 'request'"
            )
        provider = _non_empty_str(
            request["provider"],
            "llm_call provider must be a non-empty string",
        )
        return {
            "type": "llm_call",
            "provider": provider,
            "request": normalise_request(request["request"]),
        }
    _reject_unexpected(kind, request, _TOOL_KEYS)
    if "name" not in request or "args" not in request:
        raise CassetteError("tool_call request must include 'name' and 'args'")
    name = _non_empty_str(request["name"], "tool_call name must be a non-empty string")
    return {
        "type": "tool_call",
        "name": name,
        "args": normalise_request(request["args"]),
    }


def _reject_unexpected(
    kind: EventKind,
    request: Mapping[str, Any],
    allowed: frozenset[str],
) -> None:
    unexpected = sorted(set(request) - allowed)
    if not unexpected:
        return
    joined = ", ".join(unexpected)
    raise CassetteError(f"{kind} request has unexpected keys: {joined}")


def _non_empty_str(value: object, message: str) -> str:
    if isinstance(value, str) and value != "":
        return value
    raise CassetteError(message)


def _check_kind(kind: str) -> EventKind:
    if kind == "llm_call":
        return "llm_call"
    if kind == "tool_call":
        return "tool_call"
    raise CassetteError(f"unknown event kind {kind!r}")


def _check_mode(mode: str) -> Mode:
    if mode in _MODES:
        return cast(Mode, mode)
    raise ValueError(
        f"unknown cassette mode {mode!r}; expected one of: once, replay, record, none"
    )


def _check_matcher(matcher: Matcher | None) -> Matcher | None:
    if matcher is not None and not callable(matcher):
        raise TypeError("matcher must be callable")
    return matcher


def _as_path(path: object) -> Path:
    if isinstance(path, str):
        return Path(path)
    if isinstance(path, os.PathLike):
        return Path(path)
    raise TypeError("cassette path must be a filesystem path")


def _hash_equal(expected: Any, received: Any) -> bool:
    return request_hash(expected) == request_hash(received)


def _canonical_equal(expected: Any, received: Any) -> bool:
    return canonical_json(expected) == canonical_json(received)


def _format_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    seen: set[int] = set()
    current: Any = fn
    while id(current) not in seen:
        seen.add(id(current))
        if iscoroutinefunction(current):
            return True
        if isinstance(current, partial):
            current = current.func
            continue
        unbound = getattr(current, "__func__", None)
        if unbound is not None:
            current = unbound
            continue
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            return False
        current = wrapped
    return False
