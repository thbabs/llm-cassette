"""Anthropic SDK recording for the Messages API."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from types import TracebackType
from typing import Any, Protocol, cast

import httpx2
from anthropic import (
    Anthropic,
    AsyncAnthropic,
    AsyncStream,
    BaseModel,
    NotGiven,
    Omit,
    Stream,
    not_given,
)
from anthropic.lib.streaming import AsyncMessageStreamManager, MessageStreamManager
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import Message, RawMessageStreamEvent

from llm_cassette.cassette import Cassette, _CassetteContext, current_cassette
from llm_cassette.errors import CassetteError
from llm_cassette.normalise import ensure_json

_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_CREATE = "messages.create"
_STREAM = "messages.stream"
_installed = False
_depth = 0
_saved: list[tuple[type[Any], str, Any]] | None = None
_original_enter: Callable[[_CassetteContext], Cassette] | None = None
_original_exit: Callable[..., None] | None = None
_original_aenter: Callable[[_CassetteContext], Any] | None = None
_original_aexit: Callable[..., Any] | None = None


class _SyncResource(Protocol):
    _client: Anthropic


class _AsyncResource(Protocol):
    _client: AsyncAnthropic


def install() -> None:
    """Patch Anthropic clients while a cassette is open.

    ``messages.create`` and ``messages.stream`` are replaced on the sync and
    async clients for the lifetime of the cassette, then restored.
    ``create(stream=True)`` is replayed as the SDK's raw event stream.
    ``messages.stream()`` is replayed as ``MessageStream`` or
    ``AsyncMessageStream``. Other responses are rebuilt as ``Message``
    objects.
    """

    global _installed
    if _installed:
        return
    _installed = True
    _hook_cassette()
    _patch_targets()


def _hook_cassette() -> None:
    global _original_enter, _original_exit, _original_aenter, _original_aexit
    _original_enter = _CassetteContext.__enter__
    _original_exit = _CassetteContext.__exit__
    _original_aenter = _CassetteContext.__aenter__
    _original_aexit = _CassetteContext.__aexit__
    _CassetteContext.__enter__ = _enter  # type: ignore[method-assign]
    _CassetteContext.__exit__ = _exit  # type: ignore[method-assign]
    _CassetteContext.__aenter__ = _aenter  # type: ignore[method-assign]
    _CassetteContext.__aexit__ = _aexit  # type: ignore[method-assign]


def _enter(self: _CassetteContext) -> Cassette:
    if _original_enter is None:
        raise CassetteError("Anthropic adapter is not installed")
    session = _original_enter(self)
    try:
        _activate()
    except Exception:
        self.__exit__(None, None, None)
        raise
    return session


def _exit(
    self: _CassetteContext,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
) -> None:
    if _original_exit is None:
        raise CassetteError("Anthropic adapter is not installed")
    try:
        _original_exit(self, exc_type, exc, traceback)
    finally:
        _deactivate()


async def _aenter(self: _CassetteContext) -> Cassette:
    if _original_aenter is None:
        raise CassetteError("Anthropic adapter is not installed")
    session = cast(Cassette, await _original_aenter(self))
    try:
        _activate()
    except Exception:
        await self.__aexit__(None, None, None)
        raise
    return session


async def _aexit(
    self: _CassetteContext,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
) -> None:
    if _original_aexit is None:
        raise CassetteError("Anthropic adapter is not installed")
    try:
        await _original_aexit(self, exc_type, exc, traceback)
    finally:
        _deactivate()


def _patch_targets() -> None:
    global _patches
    _patches = (
        (Messages, "create", _sync_wrapper(Messages.create)),
        (AsyncMessages, "create", _async_wrapper(AsyncMessages.create)),
        (Messages, "stream", _sync_stream_wrapper(Messages.stream)),
        (AsyncMessages, "stream", _async_stream_wrapper(AsyncMessages.stream)),
    )


_patches: tuple[tuple[type[Any], str, Callable[..., Any]], ...] = ()


def _activate() -> None:
    global _depth, _saved
    if _depth == 0:
        saved: list[tuple[type[Any], str, Any]] = []
        for cls, name, wrapper in _patches:
            saved.append((cls, name, getattr(cls, name)))
            setattr(cls, name, wrapper)
        _saved = saved
    _depth += 1


def _deactivate() -> None:
    global _depth, _saved
    if _depth == 0:
        return
    _depth -= 1
    if _depth == 0 and _saved is not None:
        for cls, name, original in _saved:
            setattr(cls, name, original)
        _saved = None


def _sync_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: _SyncResource, *args: Any, **kwargs: Any) -> Any:
        session = current_cassette()
        if session is None:
            return original(self, *args, **kwargs)
        streaming = _streaming(kwargs)

        def call() -> dict[str, Any]:
            return _serialise(original(self, *args, **kwargs), streaming=streaming)

        recorded = session.record_or_replay(
            "llm_call",
            _llm_request(_CREATE, kwargs),
            call,
        )
        return _deserialise_sync(self._client, recorded, streaming=streaming)

    return wrapper


def _async_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapper(self: _AsyncResource, *args: Any, **kwargs: Any) -> Any:
        session = current_cassette()
        if session is None:
            return await original(self, *args, **kwargs)
        streaming = _streaming(kwargs)

        async def call() -> dict[str, Any]:
            result = await original(self, *args, **kwargs)
            return await _serialise_async(result, streaming=streaming)

        recorded = await session.record_or_replay(
            "llm_call",
            _llm_request(_CREATE, kwargs),
            call,
        )
        return _deserialise_async(self._client, recorded, streaming=streaming)

    return wrapper


def _sync_stream_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: _SyncResource, *args: Any, **kwargs: Any) -> Any:
        session = current_cassette()
        if session is None:
            return original(self, *args, **kwargs)
        request = _llm_request(_STREAM, kwargs)
        output_format = _output_format(kwargs)

        def api_request() -> Stream[RawMessageStreamEvent]:
            def call() -> dict[str, Any]:
                manager = original(self, *args, **kwargs)
                with manager as live:
                    return _serialise(live._raw_stream, streaming=True)

            recorded = session.record_or_replay("llm_call", request, call)
            return _sync_stream(self._client, _events(recorded))

        return MessageStreamManager(api_request, output_format=output_format)

    return wrapper


def _async_stream_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: _AsyncResource, *args: Any, **kwargs: Any) -> Any:
        session = current_cassette()
        if session is None:
            return original(self, *args, **kwargs)
        request = _llm_request(_STREAM, kwargs)
        output_format = _output_format(kwargs)

        async def api_request() -> AsyncStream[RawMessageStreamEvent]:
            async def call() -> dict[str, Any]:
                manager = original(self, *args, **kwargs)
                async with manager as live:
                    return await _serialise_async(live._raw_stream, streaming=True)

            recorded = await session.record_or_replay("llm_call", request, call)
            return _async_stream(self._client, _events(recorded))

        return AsyncMessageStreamManager(api_request(), output_format=output_format)

    return wrapper


def _streaming(kwargs: Mapping[str, Any]) -> bool:
    return kwargs.get("stream") is True


def _output_format(kwargs: Mapping[str, Any]) -> Any:
    value = kwargs.get("output_format", not_given)
    if _ignored(value):
        return not_given
    return value


def _llm_request(operation: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "request": {
            "operation": operation,
            "body": _request_body(kwargs),
        },
    }


def _request_body(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for key, value in kwargs.items():
        if _ignored(value):
            continue
        raw[key] = _to_json(value)
    body = ensure_json(raw)
    if not isinstance(body, dict):
        raise CassetteError("Anthropic request must be a JSON object")
    return body


def _ignored(value: object) -> bool:
    return isinstance(value, (NotGiven, Omit))


def _to_json(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return _to_json(value.value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CassetteError(
                    f"Anthropic request keys must be strings, got {type(key).__name__}"
                )
            converted[key] = _to_json(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value


def _serialise(result: object, *, streaming: bool) -> dict[str, Any]:
    if not streaming:
        return _dump_model(result)
    if not isinstance(result, Stream):
        raise CassetteError("expected an Anthropic stream")
    return {"events": [_dump_model(item) for item in result]}


async def _serialise_async(result: object, *, streaming: bool) -> dict[str, Any]:
    if not streaming:
        return _dump_model(result)
    if not isinstance(result, AsyncStream):
        raise CassetteError("expected an Anthropic async stream")
    events: list[dict[str, Any]] = []
    async for item in result:
        events.append(_dump_model(item))
    return {"events": events}


def _dump_model(value: object) -> dict[str, Any]:
    if not isinstance(value, BaseModel):
        raise CassetteError(
            f"cannot record an Anthropic value of type {type(value).__name__}"
        )
    return value.model_dump(mode="json")


def _deserialise_sync(
    client: Anthropic,
    recorded: object,
    *,
    streaming: bool,
) -> Any:
    if streaming:
        return _sync_stream(client, _events(recorded))
    return Message.model_validate(_object(recorded))


def _deserialise_async(
    client: AsyncAnthropic,
    recorded: object,
    *,
    streaming: bool,
) -> Any:
    if streaming:
        return _async_stream(client, _events(recorded))
    return Message.model_validate(_object(recorded))


def _events(recorded: object) -> list[dict[str, Any]]:
    raw = _object(recorded).get("events")
    if not isinstance(raw, list):
        raise CassetteError("recorded Anthropic stream is missing events")
    events: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise CassetteError("recorded Anthropic stream event must be an object")
        events.append(item)
    return events


def _object(recorded: object) -> dict[str, Any]:
    if isinstance(recorded, dict):
        return recorded
    raise CassetteError("recorded Anthropic response must be an object")


def _sync_stream(
    client: Anthropic,
    events: Sequence[Mapping[str, Any]],
) -> Stream[RawMessageStreamEvent]:
    return Stream(
        cast_to=cast(type[RawMessageStreamEvent], RawMessageStreamEvent),
        response=_event_stream(events),
        client=client,
    )


def _async_stream(
    client: AsyncAnthropic,
    events: Sequence[Mapping[str, Any]],
) -> AsyncStream[RawMessageStreamEvent]:
    return AsyncStream(
        cast_to=cast(type[RawMessageStreamEvent], RawMessageStreamEvent),
        response=_event_stream(events),
        client=client,
    )


def _event_stream(events: Sequence[Mapping[str, Any]]) -> httpx2.Response:
    lines: list[str] = []
    for event in events:
        name = event.get("type")
        if not isinstance(name, str) or name == "":
            raise CassetteError("recorded Anthropic event is missing a type")
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        lines.append(f"event: {name}\ndata: {payload}\n\n")
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(lines).encode(),
        request=httpx2.Request("POST", _MESSAGES_URL),
    )
