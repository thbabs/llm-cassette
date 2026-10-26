"""OpenAI SDK recording for chat completions and responses."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Protocol, cast

import httpx
from openai import AsyncOpenAI, AsyncStream, BaseModel, NotGiven, Omit, OpenAI, Stream
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.responses.responses import AsyncResponses, Responses
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent

from llm_cassette.cassette import Cassette, _CassetteContext, current_cassette
from llm_cassette.errors import CassetteError
from llm_cassette.normalise import ensure_json

_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_RESPONSES_URL = "https://api.openai.com/v1/responses"
_installed = False
_depth = 0
_saved: list[tuple[type[Any], str, Any]] | None = None
_original_enter: Callable[[_CassetteContext], Cassette] | None = None
_original_exit: Callable[..., None] | None = None
_original_aenter: Callable[[_CassetteContext], Any] | None = None
_original_aexit: Callable[..., Any] | None = None


class _SyncResource(Protocol):
    _client: OpenAI


class _AsyncResource(Protocol):
    _client: AsyncOpenAI


@dataclass(frozen=True)
class _Endpoint:
    operation: str
    model: type[BaseModel]
    stream_model: Any
    url: str


_CHAT = _Endpoint(
    "chat.completions.create",
    ChatCompletion,
    ChatCompletionChunk,
    _CHAT_URL,
)
_RESPONSES = _Endpoint(
    "responses.create",
    Response,
    ResponseStreamEvent,
    _RESPONSES_URL,
)


def install() -> None:
    """Patch OpenAI clients while a cassette is open.

    ``chat.completions.create`` and ``responses.create`` are replaced on the
    sync and async clients for the lifetime of the cassette, then restored.
    Responses are rebuilt as ``ChatCompletion`` and ``Response`` objects.
    Streams are rebuilt as the SDK's stream objects. A per-request
    ``timeout`` is not recorded: it is a transport setting, not part of
    the API body.
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
        raise CassetteError("OpenAI adapter is not installed")
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
        raise CassetteError("OpenAI adapter is not installed")
    try:
        _original_exit(self, exc_type, exc, traceback)
    finally:
        _deactivate()


async def _aenter(self: _CassetteContext) -> Cassette:
    if _original_aenter is None:
        raise CassetteError("OpenAI adapter is not installed")
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
        raise CassetteError("OpenAI adapter is not installed")
    try:
        await _original_aexit(self, exc_type, exc, traceback)
    finally:
        _deactivate()


def _patch_targets() -> None:
    global _patches
    _patches = (
        (Completions, "create", _sync_wrapper(Completions.create, _CHAT)),
        (AsyncCompletions, "create", _async_wrapper(AsyncCompletions.create, _CHAT)),
        (Responses, "create", _sync_wrapper(Responses.create, _RESPONSES)),
        (AsyncResponses, "create", _async_wrapper(AsyncResponses.create, _RESPONSES)),
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


def _sync_wrapper(
    original: Callable[..., Any],
    endpoint: _Endpoint,
) -> Callable[..., Any]:
    def wrapper(self: _SyncResource, *args: Any, **kwargs: Any) -> Any:
        session = current_cassette()
        if session is None:
            return original(self, *args, **kwargs)
        streaming = _streaming(kwargs)

        def call() -> dict[str, Any]:
            return _serialise(original(self, *args, **kwargs), streaming=streaming)

        recorded = session.record_or_replay(
            "llm_call",
            _llm_request(endpoint, kwargs),
            call,
        )
        return _deserialise_sync(
            self._client,
            recorded,
            endpoint=endpoint,
            streaming=streaming,
        )

    return wrapper


def _async_wrapper(
    original: Callable[..., Any],
    endpoint: _Endpoint,
) -> Callable[..., Any]:
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
            _llm_request(endpoint, kwargs),
            call,
        )
        return _deserialise_async(
            self._client,
            recorded,
            endpoint=endpoint,
            streaming=streaming,
        )

    return wrapper


def _streaming(kwargs: Mapping[str, Any]) -> bool:
    return kwargs.get("stream") is True


def _llm_request(endpoint: _Endpoint, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": "openai",
        "request": {
            "operation": endpoint.operation,
            "body": _request_body(kwargs),
        },
    }


def _request_body(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "timeout" or _ignored(value):
            continue
        raw[key] = _to_json(value)
    body = ensure_json(raw)
    if not isinstance(body, dict):
        raise CassetteError("OpenAI request must be a JSON object")
    return body


def _ignored(value: object) -> bool:
    return isinstance(value, (NotGiven, Omit))


def _to_json(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return _to_json(value.value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CassetteError(
                    f"OpenAI request keys must be strings, got {type(key).__name__}"
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
        raise CassetteError("expected an OpenAI stream")
    return {"chunks": [_dump_model(item) for item in result]}


async def _serialise_async(result: object, *, streaming: bool) -> dict[str, Any]:
    if not streaming:
        return _dump_model(result)
    if not isinstance(result, AsyncStream):
        raise CassetteError("expected an OpenAI async stream")
    chunks: list[dict[str, Any]] = []
    async for item in result:
        chunks.append(_dump_model(item))
    return {"chunks": chunks}


def _dump_model(value: object) -> dict[str, Any]:
    if not isinstance(value, BaseModel):
        raise CassetteError(
            f"cannot record an OpenAI value of type {type(value).__name__}"
        )
    return value.model_dump(mode="json")


def _deserialise_sync(
    client: OpenAI,
    recorded: object,
    *,
    endpoint: _Endpoint,
    streaming: bool,
) -> Any:
    if streaming:
        return _sync_stream(client, endpoint, _chunks(recorded))
    return endpoint.model.model_validate(_object(recorded))


def _deserialise_async(
    client: AsyncOpenAI,
    recorded: object,
    *,
    endpoint: _Endpoint,
    streaming: bool,
) -> Any:
    if streaming:
        return _async_stream(client, endpoint, _chunks(recorded))
    return endpoint.model.model_validate(_object(recorded))


def _chunks(recorded: object) -> list[dict[str, Any]]:
    raw = _object(recorded).get("chunks")
    if not isinstance(raw, list):
        raise CassetteError("recorded OpenAI stream is missing chunks")
    chunks: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise CassetteError("recorded OpenAI stream chunk must be an object")
        chunks.append(item)
    return chunks


def _object(recorded: object) -> dict[str, Any]:
    if isinstance(recorded, dict):
        return recorded
    raise CassetteError("recorded OpenAI response must be an object")


def _sync_stream(
    client: OpenAI,
    endpoint: _Endpoint,
    chunks: Sequence[Mapping[str, Any]],
) -> Stream[Any]:
    return Stream(
        cast_to=cast(type[Any], endpoint.stream_model),
        response=_event_stream(endpoint.url, chunks),
        client=client,
    )


def _async_stream(
    client: AsyncOpenAI,
    endpoint: _Endpoint,
    chunks: Sequence[Mapping[str, Any]],
) -> AsyncStream[Any]:
    return AsyncStream(
        cast_to=cast(type[Any], endpoint.stream_model),
        response=_event_stream(endpoint.url, chunks),
        client=client,
    )


def _event_stream(url: str, chunks: Sequence[Mapping[str, Any]]) -> httpx.Response:
    lines = [
        f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"
        for chunk in chunks
    ]
    lines.append("data: [DONE]\n\n")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(lines).encode(),
        request=httpx.Request("POST", url),
    )
