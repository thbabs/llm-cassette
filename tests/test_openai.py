from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, AsyncStream, BaseModel, OpenAI, Stream
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.responses.responses import AsyncResponses, Responses
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionFunctionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseTextDeltaEvent,
)

from llm_cassette import CassetteMismatchError, cassette
from llm_cassette.openai import install

API_KEY = "sk-cassette-secret-9f3e2a"
_ORIGINALS: dict[type[Any], Any] = {
    Completions: Completions.create,
    AsyncCompletions: AsyncCompletions.create,
    Responses: Responses.create,
    AsyncResponses: AsyncResponses.create,
}
install()

_calls = 0
_seen_auth: list[str] = []

_CHAT_OK: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello"},
        }
    ],
}
_CHAT_TOOL: dict[str, Any] = {
    "id": "chatcmpl-tool",
    "object": "chat.completion",
    "created": 2,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
        }
    ],
}
_CHAT_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hel"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]
_RESPONSE_OK: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-4o-mini",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ],
}
_RESPONSE_TOOL: dict[str, Any] = {
    "id": "resp_2",
    "object": "response",
    "created_at": 2,
    "model": "gpt-4o-mini",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_xyz",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        }
    ],
}
_RESPONSE_EVENTS: list[dict[str, Any]] = [
    {
        "type": "response.created",
        "sequence_number": 0,
        "response": _RESPONSE_OK,
    },
    {
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "hello",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1,
    },
    {
        "type": "response.completed",
        "sequence_number": 2,
        "response": _RESPONSE_OK,
    },
]


def _reject_key() -> str:
    raise AssertionError("api key requested")


async def _reject_key_async() -> str:
    raise AssertionError("api key requested")


def _reset() -> None:
    global _calls
    _calls = 0
    _seen_auth.clear()


def _sse(payloads: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads)
    body += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
    )


def _route(request: httpx.Request) -> httpx.Response:
    global _calls
    _calls += 1
    _seen_auth.append(request.headers.get("authorization", ""))
    body = json.loads(request.content)
    path = request.url.path
    if path == "/v1/chat/completions":
        if body.get("stream") is True:
            return _sse(_CHAT_CHUNKS)
        if "tools" in body:
            return httpx.Response(200, json=_CHAT_TOOL)
        return httpx.Response(200, json=_CHAT_OK)
    if path == "/v1/responses":
        if body.get("stream") is True:
            return _sse(_RESPONSE_EVENTS)
        if "tools" in body:
            return httpx.Response(200, json=_RESPONSE_TOOL)
        return httpx.Response(200, json=_RESPONSE_OK)
    raise AssertionError(path)


@contextmanager
def _record_client() -> Iterator[OpenAI]:
    with httpx.Client(transport=httpx.MockTransport(_route)) as http_client:
        yield OpenAI(api_key=API_KEY, http_client=http_client)


@asynccontextmanager
async def _record_async() -> AsyncIterator[AsyncOpenAI]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_route)) as http_client:
        yield AsyncOpenAI(api_key=API_KEY, http_client=http_client)


def _replay_client() -> OpenAI:
    return OpenAI(api_key=_reject_key, timeout=1.0)


def _replay_async() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_reject_key_async, timeout=1.0)


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _dumps(models: Sequence[BaseModel]) -> list[dict[str, Any]]:
    return [_dump(model) for model in models]


def _chat_text(chunks: list[ChatCompletionChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        content = chunk.choices[0].delta.content
        if content:
            parts.append(content)
    return "".join(parts)


def _response_text(events: list[BaseModel]) -> str:
    parts: list[str] = []
    for event in events:
        if isinstance(event, ResponseTextDeltaEvent):
            parts.append(event.delta)
    return "".join(parts)


def _as_models(items: list[object]) -> list[BaseModel]:
    models: list[BaseModel] = []
    for item in items:
        assert isinstance(item, BaseModel)
        models.append(item)
    return models


def _restored() -> bool:
    return (
        Completions.create is _ORIGINALS[Completions]
        and AsyncCompletions.create is _ORIGINALS[AsyncCompletions]
        and Responses.create is _ORIGINALS[Responses]
        and AsyncResponses.create is _ORIGINALS[AsyncResponses]
    )


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "OPENAI_BASE_URL",
        "OPENAI_WEBHOOK_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sync_clients_replay_without_transport_or_api_key(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "sync.jsonl"
    replay = _replay_client()
    with _record_client() as record:
        direct_chat = record.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        direct_response = record.responses.create(model="gpt-4o-mini", input="hi")
        with cassette(path, mode="record"):
            recorded_chat = record.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
            recorded_response = record.responses.create(
                model="gpt-4o-mini",
                input="hi",
            )
    assert _calls == 4

    with cassette(path, mode="replay"):
        replayed_chat = replay.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        replayed_response = replay.responses.create(model="gpt-4o-mini", input="hi")

    assert _calls == 4
    assert isinstance(replayed_chat, ChatCompletion)
    assert isinstance(replayed_response, type(direct_response))
    assert replayed_chat.choices[0].message.content == "hello"
    assert replayed_response.output_text == "hello"
    assert _dump(replayed_chat) == _dump(direct_chat) == _dump(recorded_chat)
    assert (
        _dump(replayed_response) == _dump(direct_response) == _dump(recorded_response)
    )
    assert _restored()


def test_async_clients_replay_without_transport_or_api_key(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "async.jsonl"

    async def scenario() -> None:
        replay = _replay_async()
        async with _record_async() as record:
            direct_chat = await record.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
            direct_response = await record.responses.create(
                model="gpt-4o-mini",
                input="hi",
            )
            async with cassette(path, mode="record"):
                recorded_chat = await record.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                )
                recorded_response = await record.responses.create(
                    model="gpt-4o-mini",
                    input="hi",
                )
        assert _calls == 4
        async with cassette(path, mode="replay"):
            replayed_chat = await replay.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
            replayed_response = await replay.responses.create(
                model="gpt-4o-mini",
                input="hi",
            )
        assert _calls == 4
        assert isinstance(replayed_chat, ChatCompletion)
        assert replayed_chat.choices[0].message.content == "hello"
        assert replayed_response.output_text == "hello"
        assert _dump(replayed_chat) == _dump(direct_chat) == _dump(recorded_chat)
        assert (
            _dump(replayed_response)
            == _dump(direct_response)
            == _dump(recorded_response)
        )

    asyncio.run(scenario())
    assert _restored()


def test_sync_streaming_replays_sdk_streams(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "stream.jsonl"
    replay = _replay_client()
    with _record_client() as record:
        direct_chat = list(
            record.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )
        direct_response = _as_models(
            list(record.responses.create(model="gpt-4o-mini", input="hi", stream=True))
        )
        with cassette(path, mode="record"):
            with record.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            ) as recorded_stream:
                assert type(recorded_stream) is Stream
                recorded_chat = list(recorded_stream)
            recorded_response = _as_models(
                list(
                    record.responses.create(
                        model="gpt-4o-mini",
                        input="hi",
                        stream=True,
                    )
                )
            )
    assert _calls == 4

    with cassette(path, mode="replay"):
        with replay.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ) as replayed_stream:
            assert type(replayed_stream) is Stream
            replayed_chat = list(replayed_stream)
        replayed_events = _as_models(
            list(replay.responses.create(model="gpt-4o-mini", input="hi", stream=True))
        )

    assert _calls == 4
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in replayed_chat)
    assert _chat_text(replayed_chat) == "hello"
    assert _response_text(replayed_events) == "hello"
    assert _dumps(replayed_chat) == _dumps(direct_chat) == _dumps(recorded_chat)
    assert (
        _dumps(replayed_events) == _dumps(direct_response) == _dumps(recorded_response)
    )


def test_async_streaming_replays_sdk_streams(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "async-stream.jsonl"

    async def scenario() -> None:
        replay = _replay_async()
        async with _record_async() as record:
            direct_chat = [
                chunk
                async for chunk in await record.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
            ]
            direct_response = _as_models(
                [
                    event
                    async for event in await record.responses.create(
                        model="gpt-4o-mini",
                        input="hi",
                        stream=True,
                    )
                ]
            )
            async with cassette(path, mode="record"):
                recorded_stream = await record.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
                assert type(recorded_stream) is AsyncStream
                async with recorded_stream:
                    recorded_chat = [chunk async for chunk in recorded_stream]
                recorded_response = _as_models(
                    [
                        event
                        async for event in await record.responses.create(
                            model="gpt-4o-mini",
                            input="hi",
                            stream=True,
                        )
                    ]
                )
        assert _calls == 4
        async with cassette(path, mode="replay"):
            replayed_stream = await replay.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            assert type(replayed_stream) is AsyncStream
            replayed_chat = [chunk async for chunk in replayed_stream]
            replayed_events = _as_models(
                [
                    event
                    async for event in await replay.responses.create(
                        model="gpt-4o-mini",
                        input="hi",
                        stream=True,
                    )
                ]
            )
        assert _calls == 4
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in replayed_chat)
        assert _chat_text(replayed_chat) == "hello"
        assert _response_text(replayed_events) == "hello"
        assert _dumps(replayed_chat) == _dumps(direct_chat) == _dumps(recorded_chat)
        assert (
            _dumps(replayed_events)
            == _dumps(direct_response)
            == _dumps(recorded_response)
        )

    asyncio.run(scenario())
    assert _restored()


def test_tool_calls_round_trip(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "tools.jsonl"
    parameters: dict[str, object] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    chat_tools: list[ChatCompletionFunctionToolParam] = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": parameters},
        }
    ]
    response_tools: list[FunctionToolParam] = [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": parameters,
            "strict": False,
        }
    ]
    replay = _replay_client()
    with _record_client() as record:
        direct_chat = record.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "weather"}],
            tools=chat_tools,
        )
        direct_response = record.responses.create(
            model="gpt-4o-mini",
            input="weather",
            tools=response_tools,
        )
        with cassette(path, mode="record"):
            recorded_chat = record.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "weather"}],
                tools=chat_tools,
            )
            recorded_response = record.responses.create(
                model="gpt-4o-mini",
                input="weather",
                tools=response_tools,
            )
    assert _calls == 4

    with cassette(path, mode="replay"):
        replayed_chat = replay.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "weather"}],
            tools=chat_tools,
        )
        replayed_response = replay.responses.create(
            model="gpt-4o-mini",
            input="weather",
            tools=response_tools,
        )

    assert _calls == 4
    assert _dump(replayed_chat) == _dump(direct_chat) == _dump(recorded_chat)
    assert (
        _dump(replayed_response) == _dump(direct_response) == _dump(recorded_response)
    )
    chat_calls = replayed_chat.choices[0].message.tool_calls
    assert chat_calls is not None
    chat_call = chat_calls[0]
    assert isinstance(chat_call, ChatCompletionMessageFunctionToolCall)
    assert chat_call.id == "call_abc"
    assert chat_call.function.name == "get_weather"
    assert chat_call.function.arguments == '{"city":"Paris"}'
    response_call = replayed_response.output[0]
    assert isinstance(response_call, ResponseFunctionToolCall)
    assert response_call.call_id == "call_xyz"
    assert response_call.name == "get_weather"
    assert response_call.arguments == '{"city":"Paris"}'


def test_replay_mismatch_raises(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "mismatch.jsonl"
    with _record_client() as record, cassette(path, mode="record"):
        record.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert _calls == 1
    replay = _replay_client()
    with cassette(path, mode="replay"), pytest.raises(CassetteMismatchError) as caught:
        replay.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "different"}],
        )
    assert caught.value.index == 0
    assert _calls == 1
    assert _restored()


def test_timeout_is_not_part_of_the_recorded_request(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "timeout.jsonl"
    replay = _replay_client()
    with _record_client() as record, cassette(path, mode="record"):
        recorded_chat = record.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            timeout=httpx.Timeout(5.0, connect=1.0),
        )
        recorded_response = record.responses.create(
            model="gpt-4o-mini",
            input="hi",
            timeout=2,
        )
    assert _calls == 2
    text = path.read_text(encoding="utf-8")
    assert "timeout" not in text
    with cassette(path, mode="replay"):
        replayed_chat = replay.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            timeout=30,
        )
        replayed_response = replay.responses.create(
            model="gpt-4o-mini",
            input="hi",
        )
    assert _calls == 2
    assert replayed_chat.choices[0].message.content == "hello"
    assert recorded_chat.choices[0].message.content == "hello"
    assert replayed_response.output_text == recorded_response.output_text == "hello"


def test_api_key_is_absent_from_the_cassette(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "secret.jsonl"
    with _record_client() as record, cassette(path, mode="record"):
        recorded = record.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    text = path.read_text(encoding="utf-8")
    assert _seen_auth == [f"Bearer {API_KEY}"]
    assert API_KEY not in text
    assert "Bearer" not in text
    assert recorded.choices[0].message.content == "hello"
    lines = text.splitlines()
    assert lines[0] == '{"type": "header", "format_version": 1}'
    event = json.loads(lines[1])
    assert event["provider"] == "openai"
    assert event["request"]["operation"] == "chat.completions.create"
    assert event["request"]["body"]["messages"] == [{"content": "hi", "role": "user"}]
    assert event["response"]["choices"][0]["message"]["content"] == "hello"
    assert event["response"]["id"] == "chatcmpl-1"


def test_methods_are_restored_when_the_cassette_exits(tmp_path: Path) -> None:
    assert _restored()
    outer = tmp_path / "outer.jsonl"
    inner = tmp_path / "inner.jsonl"

    with cassette(outer, mode="none"):
        assert not _restored()
        with cassette(inner, mode="none"):
            assert not _restored()
        assert not _restored()
    assert _restored()

    with pytest.raises(RuntimeError, match="boom"), cassette(outer, mode="none"):
        assert not _restored()
        raise RuntimeError("boom")
    assert _restored()

    @cassette(tmp_path / "decorated.jsonl", mode="none")
    def run() -> None:
        assert not _restored()

    run()
    assert _restored()

    async def async_run() -> None:
        async with cassette(tmp_path / "async.jsonl", mode="none"):
            assert not _restored()

    asyncio.run(async_run())
    assert _restored()
