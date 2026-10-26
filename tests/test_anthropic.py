from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic, AsyncStream, BaseModel, Stream
from anthropic.lib.streaming import AsyncMessageStream, MessageStream
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    TextBlock,
    ToolParam,
    ToolUseBlock,
)

from llm_cassette import CassetteMismatchError, cassette
from llm_cassette.anthropic import install

API_KEY = "sk-ant-cassette-secret-9f3e2a"
_MODEL = "claude-sonnet-5"
_ORIGINALS = {
    (Messages, "create"): Messages.create,
    (AsyncMessages, "create"): AsyncMessages.create,
    (Messages, "stream"): Messages.stream,
    (AsyncMessages, "stream"): AsyncMessages.stream,
}
install()

_calls = 0
_seen_auth: list[str] = []

_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1},
}
_TOOL_MESSAGE: dict[str, Any] = {
    "id": "msg_tool",
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 8, "output_tokens": 4},
}
_START_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "content": [],
    "stop_reason": None,
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 0},
}
_EVENTS: list[dict[str, Any]] = [
    {"type": "message_start", "message": _START_MESSAGE},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hel"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "lo"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]
_WEATHER_TOOL: ToolParam = {
    "name": "get_weather",
    "description": "Look up the weather.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _reset() -> None:
    global _calls
    _calls = 0
    _seen_auth.clear()


def _sse(events: list[dict[str, Any]]) -> httpx2.Response:
    parts: list[str] = []
    for event in events:
        parts.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(parts).encode(),
    )


def _route(request: httpx2.Request) -> httpx2.Response:
    global _calls
    _calls += 1
    _seen_auth.append(request.headers.get("authorization", ""))
    body = json.loads(request.content)
    if request.url.path != "/v1/messages":
        raise AssertionError(request.url.path)
    if body.get("stream") is True:
        return _sse(_EVENTS)
    if "tools" in body:
        return httpx2.Response(200, json=_TOOL_MESSAGE)
    return httpx2.Response(200, json=_MESSAGE)


def _explode(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"network call during replay: {request.url}")


@contextmanager
def _record_client() -> Iterator[Anthropic]:
    with httpx2.Client(transport=httpx2.MockTransport(_route)) as http_client:
        yield Anthropic(api_key=API_KEY, http_client=http_client)


@asynccontextmanager
async def _record_async() -> AsyncIterator[AsyncAnthropic]:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(_route)
    ) as http_client:
        yield AsyncAnthropic(api_key=API_KEY, http_client=http_client)


@contextmanager
def _replay_client() -> Iterator[Anthropic]:
    with httpx2.Client(transport=httpx2.MockTransport(_explode)) as http_client:
        yield Anthropic(api_key=API_KEY, http_client=http_client)


@asynccontextmanager
async def _replay_async() -> AsyncIterator[AsyncAnthropic]:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(_explode)
    ) as http_client:
        yield AsyncAnthropic(api_key=API_KEY, http_client=http_client)


def _text(client: Anthropic) -> Message:
    return client.messages.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )


def _tool_message(client: Anthropic) -> Message:
    return client.messages.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "weather"}],
        tools=[_WEATHER_TOOL],
    )


def _raw_stream(client: Anthropic) -> list[BaseModel]:
    with client.messages.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    ) as stream:
        assert type(stream) is Stream
        return _as_models(list(stream))


def _helper_stream(client: Anthropic) -> tuple[str, Message]:
    with client.messages.stream(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        assert type(stream) is MessageStream
        text = "".join(stream.text_stream)
        return text, stream.get_final_message()


async def _atext(client: AsyncAnthropic) -> Message:
    return await client.messages.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )


async def _atool_message(client: AsyncAnthropic) -> Message:
    return await client.messages.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "weather"}],
        tools=[_WEATHER_TOOL],
    )


async def _araw_stream(client: AsyncAnthropic) -> list[BaseModel]:
    stream = await client.messages.create(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert type(stream) is AsyncStream
    async with stream:
        return _as_models([item async for item in stream])


async def _ahelper_stream(client: AsyncAnthropic) -> tuple[str, Message]:
    async with client.messages.stream(
        model=_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        assert type(stream) is AsyncMessageStream
        text = "".join([chunk async for chunk in stream.text_stream])
        return text, await stream.get_final_message()


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _dumps(models: Sequence[BaseModel]) -> list[dict[str, Any]]:
    return [_dump(model) for model in models]


def _as_models(items: list[object]) -> list[BaseModel]:
    models: list[BaseModel] = []
    for item in items:
        assert isinstance(item, BaseModel)
        models.append(item)
    return models


def _message_text(message: Message) -> str:
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text


def _stream_text(events: Sequence[BaseModel]) -> str:
    parts: list[str] = []
    for event in events:
        if not isinstance(event, RawContentBlockDeltaEvent):
            continue
        delta = event.delta
        if delta.type == "text_delta":
            parts.append(delta.text)
    return "".join(parts)


def _restored() -> bool:
    return (
        Messages.create is _ORIGINALS[(Messages, "create")]
        and AsyncMessages.create is _ORIGINALS[(AsyncMessages, "create")]
        and Messages.stream is _ORIGINALS[(Messages, "stream")]
        and AsyncMessages.stream is _ORIGINALS[(AsyncMessages, "stream")]
    )


@pytest.fixture(autouse=True)
def _clear_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_WEBHOOK_SIGNING_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sync_client_replays_without_network(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "sync.jsonl"
    with _record_client() as record:
        direct = _text(record)
        direct_tool = _tool_message(record)
        with cassette(path, mode="record"):
            recorded = _text(record)
            recorded_tool = _tool_message(record)
    assert _calls == 4

    with _replay_client() as replay, cassette(path, mode="replay"):
        replayed = _text(replay)
        replayed_tool = _tool_message(replay)

    assert _calls == 4
    assert isinstance(replayed, Message)
    assert _message_text(replayed) == "hello"
    assert _dump(replayed) == _dump(direct) == _dump(recorded)
    tool_block = replayed_tool.content[0]
    assert isinstance(tool_block, ToolUseBlock)
    assert tool_block.id == "toolu_1"
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"city": "Paris"}
    assert _dump(replayed_tool) == _dump(direct_tool) == _dump(recorded_tool)
    assert _restored()


def test_async_client_replays_without_network(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "async.jsonl"

    async def scenario() -> None:
        async with _record_async() as record:
            direct = await _atext(record)
            direct_tool = await _atool_message(record)
            async with cassette(path, mode="record"):
                recorded = await _atext(record)
                recorded_tool = await _atool_message(record)
        assert _calls == 4
        async with _replay_async() as replay, cassette(path, mode="replay"):
            replayed = await _atext(replay)
            replayed_tool = await _atool_message(replay)
        assert _calls == 4
        assert isinstance(replayed, Message)
        assert _message_text(replayed) == "hello"
        assert _dump(replayed) == _dump(direct) == _dump(recorded)
        assert _dump(replayed_tool) == _dump(direct_tool) == _dump(recorded_tool)

    asyncio.run(scenario())
    assert _restored()


def test_sync_streaming_replays_sdk_streams(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "stream.jsonl"
    with _record_client() as record:
        direct_raw = _raw_stream(record)
        direct_text, direct_final = _helper_stream(record)
        with cassette(path, mode="record"):
            recorded_raw = _raw_stream(record)
            recorded_text, recorded_final = _helper_stream(record)
    assert _calls == 4

    with _replay_client() as replay, cassette(path, mode="replay"):
        replayed_raw = _raw_stream(replay)
        replayed_text, replayed_final = _helper_stream(replay)

    assert _calls == 4
    assert _stream_text(replayed_raw) == "hello"
    assert replayed_text == direct_text == recorded_text == "hello"
    assert _dumps(replayed_raw) == _dumps(direct_raw) == _dumps(recorded_raw)
    assert _dump(replayed_final) == _dump(direct_final) == _dump(recorded_final)
    assert _message_text(replayed_final) == "hello"


def test_async_streaming_replays_sdk_streams(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "async-stream.jsonl"

    async def scenario() -> None:
        async with _record_async() as record:
            direct_raw = await _araw_stream(record)
            direct_text, direct_final = await _ahelper_stream(record)
            async with cassette(path, mode="record"):
                recorded_raw = await _araw_stream(record)
                recorded_text, recorded_final = await _ahelper_stream(record)
        assert _calls == 4
        async with _replay_async() as replay, cassette(path, mode="replay"):
            replayed_raw = await _araw_stream(replay)
            replayed_text, replayed_final = await _ahelper_stream(replay)
        assert _calls == 4
        assert _stream_text(replayed_raw) == "hello"
        assert replayed_text == direct_text == recorded_text == "hello"
        assert _dumps(replayed_raw) == _dumps(direct_raw) == _dumps(recorded_raw)
        assert _dump(replayed_final) == _dump(direct_final) == _dump(recorded_final)

    asyncio.run(scenario())
    assert _restored()


def test_replay_mismatch_raises(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "mismatch.jsonl"
    with _record_client() as record, cassette(path, mode="record"):
        _text(record)
    assert _calls == 1
    with (
        _replay_client() as replay,
        cassette(path, mode="replay"),
        pytest.raises(CassetteMismatchError) as caught,
    ):
        replay.messages.create(
            model=_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "different"}],
        )
    assert caught.value.index == 0
    assert _calls == 1
    assert _restored()


def test_timeout_is_not_part_of_the_recorded_request(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "timeout.jsonl"
    with _record_client() as record, cassette(path, mode="record"):
        recorded = record.messages.create(
            model=_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
            timeout=httpx2.Timeout(5.0, connect=1.0),
        )
        with record.messages.stream(
            model=_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
            timeout=1,
        ) as recorded_stream:
            recorded_stream.until_done()
    assert _calls == 2
    assert "timeout" not in path.read_text(encoding="utf-8")
    with _replay_client() as replay, cassette(path, mode="replay"):
        replayed = replay.messages.create(
            model=_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
            timeout=30,
        )
        with replay.messages.stream(
            model=_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
        ) as replayed_stream:
            assert replayed_stream.get_final_text() == "hello"
    assert _calls == 2
    assert _message_text(recorded) == _message_text(replayed) == "hello"


def test_api_key_is_absent_from_the_cassette(tmp_path: Path) -> None:
    _reset()
    path = tmp_path / "secret.jsonl"
    with _record_client() as record, cassette(path, mode="record"):
        recorded = _text(record)
    text = path.read_text(encoding="utf-8")
    assert _seen_auth == [f"Bearer {API_KEY}"]
    assert API_KEY not in text
    assert "Bearer" not in text
    assert _message_text(recorded) == "hello"
    lines = text.splitlines()
    assert lines[0] == '{"type": "header", "format_version": 1}'
    event = json.loads(lines[1])
    assert event["provider"] == "anthropic"
    assert event["request"]["operation"] == "messages.create"
    assert event["request"]["body"]["messages"] == [{"content": "hi", "role": "user"}]
    assert event["request"]["body"]["model"] == _MODEL
    assert event["request"]["body"]["max_tokens"] == 32
    assert event["response"]["content"][0]["text"] == "hello"
    assert event["response"]["id"] == "msg_1"


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
