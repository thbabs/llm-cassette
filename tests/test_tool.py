from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llm_cassette import CassetteMismatchError, cassette, tool


def test_sync_tool_replays_without_calling(tmp_path: Path) -> None:
    calls: list[str] = []

    @tool
    def get_weather(city: str, unit: str = "c") -> dict[str, str]:
        calls.append(f"{city}:{unit}")
        return {"city": city, "unit": unit}

    assert get_weather.__name__ == "get_weather"
    path = tmp_path / "weather.jsonl"
    with cassette(path, mode="record"):
        assert get_weather("Paris") == {"city": "Paris", "unit": "c"}
    assert calls == ["Paris:c"]

    event = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert event == {
        "type": "tool_call",
        "name": "get_weather",
        "args": {"city": "Paris", "unit": "c"},
        "result": {"city": "Paris", "unit": "c"},
    }

    with cassette(path, mode="replay"):
        assert get_weather(city="Paris") == {"city": "Paris", "unit": "c"}
    with cassette(path, mode="replay"):
        assert get_weather("Paris", unit="c") == {"city": "Paris", "unit": "c"}
    assert calls == ["Paris:c"]

    with cassette(path, mode="replay"), pytest.raises(CassetteMismatchError) as caught:
        get_weather("Lyon")
    assert caught.value.index == 0
    assert calls == ["Paris:c"]

    assert get_weather("Lyon") == {"city": "Lyon", "unit": "c"}
    assert calls == ["Paris:c", "Lyon:c"]


def test_async_tool_replays_without_calling_and_mismatches(tmp_path: Path) -> None:
    calls: list[str] = []

    @tool
    async def get_weather(city: str) -> dict[str, int]:
        calls.append(city)
        return {"temp": 18}

    path = tmp_path / "async-weather.jsonl"

    async def scenario() -> None:
        async with cassette(path, mode="record"):
            assert await get_weather("Paris") == {"temp": 18}
        assert calls == ["Paris"]

        async with cassette(path, mode="replay"):
            assert await get_weather(city="Paris") == {"temp": 18}
        assert calls == ["Paris"]

        with pytest.raises(CassetteMismatchError) as caught:
            async with cassette(path, mode="replay"):
                await get_weather("Lyon")
        assert caught.value.index == 0
        assert calls == ["Paris"]

    asyncio.run(scenario())
    assert calls == ["Paris"]
