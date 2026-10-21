from __future__ import annotations

import pytest
from openai import OpenAI

from llm_cassette import cassette
from llm_cassette.openai import install
from openai_agent import CASSETTE, ask, calls


def _reject() -> str:
    raise AssertionError("api key requested")


def test_weather_agent_replays_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "OPENAI_BASE_URL",
        "OPENAI_WEBHOOK_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    calls.clear()
    install()
    client = OpenAI(api_key=_reject, timeout=1.0)

    with cassette(CASSETTE, mode="replay"):
        answer = ask(client, "Paris")

    assert answer == "Paris is 18 degrees."
    assert calls == []
    text = CASSETTE.read_text(encoding="utf-8")
    assert "sk-" not in text
    assert "Bearer" not in text
    assert text.startswith('{"type": "header", "format_version": 1}\n')
