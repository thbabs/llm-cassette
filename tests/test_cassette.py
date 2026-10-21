from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Awaitable, Callable
from functools import partial, wraps
from pathlib import Path
from typing import Any, cast

import pytest

import llm_cassette
from llm_cassette import (
    CassetteError,
    CassetteMismatchError,
    CassetteNotFoundError,
    cassette,
    current_cassette,
)


def _llm(body: dict[str, Any], provider: str = "openai") -> dict[str, Any]:
    return {"provider": provider, "request": body}


def _tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args}


def _forbid() -> dict[str, Any]:
    raise AssertionError("live call")


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def test_public_api() -> None:
    expected = {
        "Cassette",
        "CassetteError",
        "CassetteMismatchError",
        "CassetteNotFoundError",
        "EventKind",
        "Matcher",
        "Mode",
        "__version__",
        "cassette",
        "current_cassette",
        "normalise_request",
        "request_hash",
        "tool",
    }
    assert set(llm_cassette.__all__) == expected
    for name in expected:
        assert hasattr(llm_cassette, name)
    assert issubclass(CassetteMismatchError, CassetteError)
    assert issubclass(CassetteNotFoundError, CassetteError)


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "round.jsonl"
    request = _llm({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    calls = 0

    def live() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"content": "hello"}

    with cassette(path, mode="once") as session:
        assert session.recording
        assert session.path == path
        assert session.mode == "once"
        assert current_cassette() is session
        assert session.record_or_replay("llm_call", request, live) == {
            "content": "hello"
        }

    assert calls == 1
    assert current_cassette() is None

    with cassette(path, mode="once") as session:
        assert not session.recording
        assert session.record_or_replay("llm_call", request, live) == {
            "content": "hello"
        }

    assert calls == 1
    assert path.read_text(encoding="utf-8").startswith(
        '{"type": "header", "format_version": 1}\n'
    )


def test_sequential_matching_replays_identical_requests_in_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "order.jsonl"
    request = _llm({"model": "gpt-4o"})
    responses = [{"n": 1}, {"n": 2}]

    def live() -> dict[str, int]:
        return responses.pop(0)

    with cassette(path, mode="record") as session:
        assert session.record_or_replay("llm_call", request, live) == {"n": 1}
        assert session.record_or_replay("llm_call", request, live) == {"n": 2}

    with cassette(path, mode="replay") as session:
        assert session.record_or_replay("llm_call", request, _forbid) == {"n": 1}
        assert session.record_or_replay("llm_call", request, _forbid) == {"n": 2}


def test_jsonl_is_readable_stable_and_redacted(tmp_path: Path) -> None:
    path = tmp_path / "cassette.jsonl"
    llm_request = _llm(
        {
            "model": "gpt-4o",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "id": "req_123",
            "timestamp": "2020-01-01T00:00:00Z",
            "request_id": "rid",
            "created": 999,
            "created_at": "x",
            "headers": {
                "Authorization": "Bearer sk-live",
                "X-Request-Id": "abc",
                "Content-Type": "application/json",
            },
            "api_key": "sk-live",
        }
    )
    tool_request = _tool(
        "get_weather",
        {"city": "Paris", "id": "call_1", "api_key": "sk-tool"},
    )

    with cassette(path, mode="record") as session:
        llm_result = session.record_or_replay(
            "llm_call",
            llm_request,
            lambda: {"id": "resp_1", "content": "hello", "created": 123},
        )
        tool_result = session.record_or_replay(
            "tool_call",
            tool_request,
            lambda: {"temp": 18, "id": "obs_1"},
        )

    assert llm_result == {"content": "hello", "created": 123, "id": "resp_1"}
    assert list(llm_result) == ["content", "created", "id"]
    assert tool_result == {"id": "obs_1", "temp": 18}

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.splitlines()
    assert lines[0] == '{"type": "header", "format_version": 1}'
    assert len(lines) == 3
    assert lines[1].startswith('{"type": "llm_call", "provider": "openai", "request": ')
    assert lines[2].startswith('{"type": "tool_call", "name": "get_weather", "args": ')
    llm_event = json.loads(lines[1])
    tool_event = json.loads(lines[2])
    assert list(llm_event) == ["type", "provider", "request", "response"]
    assert list(llm_event["request"]) == [
        "api_key",
        "headers",
        "max_tokens",
        "messages",
        "model",
    ]
    assert list(llm_event["request"]["headers"]) == ["Authorization", "Content-Type"]
    assert list(llm_event["request"]["messages"][0]) == ["content", "role"]
    assert llm_event["request"] == {
        "api_key": "<redacted>",
        "headers": {
            "Authorization": "<redacted>",
            "Content-Type": "application/json",
        },
        "max_tokens": 16,
        "messages": [{"content": "hi", "role": "user"}],
        "model": "gpt-4o",
    }
    assert list(llm_event["response"]) == ["content", "created", "id"]
    assert llm_event["response"] == {"content": "hello", "created": 123, "id": "resp_1"}
    assert list(tool_event) == ["type", "name", "args", "result"]
    assert list(tool_event["args"]) == ["api_key", "city"]
    assert tool_event["args"] == {"api_key": "<redacted>", "city": "Paris"}
    assert list(tool_event["result"]) == ["id", "temp"]
    assert tool_event["result"] == {"id": "obs_1", "temp": 18}
    for secret in ("sk-live", "sk-tool", "Bearer", "req_123", "call_1"):
        assert secret not in text
    assert "resp_1" in text
    assert "obs_1" in text

    replay_llm = _llm(
        {
            "api_key": "sk-other",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-other",
            },
            "max_tokens": 16,
            "messages": [{"content": "hi", "role": "user"}],
            "model": "gpt-4o",
            "id": "different",
            "timestamp": 5,
        }
    )
    with cassette(path, mode="replay") as session:
        assert session.record_or_replay("llm_call", replay_llm, _forbid) == {
            "content": "hello",
            "created": 123,
            "id": "resp_1",
        }
        assert session.record_or_replay(
            "tool_call",
            _tool(
                "get_weather",
                {"city": "Paris", "api_key": "different", "id": "other"},
            ),
            _forbid,
        ) == {"id": "obs_1", "temp": 18}


def test_same_logical_request_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"

    def record(path: Path, body: dict[str, Any]) -> None:
        with cassette(path, mode="record") as session:
            session.record_or_replay(
                "llm_call",
                _llm(body),
                lambda: {"content": "hello", "id": "resp_1"},
            )

    record(
        first,
        {
            "model": "gpt-4o",
            "id": "a",
            "api_key": "sk-a",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    record(
        second,
        {
            "messages": [{"content": "hi", "role": "user"}],
            "api_key": "sk-b",
            "timestamp": 5,
            "model": "gpt-4o",
        },
    )

    assert first.read_bytes() == second.read_bytes()


def test_message_order_changes_the_cassette(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    hello = [{"role": "user", "content": "hi"}, {"role": "user", "content": "yo"}]
    swapped = [{"role": "user", "content": "yo"}, {"role": "user", "content": "hi"}]

    for path, messages in ((first, hello), (second, swapped)):
        with cassette(path, mode="record") as session:
            session.record_or_replay(
                "llm_call",
                _llm({"messages": messages}),
                lambda: {"content": "ok"},
            )

    assert first.read_bytes() != second.read_bytes()


def test_unicode_is_stored_as_characters(tmp_path: Path) -> None:
    path = tmp_path / "unicode.jsonl"
    request = _llm({"prompt": "你好"})

    with cassette(path, mode="record") as session:
        assert session.record_or_replay(
            "llm_call",
            request,
            lambda: {"content": "你好"},
        ) == {"content": "你好"}

    text = path.read_text(encoding="utf-8")
    assert "你好" in text
    assert "\\u" not in text
    with cassette(path, mode="replay") as session:
        assert session.record_or_replay("llm_call", request, _forbid) == {
            "content": "你好"
        }


def test_mismatch_shows_normalised_requests_and_hides_secrets(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.jsonl"
    recorded = _llm(
        {"model": "gpt-4o", "id": "req-recorded-id", "api_key": "sk-secret-a"}
    )
    received = _llm(
        {"model": "gpt-4o-mini", "id": "req-live-id", "api_key": "sk-secret-b"}
    )

    with cassette(path, mode="record") as session:
        session.record_or_replay("llm_call", recorded, lambda: {"content": "hello"})

    with cassette(path, mode="replay") as session:
        with pytest.raises(CassetteMismatchError) as caught:
            session.record_or_replay("llm_call", received, _forbid)
        assert session.record_or_replay("llm_call", recorded, _forbid) == {
            "content": "hello"
        }

    error = caught.value
    assert error.path == path
    assert error.index == 0
    assert error.expected == {
        "type": "llm_call",
        "provider": "openai",
        "request": {"api_key": "<redacted>", "model": "gpt-4o"},
    }
    assert error.received == {
        "type": "llm_call",
        "provider": "openai",
        "request": {"api_key": "<redacted>", "model": "gpt-4o-mini"},
    }
    message = str(error)
    assert "expected:" in message
    assert "received:" in message
    assert "gpt-4o-mini" in message
    assert "sk-secret-a" not in message
    assert "sk-secret-b" not in message
    assert "req-recorded-id" not in message
    assert "req-live-id" not in message


def test_wrong_order_does_not_match_a_later_interaction(tmp_path: Path) -> None:
    path = tmp_path / "order.jsonl"
    first = _llm({"model": "a"})
    second = _llm({"model": "b"})

    with cassette(path, mode="record") as session:
        session.record_or_replay("llm_call", first, lambda: {"n": 1})
        session.record_or_replay("llm_call", second, lambda: {"n": 2})

    with (
        cassette(path, mode="replay") as session,
        pytest.raises(CassetteMismatchError) as caught,
    ):
        session.record_or_replay("llm_call", second, _forbid)

    assert _mapping(caught.value.expected)["request"] == {"model": "a"}
    assert _mapping(caught.value.received)["request"] == {"model": "b"}


def test_exhausted_cassette_reports_the_received_request(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    with cassette(path, mode="record"):
        pass

    with (
        cassette(path, mode="replay") as session,
        pytest.raises(CassetteMismatchError) as caught,
    ):
        session.record_or_replay("llm_call", _llm({"model": "gpt-4o"}), _forbid)

    assert caught.value.expected is None
    assert caught.value.received == {
        "type": "llm_call",
        "provider": "openai",
        "request": {"model": "gpt-4o"},
    }
    assert caught.value.index == 0
    assert "no further recorded interactions" in str(caught.value)


def test_kind_provider_and_tool_name_are_part_of_the_match(tmp_path: Path) -> None:
    path = tmp_path / "identity.jsonl"
    with cassette(path, mode="record") as session:
        session.record_or_replay(
            "llm_call",
            _llm({"model": "gpt-4o"}),
            lambda: {"n": 1},
        )
        session.record_or_replay(
            "tool_call",
            _tool("search", {"q": "cats"}),
            lambda: {"ok": True},
        )

    with (
        cassette(path, mode="replay") as session,
        pytest.raises(CassetteMismatchError) as provider_error,
    ):
        session.record_or_replay(
            "llm_call",
            _llm({"model": "gpt-4o"}, provider="anthropic"),
            _forbid,
        )
        assert _mapping(provider_error.value.expected)["provider"] == "openai"
        assert _mapping(provider_error.value.received)["provider"] == "anthropic"

    with (
        cassette(path, mode="replay") as session,
        pytest.raises(CassetteMismatchError) as kind_error,
    ):
        session.record_or_replay(
            "tool_call",
            _tool("search", {"model": "gpt-4o"}),
            _forbid,
        )
        assert _mapping(kind_error.value.expected)["type"] == "llm_call"
        assert _mapping(kind_error.value.received)["type"] == "tool_call"

    with cassette(path, mode="replay") as session:
        session.record_or_replay("llm_call", _llm({"model": "gpt-4o"}), _forbid)
        with pytest.raises(CassetteMismatchError) as name_error:
            session.record_or_replay(
                "tool_call",
                _tool("other", {"q": "cats"}),
                _forbid,
            )
        assert _mapping(name_error.value.expected)["name"] == "search"
        assert _mapping(name_error.value.received)["name"] == "other"


def test_user_id_is_matched_and_plain_id_is_not(tmp_path: Path) -> None:
    path = tmp_path / "ids.jsonl"
    with cassette(path, mode="record") as session:
        session.record_or_replay(
            "tool_call",
            _tool("load", {"user_id": "a", "id": "1"}),
            lambda: {"ok": True},
        )

    with cassette(path, mode="replay") as session:
        assert session.record_or_replay(
            "tool_call",
            _tool("load", {"user_id": "a", "id": "2"}),
            _forbid,
        ) == {"ok": True}

    with cassette(path, mode="replay") as session, pytest.raises(CassetteMismatchError):
        session.record_or_replay(
            "tool_call",
            _tool("load", {"user_id": "b", "id": "2"}),
            _forbid,
        )


def test_strict_requires_every_interaction_to_be_replayed(tmp_path: Path) -> None:
    path = tmp_path / "strict.jsonl"
    first = _llm({"model": "a"})
    second = _llm({"model": "b"})
    with cassette(path, mode="record", strict=True) as session:
        session.record_or_replay("llm_call", first, lambda: {"n": 1})
        session.record_or_replay("llm_call", second, lambda: {"n": 2})

    with cassette(path, mode="replay", strict=True) as session:
        assert session.record_or_replay("llm_call", first, _forbid) == {"n": 1}
        assert session.record_or_replay("llm_call", second, _forbid) == {"n": 2}

    with (
        pytest.raises(
            CassetteMismatchError,
            match="2 unconsumed interactions",
        ) as caught,
        cassette(path, mode="replay", strict=True),
    ):
        pass
    assert caught.value.received is None
    assert _mapping(caught.value.expected)["request"] == {"model": "a"}
    assert current_cassette() is None

    with (
        pytest.raises(
            CassetteMismatchError,
            match="1 unconsumed interaction",
        ) as caught,
        cassette(path, mode="replay", strict=True) as session,
    ):
        session.record_or_replay("llm_call", first, _forbid)
    assert _mapping(caught.value.expected)["request"] == {"model": "b"}
    assert caught.value.index == 1

    with cassette(path, mode="replay") as session:
        pass

    with (
        pytest.raises(CassetteMismatchError, match="request does not match"),
        cassette(path, mode="replay", strict=True) as session,
    ):
        session.record_or_replay("llm_call", second, _forbid)


def test_custom_matcher_replaces_the_hash_comparison(tmp_path: Path) -> None:
    path = tmp_path / "matcher.jsonl"
    recorded = _llm(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    )
    other_model = _llm(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    )
    with cassette(path, mode="record") as session:
        session.record_or_replay("llm_call", recorded, lambda: {"content": "hello"})

    def same_messages(expected: object, received: object) -> bool:
        assert isinstance(expected, dict)
        assert isinstance(received, dict)
        same = expected["request"]["messages"] == received["request"]["messages"]
        return same is True

    with cassette(path, mode="replay", matcher=same_messages) as session:
        assert session.record_or_replay("llm_call", other_model, _forbid) == {
            "content": "hello"
        }

    with cassette(path, mode="replay") as session, pytest.raises(CassetteMismatchError):
        session.record_or_replay("llm_call", other_model, _forbid)

    def reject(expected: object, received: object) -> bool:
        return False

    with (
        cassette(path, mode="replay", matcher=reject) as session,
        pytest.raises(CassetteMismatchError) as caught,
    ):
        session.record_or_replay("llm_call", recorded, _forbid)
    request = _mapping(caught.value.expected)["request"]
    assert isinstance(request, dict)
    assert request["model"] == "gpt-4o"


def test_matcher_sees_normalised_copies(tmp_path: Path) -> None:
    path = tmp_path / "copies.jsonl"
    request = _llm({"model": "gpt-4o", "id": "req_1", "api_key": "sk-secret"})
    with cassette(path, mode="record") as session:
        session.record_or_replay("llm_call", request, lambda: {"n": 1})
        session.record_or_replay("llm_call", request, lambda: {"n": 2})

    seen: list[dict[str, Any]] = []

    def matcher(expected: object, received: object) -> bool:
        assert isinstance(expected, dict)
        assert isinstance(received, dict)
        seen.append(cast(dict[str, Any], expected).copy())
        request_body = expected["request"]
        assert isinstance(request_body, dict)
        assert "id" not in request_body
        assert request_body["api_key"] == "<redacted>"
        assert "sk-secret" not in str(expected)
        expected.clear()
        return True

    with cassette(path, mode="replay", matcher=matcher) as session:
        assert session.record_or_replay(
            "llm_call",
            _llm({"model": "gpt-4o", "id": "other", "api_key": "sk-other"}),
            _forbid,
        ) == {"n": 1}
        assert session.record_or_replay("llm_call", request, _forbid) == {"n": 2}

    assert seen[0]["request"]["model"] == "gpt-4o"
    assert seen[1]["request"]["model"] == "gpt-4o"


def test_strict_still_requires_consumption_when_a_matcher_is_set(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-matcher.jsonl"
    with cassette(path, mode="record") as session:
        session.record_or_replay("llm_call", _llm({"model": "a"}), lambda: {"n": 1})
        session.record_or_replay("llm_call", _llm({"model": "b"}), lambda: {"n": 2})

    def always(expected: object, received: object) -> bool:
        return True

    with (
        pytest.raises(CassetteMismatchError, match="1 unconsumed interaction"),
        cassette(path, mode="replay", strict=True, matcher=always) as session,
    ):
        session.record_or_replay("llm_call", _llm({"model": "nope"}), _forbid)


def test_modes(tmp_path: Path) -> None:
    path = tmp_path / "modes.jsonl"
    request = _llm({"model": "gpt-4o"})
    calls = 0

    def live() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"content": "v1"}

    with pytest.raises(CassetteNotFoundError) as missing, cassette(path, mode="replay"):
        raise AssertionError("body should not run")
    assert missing.value.path == path

    with cassette(path, mode="record") as session:
        assert session.recording
        assert session.record_or_replay("llm_call", request, live) == {"content": "v1"}
    assert calls == 1
    original = path.read_text(encoding="utf-8")

    with cassette(path, mode="replay") as session:
        assert not session.recording
        assert session.record_or_replay("llm_call", request, live) == {"content": "v1"}
    assert calls == 1
    assert path.read_text(encoding="utf-8") == original

    def updated() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"content": "v2"}

    with cassette(path, mode="record") as session:
        assert session.record_or_replay("llm_call", request, updated) == {
            "content": "v2"
        }
    assert calls == 2
    with cassette(path, mode="replay") as session:
        assert session.record_or_replay("llm_call", request, _forbid) == {
            "content": "v2"
        }

    path.write_text("old", encoding="utf-8")
    with (
        pytest.raises(RuntimeError, match="fail"),
        cassette(path, mode="record") as session,
    ):
        session.record_or_replay("llm_call", request, lambda: {"content": "new"})
        raise RuntimeError("fail")
    assert path.read_text(encoding="utf-8") == "old"

    nested = tmp_path / "new" / "once.jsonl"
    with (
        pytest.raises(RuntimeError, match="fail"),
        cassette(nested, mode="once") as session,
    ):
        session.record_or_replay("llm_call", request, lambda: {"content": "new"})
        raise RuntimeError("fail")
    assert not nested.exists()
    assert not nested.parent.exists()

    sentinel = object()
    path.write_text("garbage", encoding="utf-8")
    with cassette(path, mode="none", strict=True) as session:
        assert session.mode == "none"
        assert not session.recording
        assert (
            session.record_or_replay("llm_call", request, lambda: sentinel) is sentinel
        )
    assert path.read_text(encoding="utf-8") == "garbage"

    with (
        pytest.raises(CassetteError, match="not a file"),
        cassette(tmp_path, mode="replay"),
    ):
        pass
    with pytest.raises(ValueError, match="unknown cassette mode"):
        cassette(path, cast(Any, "sometimes"))
    with pytest.raises(TypeError, match="filesystem path"):
        cassette(cast(Any, 12))
    with pytest.raises(TypeError, match="callable"):
        cassette(path, matcher=cast(Any, "nope"))


def test_zero_calls_write_a_header_only_cassette(tmp_path: Path) -> None:
    path = tmp_path / "header.jsonl"
    with cassette(path, mode="record"):
        pass
    assert path.read_text(encoding="utf-8") == (
        '{"type": "header", "format_version": 1}\n'
    )
    with cassette(path, mode="replay", strict=True):
        pass


def test_returned_values_are_copies(tmp_path: Path) -> None:
    path = tmp_path / "copy.jsonl"
    request = _llm({"model": "gpt-4o"})
    with cassette(path, mode="record") as session:
        value = session.record_or_replay(
            "llm_call",
            request,
            lambda: {"content": "hello"},
        )
        value["content"] = "mutated"
    assert "mutated" not in path.read_text(encoding="utf-8")
    with cassette(path, mode="replay") as session:
        replayed = session.record_or_replay("llm_call", request, _forbid)
        replayed["content"] = "mutated"
    with cassette(path, mode="replay") as session:
        assert session.record_or_replay("llm_call", request, _forbid) == {
            "content": "hello"
        }


def test_response_secrets_are_redacted(tmp_path: Path) -> None:
    path = tmp_path / "response.jsonl"
    request = _llm({"model": "gpt-4o"})

    def live() -> dict[str, str]:
        return {"api_key": "sk-response", "content": "hello"}

    with cassette(path, mode="record") as session:
        assert session.record_or_replay("llm_call", request, live) == {
            "api_key": "<redacted>",
            "content": "hello",
        }
    assert "sk-response" not in path.read_text(encoding="utf-8")
    with cassette(path, mode="replay") as session:
        assert session.record_or_replay("llm_call", request, _forbid) == {
            "api_key": "<redacted>",
            "content": "hello",
        }


def test_scalar_and_tuple_results_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "shapes.jsonl"

    def pair() -> object:
        return (1, 2)

    with cassette(path, mode="record") as session:
        assert (
            session.record_or_replay("tool_call", _tool("label", {}), lambda: "plain")
            == "plain"
        )
        recorded: object = session.record_or_replay(
            "tool_call",
            _tool("pair", {}),
            pair,
        )
        assert recorded == [1, 2]

    with cassette(path, mode="replay") as session:
        replayed_label: object = session.record_or_replay(
            "tool_call",
            _tool("label", {}),
            _forbid,
        )
        assert replayed_label == "plain"
        replayed: object = session.record_or_replay(
            "tool_call",
            _tool("pair", {}),
            _forbid,
        )
        assert replayed == [1, 2]


def test_invalid_requests_do_not_call_live(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    calls = 0

    def live() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"ok": "yes"}

    with (
        pytest.raises(CassetteError, match="provider"),
        cassette(path, mode="record") as session,
    ):
        session.record_or_replay("llm_call", {"request": {}}, live)
    with (
        pytest.raises(CassetteError, match="unexpected keys"),
        cassette(path, mode="record") as session,
    ):
        session.record_or_replay(
            "llm_call",
            {"provider": "openai", "request": {}, "extra": 1},
            live,
        )
    with (
        pytest.raises(CassetteError, match="unknown event kind"),
        cassette(path, mode="record") as session,
    ):
        session.record_or_replay(cast(Any, "nope"), _llm({}), live)
    with (
        pytest.raises(CassetteError, match="cannot record"),
        cassette(path, mode="record") as session,
    ):
        session.record_or_replay(
            "llm_call",
            _llm({"model": "gpt-4o"}),
            object,
        )
    assert calls == 0
    assert not path.exists()


def test_rejects_invalid_cassettes(tmp_path: Path) -> None:
    cases = [
        ("", "empty"),
        ("{", "invalid JSON"),
        ('{"type": "header", "format_version": 2}\n', "unsupported"),
        ('{"type": "header", "format_version": true}\n', "unsupported"),
        ("NaN\n", "finite"),
        ("[]\n", "JSON object"),
        (
            '{"type": "llm_call", "provider": "openai", "request": {}, '
            '"response": {}}\n',
            "header",
        ),
        (
            '{"type": "header", "format_version": 1}\n'
            '{"type": "llm_call", "provider": "openai", "request": {}}\n',
            "missing",
        ),
        (
            '{"type": "header", "format_version": 1}\n{"type": "other"}\n',
            "unknown event type",
        ),
        (
            '{"type": "header", "format_version": 1}\n'
            '{"type": "llm_call", "provider": "", "request": {}, "response": {}}\n',
            "non-empty",
        ),
    ]
    for index, (content, message) in enumerate(cases):
        path = tmp_path / f"{index}.jsonl"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(CassetteError, match=message), cassette(path, mode="replay"):
            pass

    empty = tmp_path / "once-empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(CassetteError, match="empty"), cassette(empty, mode="once"):
        pass


def test_hand_written_cassette_ignores_key_order_and_volatile_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hand.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"format_version": 1, "type": "header"}',
                "",
                (
                    '{"response": {"api_key": "sk-from-file", "content": "hello", '
                    '"id": "resp"}, "request": {"id": "req", "model": "gpt-4o"}, '
                    '"provider": "openai", "type": "llm_call"}'
                ),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with cassette(path, mode="replay") as session:
        result = session.record_or_replay(
            "llm_call",
            _llm({"model": "gpt-4o", "timestamp": 9}),
            _forbid,
        )
    assert result == {
        "api_key": "<redacted>",
        "content": "hello",
        "id": "resp",
    }
    assert "sk-from-file" in path.read_text(encoding="utf-8")
    assert "sk-from-file" not in str(result)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text(
        '{"type": "header", "format_version": 1}\n'
        "\n"
        '{"type": "llm_call", "provider": "openai", "request": {"model": "gpt-4o"}, '
        '"response": {"content": "hello"}}\n',
        encoding="utf-8",
    )
    with cassette(path, mode="replay") as session:
        assert session.record_or_replay(
            "llm_call",
            _llm({"model": "gpt-4o"}),
            _forbid,
        ) == {"content": "hello"}


def test_closed_cassette_and_nested_context(tmp_path: Path) -> None:
    outer = tmp_path / "outer.jsonl"
    inner = tmp_path / "inner.jsonl"
    request = _llm({"model": "gpt-4o"})

    with cassette(outer, mode="none") as first:
        assert current_cassette() is first
        with (
            pytest.raises(RuntimeError, match="inner"),
            cassette(inner, mode="none") as second,
        ):
            assert current_cassette() is second
            raise RuntimeError("inner")
        assert current_cassette() is first
        held = first
    assert current_cassette() is None
    with pytest.raises(CassetteError, match="closed"):
        held.record_or_replay("llm_call", request, lambda: {"content": "hello"})


def test_sync_decorator_records_once(tmp_path: Path) -> None:
    path = tmp_path / "decorated.jsonl"
    request = _llm({"model": "gpt-4o"})
    calls = 0

    def live() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"content": "hello"}

    @cassette(path, mode="once")
    def run() -> dict[str, Any]:
        session = current_cassette()
        assert session is not None
        return session.record_or_replay("llm_call", request, live)

    assert run.__name__ == "run"
    assert run() == {"content": "hello"}
    assert run() == {"content": "hello"}
    assert calls == 1


def test_async_record_replay_and_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "async.jsonl"

    async def scenario() -> None:
        request = _llm({"model": "gpt-4o"})

        async def live() -> dict[str, str]:
            return {"content": "hello"}

        async def fail() -> dict[str, str]:
            raise AssertionError("live call")

        async with cassette(path, mode="record") as session:
            assert current_cassette() is session
            llm = await session.record_or_replay("llm_call", request, live)
            tool = session.record_or_replay(
                "tool_call",
                _tool("lookup", {"q": "x"}),
                lambda: {"ok": True},
            )
        assert llm == {"content": "hello"}
        assert tool == {"ok": True}

        async with cassette(path, mode="replay") as session:
            assert await session.record_or_replay("llm_call", request, fail) == {
                "content": "hello"
            }
            assert session.record_or_replay(
                "tool_call",
                _tool("lookup", {"q": "x"}),
                _forbid,
            ) == {"ok": True}
            with pytest.raises(CassetteMismatchError):
                await session.record_or_replay(
                    "llm_call",
                    _llm({"model": "other"}),
                    fail,
                )

        calls = 0

        async def counted() -> dict[str, str]:
            nonlocal calls
            calls += 1
            return {"content": "hello"}

        once_path = tmp_path / "once.jsonl"

        @cassette(once_path, mode="once")
        async def run() -> dict[str, Any]:
            session = current_cassette()
            assert session is not None
            return await session.record_or_replay("llm_call", request, counted)

        assert run.__name__ == "run"
        assert await run() == {"content": "hello"}
        assert await run() == {"content": "hello"}
        assert calls == 1

        async def live_with(extra: str) -> dict[str, str]:
            return {"content": extra}

        call: Callable[[], Awaitable[dict[str, str]]] = partial(live_with, "hello")
        partial_path = tmp_path / "partial.jsonl"
        async with cassette(partial_path, mode="record") as session:
            assert await session.record_or_replay("llm_call", request, call) == {
                "content": "hello"
            }

        @wraps(live)
        def wrapped() -> Awaitable[dict[str, str]]:
            return live()

        wrapped_path = tmp_path / "wrapped.jsonl"
        async with cassette(wrapped_path, mode="record") as session:
            assert await session.record_or_replay("llm_call", request, wrapped) == {
                "content": "hello"
            }

    asyncio.run(scenario())


def test_unawaited_async_call_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "unawaited.jsonl"

    async def live() -> dict[str, str]:
        return {"content": "hi"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with (
            pytest.raises(CassetteError, match="not awaited"),
            cassette(path, mode="record") as session,
        ):
            session.record_or_replay("llm_call", _llm({"model": "gpt-4o"}), live)
    assert not path.exists()
