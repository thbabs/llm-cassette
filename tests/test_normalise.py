from __future__ import annotations

import copy
import hashlib
import json

import pytest

from llm_cassette import CassetteError, normalise_request, request_hash


def test_drops_volatile_fields_and_redacts_secrets() -> None:
    value = {
        "headers": {
            "Authorization": "Bearer sk-secret",
            "X-Request-Id": "abc",
            "Content-Type": "application/json",
        },
        "messages": [{"role": "user", "content": "key sk-secret"}],
        "max_tokens": 10,
        "max_completion_tokens": 5,
        "prompt_tokens": 3,
        "user_id": "u1",
        "id": "req_1",
        "created": 1,
        "created_at": "now",
        "createdAt": "also",
        "updated_at": "then",
        "started_at": "start",
        "completed_at": "end",
        "timestamp": 2,
        "request_id": "rid",
        "requestId": "rid-2",
        "X-Request-Id": "headerish",
        "api_key": "sk-1",
        "apiKey": "sk-2",
        "token": "abc",
        "session_token": "xyz",
    }

    assert normalise_request(value) == {
        "headers": {
            "Authorization": "<redacted>",
            "Content-Type": "application/json",
        },
        "messages": [{"role": "user", "content": "key sk-secret"}],
        "max_tokens": 10,
        "max_completion_tokens": 5,
        "prompt_tokens": 3,
        "user_id": "u1",
        "api_key": "<redacted>",
        "apiKey": "<redacted>",
        "token": "<redacted>",
        "session_token": "<redacted>",
    }


def test_hash_ignores_key_order_secrets_and_volatile_fields() -> None:
    left = {
        "model": "gpt-4o",
        "id": "1",
        "api_key": "sk-1",
        "timestamp": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    right = {
        "messages": [{"content": "hi", "role": "user"}],
        "timestamp": 2,
        "api_key": "sk-2",
        "model": "gpt-4o",
        "request_id": "r",
    }

    assert normalise_request(left) == {
        "model": "gpt-4o",
        "api_key": "<redacted>",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert normalise_request(right) == normalise_request(left)
    assert request_hash(left) == request_hash(right)
    assert request_hash(left) != request_hash({"model": "gpt-4o-mini"})
    assert request_hash([1, 2]) != request_hash([2, 1])
    assert request_hash({"provider": "openai"}) != request_hash(
        {"provider": "anthropic"}
    )


def test_request_hash_is_sha256_of_canonical_json() -> None:
    value = {"b": 1, "a": "你好"}
    encoded = json.dumps(
        {"a": "你好", "b": 1},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    assert request_hash(value) == expected
    assert request_hash({"a": "你好", "b": 1}) == expected
    assert request_hash({"n": 1.5}) == hashlib.sha256(b'{"n":1.5}').hexdigest()


def test_normalise_is_idempotent_and_does_not_mutate_input() -> None:
    original = {"id": "1", "nested": {"api_key": "sk", "n": 2}, "b": 1, "a": 2}
    snapshot = copy.deepcopy(original)

    once = normalise_request(original)

    assert original == snapshot
    assert normalise_request(once) == once
    order = normalise_request({"b": 1, "a": 2})
    assert isinstance(order, dict)
    assert list(order) == ["b", "a"]
    assert normalise_request((1, {"id": "x", "n": 2})) == [1, {"n": 2}]


def test_rejects_values_that_are_not_json() -> None:
    with pytest.raises(CassetteError, match="cannot record"):
        normalise_request({1, 2})
    with pytest.raises(CassetteError, match="finite"):
        normalise_request({"a": float("nan")})
    with pytest.raises(CassetteError, match="strings"):
        normalise_request({1: "a"})
    with pytest.raises(CassetteError, match="cannot record"):
        normalise_request(object())


def test_redacts_header_and_secret_suffixes() -> None:
    assert normalise_request(
        {
            "headers": {"authorization": "secret", "x-api-key": "sk"},
            "openai_api_key": "sk",
            "secret_key": "sk",
            "user_password": "pw",
        }
    ) == {
        "headers": {"authorization": "<redacted>", "x-api-key": "<redacted>"},
        "openai_api_key": "<redacted>",
        "secret_key": "<redacted>",
        "user_password": "<redacted>",
    }
    assert request_hash({"apiKey": "a"}) == request_hash({"apiKey": "b"})
