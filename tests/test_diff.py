from __future__ import annotations

import json
import subprocess
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest

from llm_cassette.cli import main
from llm_cassette.format import write_cassette


def _llm(
    content: str,
    *,
    model: str = "gpt-4o",
    response_id: str = "resp_1",
) -> dict[str, Any]:
    return {
        "type": "llm_call",
        "provider": "openai",
        "request": {"model": model},
        "response": {"content": content, "id": response_id},
    }


def _tool(city: str = "Paris", *, temp: int = 18) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "name": "get_weather",
        "args": {"city": city},
        "result": {"temp": temp},
    }


def _write(path: Path, events: list[dict[str, Any]]) -> None:
    write_cassette(path, events)


def test_identical_cassettes_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_llm("hello"), _tool()])
    text = left.read_text(encoding="utf-8")
    original = (
        '{"type": "llm_call", "provider": "openai", '
        '"request": {"model": "gpt-4o"}, '
        '"response": {"content": "hello", "id": "resp_1"}}'
    )
    reordered = (
        '{"response": {"id": "resp_1", "content": "hello"}, '
        '"request": {"model": "gpt-4o"}, '
        '"provider": "openai", "type": "llm_call"}'
    )
    swapped = text.replace(original, reordered)
    assert swapped != text
    right.write_text(swapped.replace("\n", "\n\n", 1), encoding="utf-8")

    assert main(["diff", str(left), str(right)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == "cassettes are identical (2 events)\n"


def test_request_ids_do_not_count_as_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_llm("hello")])
    event = json.loads(left.read_text(encoding="utf-8").splitlines()[1])
    event["request"]["id"] = "req_a"
    event["request"]["timestamp"] = "2020-01-01T00:00:00Z"
    lines = left.read_text(encoding="utf-8").splitlines()
    lines[1] = json.dumps(event)
    right.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert main(["diff", str(left), str(right)]) == 0
    assert capsys.readouterr().out == "cassettes are identical (1 event)\n"


def test_added_event_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_llm("hello")])
    _write(right, [_llm("hello"), _tool()])

    assert main(["diff", str(left), str(right)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "1 added" in captured.out
    assert "added tool_call 2 (get_weather)" in captured.out
    assert '"city": "Paris"' in captured.out
    assert "removed" not in captured.out
    assert "changed" not in captured.out


def test_removed_event_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_llm("hello"), _tool()])
    _write(right, [_llm("hello")])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    assert "1 removed" in output
    assert "removed tool_call 2 (get_weather)" in output
    assert "added" not in output
    assert "changed" not in output


def test_inserted_event_keeps_the_tail_aligned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_llm("hello"), _llm("goodbye", response_id="resp_2")])
    _write(right, [_llm("hello"), _tool(), _llm("goodbye", response_id="resp_2")])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    assert "1 added" in output
    assert "added tool_call 2 (get_weather)" in output
    assert "changed" not in output
    assert "removed" not in output


def test_changed_request_shows_unified_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "old.jsonl"
    right = tmp_path / "new.jsonl"
    _write(left, [_llm("hello", model="gpt-4o")])
    _write(right, [_llm("hello", model="gpt-4o-mini")])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    assert "1 changed" in output
    assert "changed llm_call 1 (openai)" in output
    assert "\nrequest:\n" in output
    assert "\nresponse:\n" not in output
    assert f"--- {left} request" in output
    assert f"+++ {right} request" in output
    assert '-    "model": "gpt-4o"' in output
    assert '+    "model": "gpt-4o-mini"' in output
    assert "@@" in output


def test_changed_response_shows_unified_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "old.jsonl"
    right = tmp_path / "new.jsonl"
    _write(left, [_llm("hello")])
    _write(right, [_llm("goodbye")])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    assert "changed llm_call 1 (openai)" in output
    assert "\nresponse:\n" in output
    assert "\nrequest:\n" not in output
    assert f"--- {left} response" in output
    assert f"+++ {right} response" in output
    assert '-  "content": "hello",' in output
    assert '+  "content": "goodbye",' in output


def test_response_ids_are_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "old.jsonl"
    right = tmp_path / "new.jsonl"
    _write(left, [_llm("hello", response_id="resp_1")])
    _write(right, [_llm("hello", response_id="resp_2")])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    assert "\nresponse:\n" in output
    assert "\nrequest:\n" not in output
    assert '-  "id": "resp_1"' in output
    assert '+  "id": "resp_2"' in output


def test_changed_tool_result_is_the_response(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "old.jsonl"
    right = tmp_path / "new.jsonl"
    _write(left, [_tool(temp=18)])
    _write(right, [_tool(temp=19)])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    assert "changed tool_call 1 (get_weather)" in output
    assert "\nresponse:\n" in output
    assert "\nrequest:\n" not in output
    assert '-  "temp": 18' in output
    assert '+  "temp": 19' in output


def test_request_and_response_can_both_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "old.jsonl"
    right = tmp_path / "new.jsonl"
    _write(left, [_llm("hello", model="gpt-4o")])
    _write(right, [_llm("goodbye", model="gpt-4o-mini")])

    assert main(["diff", str(left), str(right)]) == 1
    output = capsys.readouterr().out
    request_at = output.index("\nrequest:\n")
    response_at = output.index("\nresponse:\n")
    assert request_at < response_at
    assert '-    "model": "gpt-4o"' in output
    assert '+  "content": "goodbye",' in output


def test_diff_redacts_secrets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    secret = "sk-live-secret"
    _write(
        left,
        [
            {
                "type": "llm_call",
                "provider": "openai",
                "request": {"model": "gpt-4o", "api_key": secret},
                "response": {"content": "hello", "api_key": secret},
            }
        ],
    )
    _write(right, [_llm("goodbye")])

    assert main(["diff", str(left), str(right)]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "<redacted>" in captured.out


def test_missing_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    present = tmp_path / "present.jsonl"
    missing = tmp_path / "missing.jsonl"
    _write(present, [_llm("hello")])

    assert main(["diff", str(missing), str(present)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cassette not found: {missing}" in captured.err

    assert main(["diff", str(present), str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cassette not found: {missing}" in captured.err


def test_invalid_cassette_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_llm("hello")])
    right.write_text("{not json}\n", encoding="utf-8")

    assert main(["diff", str(left), str(right)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid JSON" in captured.err
    assert str(right) in captured.err


def test_empty_cassettes_are_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [])
    _write(right, [])

    assert main(["diff", str(left), str(right)]) == 0
    assert capsys.readouterr().out == "cassettes are identical (0 events)\n"


def test_console_script_entry_point() -> None:
    matches = [
        entry
        for entry in entry_points(group="console_scripts")
        if entry.name == "llm-cassette"
    ]
    assert [entry.value for entry in matches] == ["llm_cassette.cli:main"]


def test_console_script_exit_codes(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    missing = tmp_path / "missing.jsonl"
    _write(left, [_llm("hello")])
    _write(right, [_llm("hello")])

    identical = subprocess.run(
        ["llm-cassette", "diff", str(left), str(right)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert identical.returncode == 0
    assert "identical" in identical.stdout
    assert identical.stderr == ""

    _write(right, [_llm("hello"), _tool()])
    differed = subprocess.run(
        ["llm-cassette", "diff", str(left), str(right)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert differed.returncode == 1
    assert "added tool_call 2 (get_weather)" in differed.stdout

    failed = subprocess.run(
        ["llm-cassette", "diff", str(missing), str(left)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert "cassette not found" in failed.stderr
    assert failed.stdout == ""
