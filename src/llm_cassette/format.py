"""JSONL cassette files."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_cassette.errors import CassetteError
from llm_cassette.normalise import ensure_json
from llm_cassette.types import JSONValue

FORMAT_VERSION = 1
_HEADER_KEYS = ("type", "format_version")
_LLM_KEYS = ("type", "provider", "request", "response")
_TOOL_KEYS = ("type", "name", "args", "result")
_LLM_REQUIRED = ("provider", "request", "response")
_TOOL_REQUIRED = ("name", "args", "result")


def read_events(path: Path) -> list[dict[str, JSONValue]]:
    """Read interaction events from a cassette file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise CassetteError(f"{path}: cassette is empty")
    header = _load_line(lines[0], path=path, number=1)
    _validate_header(header, path=path, line=1)
    events: list[dict[str, JSONValue]] = []
    for number, line in enumerate(lines[1:], start=2):
        if line.strip() == "":
            continue
        value = _load_line(line, path=path, number=number)
        events.append(_validate_event(value, path=path, line=number))
    return events


def write_cassette(path: Path, events: Sequence[Mapping[str, JSONValue]]) -> None:
    """Atomically replace ``path`` with a cassette containing ``events``."""

    header = _ordered(
        {"type": "header", "format_version": FORMAT_VERSION},
        _HEADER_KEYS,
    )
    lines = [_dumps(header)]
    for event in events:
        kind = event.get("type")
        if kind == "llm_call":
            keys = _LLM_KEYS
        elif kind == "tool_call":
            keys = _TOOL_KEYS
        else:
            raise CassetteError(f"cannot write event type {kind!r}")
        lines.append(_dumps(_ordered(event, keys)))
    payload = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sorted_json(value: JSONValue) -> JSONValue:
    """Recursively sort object keys and leave list order unchanged."""

    if isinstance(value, dict):
        return {key: sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sorted_json(item) for item in value]
    return value


def _load_line(text: str, *, path: Path, number: int) -> JSONValue:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CassetteError(f"{path}: line {number}: invalid JSON") from exc
    try:
        return ensure_json(raw)
    except CassetteError as exc:
        raise CassetteError(f"{path}: line {number}: {exc}") from exc


def _validate_header(value: JSONValue, *, path: Path, line: int) -> None:
    header = _as_object(value, path=path, line=line)
    if header.get("type") != "header":
        raise CassetteError(f"{path}: line {line}: missing cassette header")
    version = header.get("format_version")
    if not _is_exact_int(version) or version != FORMAT_VERSION:
        raise CassetteError(
            f"{path}: line {line}: unsupported cassette format version {version!r}"
        )


def _validate_event(value: JSONValue, *, path: Path, line: int) -> dict[str, JSONValue]:
    event = _as_object(value, path=path, line=line)
    kind = event.get("type")
    if kind == "llm_call":
        _require_keys(event, _LLM_REQUIRED, path=path, line=line, kind=kind)
        _require_name(event.get("provider"), path=path, line=line, field="provider")
        return event
    if kind == "tool_call":
        _require_keys(event, _TOOL_REQUIRED, path=path, line=line, kind=kind)
        _require_name(event.get("name"), path=path, line=line, field="name")
        return event
    raise CassetteError(f"{path}: line {line}: unknown event type {kind!r}")


def _require_keys(
    event: Mapping[str, JSONValue],
    keys: tuple[str, ...],
    *,
    path: Path,
    line: int,
    kind: str,
) -> None:
    missing = [key for key in keys if key not in event]
    if missing:
        joined = ", ".join(missing)
        raise CassetteError(f"{path}: line {line}: {kind} is missing {joined}")


def _require_name(value: object, *, path: Path, line: int, field: str) -> None:
    if isinstance(value, str) and value != "":
        return
    raise CassetteError(f"{path}: line {line}: {field} must be a non-empty string")


def _as_object(value: JSONValue, *, path: Path, line: int) -> dict[str, JSONValue]:
    if isinstance(value, dict):
        return value
    raise CassetteError(f"{path}: line {line}: expected a JSON object")


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ordered(
    value: Mapping[str, JSONValue], keys: tuple[str, ...]
) -> dict[str, JSONValue]:
    sorted_value = {key: sorted_json(item) for key, item in value.items()}
    ordered: dict[str, JSONValue] = {}
    for key in keys:
        if key in sorted_value:
            ordered[key] = sorted_value[key]
    for key in sorted(sorted_value):
        if key not in ordered:
            ordered[key] = sorted_value[key]
    return ordered


def _dumps(value: Mapping[str, JSONValue]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(", ", ": "),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CassetteError("could not write cassette JSON") from exc
