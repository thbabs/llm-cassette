"""Compare cassette files event by event."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_cassette.errors import CassetteError, CassetteNotFoundError
from llm_cassette.format import read_events
from llm_cassette.normalise import normalise_request, redact_secrets
from llm_cassette.types import JSONValue


@dataclass(frozen=True)
class CassetteDiff:
    """The report for two cassette files."""

    identical: bool
    text: str


@dataclass(frozen=True)
class _Side:
    index: int
    event: Mapping[str, JSONValue]
    request: JSONValue
    response: JSONValue
    key: str


@dataclass(frozen=True)
class _Change:
    kind: str
    text: str


def diff_cassettes(left: Path, right: Path) -> CassetteDiff:
    """Compare ``left`` and ``right``.

    Events are aligned in order. A request is normalised: volatile ids and
    timestamps are removed and secrets are redacted. A response keeps
    recorded ids and timestamps, and secrets are redacted. Tool results are
    the response. The report names added, removed, and changed events.
    """

    left_events = _read(left)
    right_events = _read(right)
    changes = _changes(left, right, _prepare(left_events), _prepare(right_events))
    if not changes:
        noun = "event" if len(left_events) == 1 else "events"
        return CassetteDiff(
            identical=True,
            text=f"cassettes are identical ({len(left_events)} {noun})\n",
        )
    body = "\n\n".join(change.text for change in changes)
    text = f"--- {left}\n+++ {right}\n\n{_summary(changes)}\n\n{body}\n"
    return CassetteDiff(identical=False, text=text)


def _read(path: Path) -> list[dict[str, JSONValue]]:
    if not path.is_file():
        raise CassetteNotFoundError(path)
    return read_events(path)


def _prepare(events: Sequence[Mapping[str, JSONValue]]) -> list[_Side]:
    prepared: list[_Side] = []
    for index, event in enumerate(events):
        request, response = _split(event)
        prepared.append(
            _Side(
                index=index,
                event=event,
                request=request,
                response=response,
                key=_canonical({"request": request, "response": response}),
            )
        )
    return prepared


def _split(event: Mapping[str, JSONValue]) -> tuple[JSONValue, JSONValue]:
    kind = event.get("type")
    if kind == "llm_call":
        response_key = "response"
    elif kind == "tool_call":
        response_key = "result"
    else:
        raise CassetteError(f"unknown event type {kind!r}")
    request = {
        key: value for key, value in event.items() if key not in {response_key, "type"}
    }
    return normalise_request(request), redact_secrets(event[response_key])


def _changes(
    left_path: Path,
    right_path: Path,
    left: Sequence[_Side],
    right: Sequence[_Side],
) -> list[_Change]:
    changes: list[_Change] = []
    matcher = difflib.SequenceMatcher(
        a=[item.key for item in left],
        b=[item.key for item in right],
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            changes.extend(_added(right[index]) for index in range(j1, j2))
            continue
        if tag == "delete":
            changes.extend(_removed(left[index]) for index in range(i1, i2))
            continue
        paired = min(i2 - i1, j2 - j1)
        for offset in range(paired):
            changes.append(
                _changed(left_path, right_path, left[i1 + offset], right[j1 + offset])
            )
        changes.extend(_removed(left[index]) for index in range(i1 + paired, i2))
        changes.extend(_added(right[index]) for index in range(j1 + paired, j2))
    return changes


def _added(side: _Side) -> _Change:
    return _Change("added", _presence("added", side))


def _removed(side: _Side) -> _Change:
    return _Change("removed", _presence("removed", side))


def _presence(label: str, side: _Side) -> str:
    return "\n".join([f"{label} {_describe(side)}", "request:", _pretty(side.request)])


def _changed(left_path: Path, right_path: Path, left: _Side, right: _Side) -> _Change:
    left_label = _describe(left)
    right_label = _describe(right)
    heading = (
        f"changed {left_label}"
        if left_label == right_label
        else f"changed {left_label} -> {right_label}"
    )
    lines = [heading]
    if left.request != right.request:
        lines.append(
            _section("request", left.request, right.request, left_path, right_path)
        )
    if left.response != right.response:
        lines.append(
            _section("response", left.response, right.response, left_path, right_path)
        )
    return _Change("changed", "\n".join(lines))


def _section(
    title: str,
    left: JSONValue,
    right: JSONValue,
    left_path: Path,
    right_path: Path,
) -> str:
    diff = _unified(left, right, f"{left_path} {title}", f"{right_path} {title}")
    return f"{title}:\n{diff}"


def _describe(side: _Side) -> str:
    number = side.index + 1
    kind = side.event.get("type")
    if kind == "llm_call":
        return f"llm_call {number} ({side.event.get('provider')})"
    if kind == "tool_call":
        return f"tool_call {number} ({side.event.get('name')})"
    return f"{kind} {number}"


def _summary(changes: Sequence[_Change]) -> str:
    parts = [
        f"{sum(change.kind == name for change in changes)} {name}"
        for name in ("added", "removed", "changed")
    ]
    return ", ".join(part for part in parts if not part.startswith("0 "))


def _unified(left: JSONValue, right: JSONValue, fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(
        _pretty(left).splitlines(),
        _pretty(right).splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    return "\n".join(diff)


def _pretty(value: JSONValue) -> str:
    try:
        return json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CassetteError("could not render cassette JSON") from exc


def _canonical(value: JSONValue) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CassetteError("could not compare cassette JSON") from exc
