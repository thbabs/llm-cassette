"""Pytest plugin that records and replays one cassette per test.

The plugin is registered through the ``pytest11`` entry point. Tests opt in
with ``@pytest.mark.cassette`` or the ``cassette`` fixture.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from pytest import StashKey, UsageError

from llm_cassette.cassette import Cassette
from llm_cassette.cassette import cassette as open_cassette
from llm_cassette.types import Mode

_MODES = frozenset({"once", "replay", "record", "none"})
_SAFE_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-[]"
)
_RESERVED_STEMS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_MAX_COMPONENT_BYTES = 180
_ACTIVE: StashKey[Cassette] = StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "replay_mode",
        "Default cassette mode for opted-in tests: once, replay, record, or none.",
        type="string",
        default="once",
    )
    group = parser.getgroup("llm-cassette")
    group.addoption(
        "--replay-mode",
        action="store",
        dest="replay_mode",
        choices=("once", "replay", "record", "none"),
        default=None,
        help=(
            "Cassette mode for @pytest.mark.cassette and the cassette fixture. "
            "once records when the file is missing and replays otherwise; "
            "replay fails if the file is missing; "
            "record calls through and overwrites the file when the test passes; "
            "none calls through and leaves the file untouched. "
            "Overrides the replay_mode ini option. Default: once."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cassette(name=None, mode=None): "
        "record and replay this test from a JSONL cassette",
    )
    raw: object = config.getini("replay_mode")
    if not isinstance(raw, str) or raw not in _MODES:
        raise UsageError(_unknown_mode("replay_mode", raw))


@pytest.fixture
def cassette(request: pytest.FixtureRequest) -> Iterator[Cassette]:
    """Open the cassette for this test.

    The file is ``cassettes/<module>/<test_name>.jsonl`` next to the test
    module, unless ``@pytest.mark.cassette`` sets a name. The mode comes
    from that marker, then ``--replay-mode``, then the ``replay_mode`` ini
    option, and otherwise ``once``.
    """

    with _opened(request) as session:
        yield session


@pytest.fixture(autouse=True)
def _llm_cassette_marker(request: pytest.FixtureRequest) -> Iterator[None]:
    marker = request.node.get_closest_marker("cassette")
    if marker is None or _ACTIVE in request.node.stash:
        yield
        return
    with _opened(request):
        yield


@contextmanager
def _opened(request: pytest.FixtureRequest) -> Iterator[Cassette]:
    node = request.node
    active = node.stash.get(_ACTIVE, None)
    if active is not None:
        yield active
        return
    name, marker_mode = _marker_settings(node.get_closest_marker("cassette"))
    if marker_mode is not None:
        mode = _as_mode(marker_mode)
    else:
        mode = _configured_mode(request.config)
    with open_cassette(_cassette_path(node, name), mode) as session:
        node.stash[_ACTIVE] = session
        try:
            yield session
        finally:
            del node.stash[_ACTIVE]


def _configured_mode(config: pytest.Config) -> Mode:
    selected: object = config.getoption("replay_mode")
    if selected is None:
        selected = config.getini("replay_mode")
    if not isinstance(selected, str):
        raise UsageError(_unknown_mode("replay_mode", selected))
    return _as_mode(selected)


def _as_mode(value: str) -> Mode:
    if value not in _MODES:
        raise ValueError(_unknown_mode("cassette mode", value))
    return cast(Mode, value)


def _unknown_mode(label: str, value: object) -> str:
    return f"unknown {label} {value!r}; expected one of: once, replay, record, none"


def _marker_settings(marker: pytest.Mark | None) -> tuple[str | None, str | None]:
    if marker is None:
        return None, None
    if len(marker.args) > 1:
        raise TypeError(
            "cassette marker accepts at most one positional argument, the cassette name"
        )
    positional: str | None = None
    if marker.args:
        positional = _marker_name(marker.args[0])
    unexpected = sorted(set(marker.kwargs) - {"name", "mode"})
    if unexpected:
        joined = ", ".join(unexpected)
        raise TypeError(f"cassette marker has unexpected arguments: {joined}")
    named = marker.kwargs.get("name", None)
    if named is not None:
        if positional is not None:
            raise TypeError(
                "pass the cassette name either as a positional argument or as name="
            )
        positional = _marker_name(named)
    mode = marker.kwargs.get("mode", None)
    if mode is not None and not isinstance(mode, str):
        raise TypeError("cassette mode must be a string")
    return positional, mode


def _marker_name(value: object) -> str:
    if isinstance(value, str) and value != "":
        return value
    raise TypeError("cassette name must be a non-empty string")


def _cassette_path(item: pytest.Item, name: str | None) -> Path:
    module_dir = item.path.parent / "cassettes" / _safe_component(item.path.stem)
    if name is None:
        parts = [*_class_parts(item), item.name]
        safe_parts = [_safe_component(part) for part in parts]
        return module_dir.joinpath(*safe_parts[:-1], f"{safe_parts[-1]}.jsonl")
    directories, stem = _split_name(name)
    filename = _safe_component(stem + _param_suffix(item)) + ".jsonl"
    safe_directories = [_safe_component(part) for part in directories]
    return module_dir.joinpath(*safe_directories, filename)


def _class_parts(item: pytest.Item) -> list[str]:
    cls = getattr(item, "cls", None)
    qualname = getattr(cls, "__qualname__", None)
    if not isinstance(qualname, str) or qualname == "":
        return []
    return [part for part in qualname.split(".") if part != ""]


def _param_suffix(item: pytest.Item) -> str:
    original = getattr(item, "originalname", None)
    if not isinstance(original, str):
        return ""
    name = item.name
    if name.startswith(original) and len(name) > len(original):
        return name[len(original) :]
    return ""


def _split_name(name: str) -> tuple[list[str], str]:
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    raw = Path(name)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(
            "cassette name must be a relative path under the cassette directory"
        )
    parts = list(raw.parts)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("cassette name must be a non-empty relative path")
    return parts[:-1], parts[-1]


def _safe_component(text: str) -> str:
    pieces: list[str] = []
    for char in text:
        if char in _SAFE_CHARS:
            pieces.append(char)
        else:
            pieces.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
    safe = "".join(pieces)
    if safe in {"", ".", ".."}:
        safe = f"{safe}%5F" if safe else "%5F"
    else:
        head, dot, tail = safe.partition(".")
        if head.upper() in _RESERVED_STEMS:
            safe = f"{head}%5F{dot}{tail}"
    encoded = safe.encode("utf-8")
    if len(encoded) <= _MAX_COMPONENT_BYTES:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    prefix = encoded[: _MAX_COMPONENT_BYTES - 17].decode("utf-8", errors="ignore")
    prefix = re.sub(r"%[0-9A-F]?$", "", prefix)
    return f"{prefix}-{digest}"
