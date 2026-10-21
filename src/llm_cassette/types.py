"""Shared types for the cassette core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

Mode = Literal["once", "replay", "record", "none"]
EventKind = Literal["llm_call", "tool_call"]
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
Matcher = Callable[[Any, Any], bool]
