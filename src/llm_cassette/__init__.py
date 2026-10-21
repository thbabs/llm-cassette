"""Record and replay LLM and tool calls."""

from __future__ import annotations

from importlib.metadata import version

from llm_cassette.cassette import Cassette, cassette, current_cassette
from llm_cassette.errors import (
    CassetteError,
    CassetteMismatchError,
    CassetteNotFoundError,
)
from llm_cassette.normalise import normalise_request, request_hash
from llm_cassette.tool import tool
from llm_cassette.types import EventKind, Matcher, Mode

__all__ = [
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
]

__version__ = version("llm-cassette")
