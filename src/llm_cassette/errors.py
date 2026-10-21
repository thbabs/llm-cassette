"""Errors raised while recording and replaying cassettes."""

from __future__ import annotations

from pathlib import Path

from llm_cassette.types import JSONValue


class CassetteError(Exception):
    """Base class for cassette failures."""


class CassetteNotFoundError(CassetteError):
    """Raised when replay mode cannot find the cassette file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"cassette not found: {path}")


class CassetteMismatchError(CassetteError):
    """Raised when a call does not match the recorded interaction."""

    def __init__(
        self,
        message: str,
        *,
        expected: JSONValue | None,
        received: JSONValue | None,
        index: int,
        path: Path,
    ) -> None:
        super().__init__(message)
        self.expected = expected
        self.received = received
        self.index = index
        self.path = path
