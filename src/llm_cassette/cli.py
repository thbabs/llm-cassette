"""Command line for llm-cassette."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_cassette.diff import diff_cassettes
from llm_cassette.errors import CassetteError, CassetteNotFoundError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llm-cassette command. Returns a process exit code."""

    parser = _parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command != "diff":
        print(f"llm-cassette: unknown command {args.command}", file=sys.stderr)
        return 2
    return _diff(args.left, args.right)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-cassette")
    sub = parser.add_subparsers(dest="command", required=True)
    diff = sub.add_parser(
        "diff",
        help="compare two cassette files",
        description=(
            "Compare two cassettes event by event. "
            "Events that were added or removed are named, and a changed event "
            "shows a unified diff of its normalised request, its response, or both. "
            "Requests drop volatile ids and timestamps. Responses keep recorded "
            "ids and timestamps. Secrets are redacted on both sides. "
            "Tool results are shown as the response. "
            "Exit 0 when the cassettes match, 1 when they differ, "
            "and 2 when a file is missing or invalid."
        ),
    )
    diff.add_argument(
        "left",
        type=Path,
        metavar="A.jsonl",
        help="cassette to diff from",
    )
    diff.add_argument(
        "right",
        type=Path,
        metavar="B.jsonl",
        help="cassette to diff to",
    )
    return parser


def _diff(left: Path, right: Path) -> int:
    try:
        result = diff_cassettes(left, right)
    except CassetteNotFoundError as exc:
        print(f"llm-cassette: {exc}", file=sys.stderr)
        return 2
    except CassetteError as exc:
        print(f"llm-cassette: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"llm-cassette: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(result.text)
    return 0 if result.identical else 1
