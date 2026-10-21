# llm-cassette

[![CI](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml/badge.svg)](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml)

Record an LLM agent's OpenAI and Anthropic SDK calls and Python tool calls into readable JSONL cassettes, and replay them offline in pytest.

**Status: early, not yet usable.** Version 0.0.1 is a package scaffold. Recording, replay, and the diff command are not implemented.

## Planned for v0.1

- Record and replay OpenAI and Anthropic SDK calls, including async clients and streaming responses
- Record Python functions decorated with `@tool`
- A pytest fixture with a `--replay-mode` option
- Strict errors when a live call does not match the cassette
- An `llm-cassette diff` command

## Development

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
ruff check .
ruff format --check .
mypy
pytest
```

CI runs ruff, mypy, and pytest on Python 3.10, 3.11, 3.12, and 3.13.
