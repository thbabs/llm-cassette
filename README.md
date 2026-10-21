# llm-cassette

[![CI](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml/badge.svg)](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml)

Record and replay LLM and tool calls as readable JSONL cassettes.

**Status: core only.** `cassette()` can record and replay `llm_call` and `tool_call` events through `Cassette.record_or_replay`, including async callables. OpenAI and Anthropic SDK adapters, the `@tool` decorator, the pytest plugin, and `llm-cassette diff` are still to come.

## Usage

```python
from llm_cassette import cassette

request = {
    "provider": "openai",
    "request": {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    },
}

with cassette("cassettes/example.jsonl", mode="once") as session:
    response = session.record_or_replay(
        "llm_call",
        request,
        lambda: {"content": "hello"},
    )
```

`cassette(path, mode=...)` is a sync context manager, an async context manager, and a decorator. Inside the block, `record_or_replay(kind, request, call_fn)` runs the call or returns the recorded JSON. Await it when `call_fn` is async. Tool calls use `kind="tool_call"` with `name` and `args`; the callable's return value is the `result`. `current_cassette()` returns the session active in this context.

| Mode | What it does |
| --- | --- |
| `once` | Record when the file is missing, and replay it otherwise. |
| `replay` | Replay the file. A missing file raises `CassetteNotFoundError`. |
| `record` | Call through and overwrite the file when the block finishes successfully. |
| `none` | Call through and leave the file untouched. |

Replay walks the file in order and matches a hash of the normalised request. `strict=True` compares the normalised requests and raises `CassetteMismatchError` if a recorded interaction was not used. Pass `matcher` to replace the comparison. On a miss, the error shows the expected and received normalised requests.

Before hashing, matching, or writing, volatile fields (`id`, timestamps, and request ids) are removed. Secret fields (API keys, `Authorization`, and similar names) are replaced with `<redacted>`. `max_tokens` is kept. Recorded values come back as JSON, with tuples turned into lists, so a recording and a later replay match.

```json
{"type": "header", "format_version": 1}
{"type": "llm_call", "provider": "openai", "request": {"model": "gpt-4o"}, "response": {"content": "hello"}}
{"type": "tool_call", "name": "get_weather", "args": {"city": "Paris"}, "result": {"temp": 18}}
```

## Still to come

- OpenAI and Anthropic SDK clients, including async clients and streaming
- Recording Python functions decorated with `@tool`
- A pytest fixture and `--replay-mode`
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
