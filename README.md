# llm-cassette

[![CI](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml/badge.svg)](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml)

Record and replay LLM and tool calls as readable JSONL cassettes.

**Status: core, OpenAI, Anthropic, and `@tool`.** `cassette()` records and replays `llm_call` and `tool_call` events through `Cassette.record_or_replay`, including async callables. The optional `openai` extra patches the OpenAI Python SDK 1.x, and the optional `anthropic` extra patches the Anthropic Python SDK 2.x, while a cassette is open. `@tool` records plain Python functions. The pytest plugin and `llm-cassette diff` are still to come.

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

## OpenAI

`pip install "llm-cassette[openai]"` installs the OpenAI Python SDK 1.x. `install()` patches `chat.completions.create` and `responses.create` on the sync and async clients while a cassette is open, and restores the original methods when it exits. Replay rebuilds `ChatCompletion` and `Response` objects, and re-emits streams as the SDK's stream objects. The API key is not written to the cassette.

```python
from openai import OpenAI

from llm_cassette import cassette
from llm_cassette.openai import install

install()

with cassette("cassettes/hello.jsonl", mode="once"):
    completion = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )

print(completion.choices[0].message.content)
```

## Anthropic

`pip install "llm-cassette[anthropic]"` installs the Anthropic Python SDK 2.x. `install()` patches `messages.create` and `messages.stream` on the sync and async clients while a cassette is open, and restores the original methods when it exits. `create(stream=True)` replays as the SDK's `Stream` or `AsyncStream` of raw message events. `messages.stream()` replays as `MessageStream` or `AsyncMessageStream`, including `text_stream` and `get_final_message()`. That helper is recorded when its context manager is entered, which is when the SDK sends the request. Other calls replay as `anthropic.types.Message` objects. The API key is not written to the cassette.

```python
from anthropic import Anthropic

from llm_cassette import cassette
from llm_cassette.anthropic import install

install()

with cassette("cassettes/hello.jsonl", mode="once"):
    message = Anthropic().messages.create(
        model="claude-sonnet-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hello"}],
    )

print(message.content[0].text)
```

## Tools

`@tool` records a function's name, its bound arguments (including defaults), and its JSON result as a `tool_call`. Replay returns that result and does not call the function. A different argument raises `CassetteMismatchError`. Async functions work the same way. With no cassette open, the function runs as usual. Results follow the cassette rules: they must be JSON, tuples become lists, and secret fields are redacted.

```python
from llm_cassette import cassette, tool


@tool
def get_weather(city: str, unit: str = "c") -> dict[str, str]:
    return {"city": city, "unit": unit}


with cassette("cassettes/weather.jsonl", mode="once"):
    weather = get_weather("Paris")
```

`get_weather("Paris")` and `get_weather(city="Paris", unit="c")` match, because the recorded arguments include the default `unit`.

## Still to come

- A pytest fixture and `--replay-mode`
- An `llm-cassette diff` command

## Development

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,openai,anthropic]"
pre-commit install
ruff check .
ruff format --check .
mypy
pytest
```

CI runs ruff, mypy, and pytest on Python 3.10, 3.11, 3.12, and 3.13.
