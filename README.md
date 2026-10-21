# llm-cassette

[![CI](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml/badge.svg)](https://github.com/thbabs/llm-cassette/actions/workflows/ci.yml)

Record and replay LLM and tool calls as readable JSONL cassettes.

**Status: 0.1.0.** `cassette()` records and replays `llm_call` and `tool_call` events through `Cassette.record_or_replay`, including async callables. The optional `openai` extra patches the OpenAI Python SDK 1.x, and the optional `anthropic` extra patches the Anthropic Python SDK 2.x, while a cassette is open. `@tool` records plain Python functions. The pytest plugin opens a cassette for a test that uses `@pytest.mark.cassette` or the `cassette` fixture. `llm-cassette diff` compares two cassettes.

## Quickstart

```python
import pytest
from llm_cassette import tool


@tool
def get_weather(city: str) -> dict[str, int]:
    return {"temp": 18}


@pytest.mark.cassette
def test_paris() -> None:
    assert get_weather("Paris") == {"temp": 18}
```

Put that in `tests/test_agent.py`. The first run writes `tests/cassettes/test_agent/test_paris.jsonl`. The next run replays that file and does not call `get_weather`. A different request fails the test with `CassetteMismatchError`.

```bash
llm-cassette diff old.jsonl new.jsonl
```

The command exits 0 when the cassettes match and 1 when they do not. A missing or invalid file exits 2.

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

## pytest

llm-cassette registers a pytest plugin. Opt in with `@pytest.mark.cassette` or the `cassette` fixture. The cassette sits next to the test module and is named after the test:

`tests/cassettes/test_agent/test_paris.jsonl`

A class method is nested under the class name. A parametrised test includes its id. Characters that are unsafe in a file name are percent-encoded, and `%` is encoded too, so each id gets its own file.

```python
import pytest

from llm_cassette import tool


@tool
def get_weather(city: str) -> dict[str, str]:
    return {"city": city}


@pytest.mark.cassette
def test_paris() -> None:
    assert get_weather("Paris") == {"city": "Paris"}
```

`@pytest.mark.cassette("weather", mode="replay")` uses `tests/cassettes/test_agent/weather.jsonl` and forces `replay` for that test. On a parametrised test the parameter id is added to that name. The `cassette` fixture is the open `Cassette`.

`--replay-mode` chooses `once` (the default), `replay`, `record`, or `none` for every opted-in test. `replay_mode` in a pytest config file is the same setting. The command line overrides the config file, and `mode=` on the marker overrides both.

| `--replay-mode` | What it does |
| --- | --- |
| `once` | Record when the file is missing, and replay it otherwise. |
| `replay` | Replay the file. A missing file raises `CassetteNotFoundError`. |
| `record` | Call through and overwrite the file when the test passes. |
| `none` | Call through and leave the file untouched. |

These are the same modes as `cassette()`. A request that does not match the recording fails the test with `CassetteMismatchError`, and the failure shows the expected and received requests. The plugin leaves `strict` off, so a test that does not replay every recorded interaction still passes. It does not call `install()` for the OpenAI or Anthropic adapters; those patches still have to be installed for SDK calls to be recorded.

`replay` is the mode that fails when a cassette is missing. `none` does not: it never reads the file.

```toml
[tool.pytest.ini_options]
replay_mode = "replay"
```

```bash
pytest --replay-mode=replay
```

## Example

`examples/openai_agent.py` is a two-step OpenAI agent. The model requests `get_weather`, the cassette supplies the temperature, and the model answers. Replay reads `examples/cassettes/weather.jsonl`. No API key and no network:

```bash
python examples/openai_agent.py
```

`examples/test_openai_agent.py` runs that replay under pytest. CI installs the OpenAI extra and does not set a key.

## Diff

`llm-cassette diff A.jsonl B.jsonl` compares interactions in order. It reports events that were added, removed, or changed. A changed `llm_call` shows a unified diff of the normalised request, the response, or both. A tool call's result is the response. Event numbers start at 1 and skip the header line.

Requests are normalised the same way recording is: volatile ids and timestamps are removed, and secret fields are redacted. Responses keep recorded ids and timestamps, because replay rebuilds SDK objects from them, and secret fields are redacted there too. Two files can differ on disk and still compare equal when the only request changes are volatile fields.

```text
--- old.jsonl
+++ new.jsonl

1 added, 1 changed

changed llm_call 1 (openai)
request:
--- old.jsonl request
+++ new.jsonl request
@@ -1,6 +1,6 @@
 {
   "provider": "openai",
   "request": {
-    "model": "gpt-4o"
+    "model": "gpt-4o-mini"
   }
 }

added tool_call 2 (get_weather)
request:
{
  "args": {
    "city": "Paris"
  },
  "name": "get_weather"
}
```

Exit 0 when that report says the cassettes are identical. Exit 1 when it lists changes. Exit 2 when a path is missing or the file is not a cassette. The command only reads the files.

## Limitations

- The OpenAI adapter covers `chat.completions.create` and `responses.create` on the Python SDK 1.x. The Anthropic adapter covers `messages.create` and `messages.stream` on the Python SDK 2.x. Other providers, other methods, and raw HTTP are not recorded.
- SDK calls are recorded only after `install()`. The pytest plugin does not call it.
- One cassette is not safe for concurrent use. An async `record_or_replay` has to be awaited.
- Recorded values have to be JSON. Tuples become lists. Datetimes, bytes, and other objects are rejected.
- Secret redaction matches field names such as `api_key`, `authorization`, and `token`. A secret written in ordinary text is stored as text. Read a cassette before committing it.
- Volatile-field removal matches key names. A request field named `id` is dropped wherever it sits, including an OpenAI tool call's `id`. `tool_call_id` is kept. Response ids and timestamps are kept. `llm-cassette diff` follows those rules: a response id change shows up, and a request id change does not.
- Matching walks the file in order. The same request twice is two interactions. The default comparison is a hash of the normalised request, not a comparison of the response.
- The pytest plugin leaves `strict` off, so a test can pass without replaying every recorded interaction. `replay` fails when the file is missing. `none` does not read the file.

## Roadmap

- Adapters for more SDKs and methods.
- A way to choose which field names are secret or volatile.
- A pytest option that installs the OpenAI and Anthropic adapters.
- An opt-in strict default, so unused interactions fail once a suite asks for that.

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

CI runs ruff, mypy, and pytest on Python 3.10, 3.11, 3.12, and 3.13. That includes `examples/test_openai_agent.py`.

## Releasing

Pushing a `v*` tag runs `.github/workflows/release.yml`. The tag has to match `pyproject.toml`: `v0.1.0` publishes version `0.1.0`. The workflow builds an sdist and a wheel and publishes them to PyPI with trusted publishing. It does not use an API token.
