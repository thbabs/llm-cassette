# Changelog

## 0.1.0 - 2026-10-21

First release.

### Added

- JSONL cassettes with `once`, `replay`, `record`, and `none`, ordered matching, request normalisation, and secret redaction.
- OpenAI Python SDK 1.x adapter for `chat.completions.create` and `responses.create`, including streams.
- Anthropic Python SDK 2.x adapter for `messages.create` and `messages.stream`.
- `@tool` for recording plain Python functions.
- Pytest plugin with `@pytest.mark.cassette`, the `cassette` fixture, and `--replay-mode`.
- `llm-cassette diff` for reviewing re-recorded cassettes.
- An offline OpenAI weather agent in `examples/`.
