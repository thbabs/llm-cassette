"""Replay a two-step OpenAI weather agent from a cassette.

No API key and no network. From a checkout with the OpenAI extra installed:

    python examples/openai_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
    Function,
)

from llm_cassette import cassette, tool

CASSETTE = Path(__file__).resolve().parent / "cassettes" / "weather.jsonl"
MODEL = "gpt-4o-mini"

# Replay leaves this empty. Recording appends the city.
calls: list[str] = []

_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
_TOOLS: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature in a city.",
            "parameters": _PARAMETERS,
        },
    }
]


@tool
def get_weather(city: str) -> dict[str, int]:
    """Return the temperature for a city."""

    calls.append(city)
    return {"temp_c": 18}


def ask(client: OpenAI, city: str) -> str:
    """Ask the model for the weather and follow its tool call."""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": f"What is the weather in {city}?"}
    ]
    for _ in range(2):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=_TOOLS,
        )
        message = completion.choices[0].message
        tool_calls = _function_calls(message)
        if not tool_calls:
            content = message.content
            if not isinstance(content, str) or content == "":
                raise RuntimeError("model returned an empty answer")
            return content
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [_tool_call(call) for call in tool_calls],
            }
        )
        for call in tool_calls:
            result = get_weather(_city(call.function.arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
    raise RuntimeError("model did not return an answer")


def main() -> None:
    from llm_cassette.openai import install

    install()
    client = OpenAI(api_key="sk-replay")
    with cassette(CASSETTE, mode="replay"):
        print(ask(client, "Paris"))


def _function_calls(
    message: ChatCompletionMessage,
) -> list[ChatCompletionMessageFunctionToolCall]:
    found: list[ChatCompletionMessageFunctionToolCall] = []
    for call in message.tool_calls or []:
        if not isinstance(call, ChatCompletionMessageFunctionToolCall):
            raise RuntimeError("only function tool calls are supported")
        found.append(call)
    return found


def _tool_call(
    call: ChatCompletionMessageFunctionToolCall,
) -> ChatCompletionMessageFunctionToolCallParam:
    function: Function = {
        "name": call.function.name,
        "arguments": call.function.arguments,
    }
    return {"id": call.id, "type": "function", "function": function}


def _city(arguments: str) -> str:
    payload: object = json.loads(arguments)
    if not isinstance(payload, dict):
        raise RuntimeError("get_weather arguments must be an object")
    city = payload.get("city")
    if not isinstance(city, str) or city == "":
        raise RuntimeError("get_weather arguments must include city")
    return city


if __name__ == "__main__":
    main()
