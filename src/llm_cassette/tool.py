"""Record and replay plain Python tool functions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from inspect import iscoroutinefunction, signature
from typing import Any, ParamSpec, TypeVar, cast

from llm_cassette.cassette import current_cassette

P = ParamSpec("P")
R = TypeVar("R")


def tool(fn: Callable[P, R]) -> Callable[P, R]:
    """Record and replay a tool function.

    With a cassette open, the call is stored as a ``tool_call`` under the
    function's ``__name__``. The arguments are the bound call, including
    defaults. Replay returns the recorded JSON result and does not call the
    function. A different argument raises ``CassetteMismatchError``. With no
    cassette open, the function runs as usual.

    Results follow the cassette rules: they must be JSON, tuples become
    lists, and secret fields are redacted.
    """

    declared = signature(fn)
    name = fn.__name__

    def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = declared.bind(*args, **kwargs)
        bound.apply_defaults()
        return {"name": name, "args": dict(bound.arguments)}

    if iscoroutinefunction(fn):
        async_fn = cast(Callable[P, Awaitable[Any]], fn)

        @wraps(fn)
        async def async_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
            session = current_cassette()
            if session is None:
                return await async_fn(*args, **kwargs)

            async def call() -> Any:
                return await async_fn(*args, **kwargs)

            return await session.record_or_replay(
                "tool_call",
                request(*args, **kwargs),
                call,
            )

        return cast(Callable[P, R], async_wrapped)

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        session = current_cassette()
        if session is None:
            return fn(*args, **kwargs)

        def call() -> R:
            return fn(*args, **kwargs)

        return session.record_or_replay(
            "tool_call",
            request(*args, **kwargs),
            call,
        )

    return wrapped
