from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Backend-agnostic lifecycle hooks for agent execution.

    Middleware implementations receive agent names and state dicts rather
    than framework-specific objects.  This keeps the middleware portable
    across backends (ADK, LangChain, PydanticAI, etc.).

    Return values:
        ``before_agent``: return a ``dict`` to short-circuit execution
        (the dict is treated as the agent's output/response).  Return
        ``None`` to proceed normally.

        ``after_agent``: return value is ignored.
    """

    async def before_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> dict[str, str] | None:
        """Called before an agent starts executing."""
        ...

    async def after_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> None:
        """Called after an agent finishes executing."""
        ...
