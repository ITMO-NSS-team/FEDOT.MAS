from __future__ import annotations

from collections.abc import Callable
from typing import Any

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")

CheckFn = Callable[[dict[str, Any]], str | None]


class EvaluationError(RuntimeError):
    """Raised by EvalPlugin when agent output fails evaluation."""

    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}' failed evaluation: {message}")


class EvalPlugin:
    """Evaluates agent outputs after execution, raises on failure.

    Implements :class:`MiddlewareProtocol`.

    Each check function receives the full pipeline state and returns
    an error message string if the output is bad, or ``None`` if OK.
    """

    def __init__(self, checks: dict[str, CheckFn]) -> None:
        self._checks = checks

    async def before_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> dict[str, str] | None:
        return None

    async def after_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> None:
        if agent_name.startswith(_WORKFLOW_PREFIXES):
            return

        check = self._checks.get(agent_name)
        if check is None:
            return

        error_msg = check(state)
        if error_msg is not None:
            raise EvaluationError(agent_name, error_msg)
