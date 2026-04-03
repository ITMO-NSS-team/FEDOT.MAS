from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin
from google.genai import types

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")

CheckFn = Callable[[dict[str, Any]], str | None]


class EvaluationError(RuntimeError):
    """Raised by EvalPlugin when agent output fails evaluation."""

    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}' failed evaluation: {message}")


class EvalPlugin(BasePlugin):
    """Evaluates agent outputs after execution, raises on failure.

    Each check function receives the full pipeline state and returns
    an error message string if the output is bad, or ``None`` if OK.

    Usage::

        def check_number(state: dict) -> str | None:
            if "5" not in str(state.get("number", "")):
                return "Expected 5"
            return None

        plugin = EvalPlugin({"calculator": check_number})
    """

    def __init__(self, checks: dict[str, CheckFn]) -> None:
        super().__init__(name="fedotmas_eval")
        self._checks = checks

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if agent.name.startswith(_WORKFLOW_PREFIXES):
            return None

        check = self._checks.get(agent.name)
        if check is None:
            return None

        state = callback_context.state.to_dict()
        error_msg = check(state)
        if error_msg is not None:
            raise EvaluationError(agent.name, error_msg)
        return None
