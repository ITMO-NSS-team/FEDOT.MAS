from __future__ import annotations

from typing import Any, Optional

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.tool_error_circuit_breaker")


class ToolErrorCircuitOpen(RuntimeError):
    """Raised when repeated tool failures trip a circuit breaker."""


class ToolErrorCircuitBreakerPlugin(BasePlugin):
    """Abort runs that repeatedly hit tool errors for the same agent/tool pattern."""

    def __init__(
        self,
        *,
        max_errors_per_agent: int = 10,
        max_same_tool_error_type: int = 3,
        name: str = "fedotmas_tool_error_circuit_breaker",
    ) -> None:
        if max_errors_per_agent < 1:
            raise ValueError("max_errors_per_agent must be >= 1")
        if max_same_tool_error_type < 1:
            raise ValueError("max_same_tool_error_type must be >= 1")
        super().__init__(name=name)
        self.max_errors_per_agent = max_errors_per_agent
        self.max_same_tool_error_type = max_same_tool_error_type
        self._total_errors: dict[tuple[str, str], int] = {}
        self._pattern_errors: dict[tuple[str, str, str, str], int] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        session_id = invocation_context.session.id
        self._total_errors = {
            key: count
            for key, count in self._total_errors.items()
            if key[0] != session_id
        }
        self._pattern_errors = {
            key: count
            for key, count in self._pattern_errors.items()
            if key[0] != session_id
        }
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        if not _is_error_result(result):
            return None

        error_type = _error_type_from_result(result)
        self._record_error(tool=tool, tool_context=tool_context, error_type=error_type)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        self._record_error(
            tool=tool,
            tool_context=tool_context,
            error_type=type(error).__name__,
        )
        return None

    def _record_error(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        error_type: str,
    ) -> None:
        session_id = tool_context._invocation_context.session.id
        agent_name = tool_context._invocation_context.agent.name  # noqa: E501  # ty: ignore[unresolved-attribute]
        total_key = (session_id, agent_name)
        pattern_key = (session_id, agent_name, tool.name, error_type)

        total = self._total_errors.get(total_key, 0) + 1
        pattern_total = self._pattern_errors.get(pattern_key, 0) + 1
        self._total_errors[total_key] = total
        self._pattern_errors[pattern_key] = pattern_total

        _log.warning(
            "Tool error recorded | agent={} tool={} error_type={} total={}/{} pattern={}/{}",
            agent_name,
            tool.name,
            error_type,
            total,
            self.max_errors_per_agent,
            pattern_total,
            self.max_same_tool_error_type,
        )

        if pattern_total >= self.max_same_tool_error_type:
            raise ToolErrorCircuitOpen(
                "Tool error circuit opened for agent "
                f"'{agent_name}' on tool '{tool.name}' with error type "
                f"'{error_type}': {pattern_total} repeated failures."
            )
        if total >= self.max_errors_per_agent:
            raise ToolErrorCircuitOpen(
                "Tool error circuit opened for agent "
                f"'{agent_name}': {total} tool errors in this run."
            )


def _is_error_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("isError") is True:
        return True
    return "error" in result and result.get("error") not in {None, ""}


def _error_type_from_result(result: dict) -> str:
    value = result.get("error")
    if isinstance(value, dict):
        for key in ("type", "code", "error_type"):
            if value.get(key):
                return str(value[key])
    if isinstance(value, str) and value.strip():
        return value.split(":", 1)[0][:80]
    return "ToolErrorResult"
