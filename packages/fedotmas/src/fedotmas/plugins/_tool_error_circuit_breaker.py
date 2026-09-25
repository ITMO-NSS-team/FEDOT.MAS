from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger
from fedotmas.mcp import strip_tool_name_prefix

if TYPE_CHECKING:
    from fedotmas.plugins._research_telemetry import ResearchTelemetry

_log = get_logger("fedotmas.plugins.tool_error_circuit_breaker")

#: Set in a tool result's ``meta`` by a server that answered a failed call with
#: a usable substitute -- the web-scraping proxy serves page markdown when
#: lightpanda's ``extract`` rejects a schema.  Such a result is still an error,
#: since the call did not do what was asked, but the agent got what it needed,
#: so it must not count towards a breaker meant to stop an agent battering a
#: tool that gives it nothing.  A bare string because the server that sets it
#: is a separate package.
RESCUED_META_KEY = "fedotmas/rescued"

# Machine-readable control-flow result returned when an agent's own web budget
# is exhausted. It is a blocked call, not a failure of the underlying tool.
WEB_BUDGET_EXHAUSTED = "WEB_BUDGET_EXHAUSTED"
DUPLICATE_TOOL_CALL = "DUPLICATE_TOOL_CALL"
TOOL_CIRCUIT_OPEN = "TOOL_CIRCUIT_OPEN"


class ToolErrorCircuitOpen(RuntimeError):
    """Legacy exception retained for import compatibility; circuits are local."""


class ToolErrorCircuitBreakerPlugin(BasePlugin):
    """Open a local agent/tool circuit after repeated errors."""

    def __init__(
        self,
        *,
        max_errors_per_agent: int = 10,
        max_same_tool_error_type: int = 3,
        telemetry: ResearchTelemetry | None = None,
        name: str = "fedotmas_tool_error_circuit_breaker",
    ) -> None:
        if max_errors_per_agent < 1:
            raise ValueError("max_errors_per_agent must be >= 1")
        if max_same_tool_error_type < 1:
            raise ValueError("max_same_tool_error_type must be >= 1")
        super().__init__(name=name)
        self.max_errors_per_agent = max_errors_per_agent
        self.max_same_tool_error_type = max_same_tool_error_type
        self.telemetry = telemetry
        self._total_errors: dict[tuple[str, str], int] = {}
        self._pattern_errors: dict[tuple[str, str, str, str], int] = {}
        self._open_circuits: dict[tuple[str, str, str], str] = {}

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
        self._open_circuits = {
            key: reason
            for key, reason in self._open_circuits.items()
            if key[0] != session_id
        }

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        session_id, agent_name = _session_agent(tool_context)
        circuit_key = (session_id, agent_name, tool.name)
        reason = self._open_circuits.get(circuit_key)
        if reason is None:
            return None
        if self.telemetry is not None:
            self.telemetry.circuit_blocked(agent_name, tool.name)
            self.telemetry.record_blocked(
                agent_name,
                tool.name,
                tool_args,
                category="circuit_breaker",
                call_id=getattr(tool_context, "function_call_id", None),
            )
        return {
            "isError": True,
            "error_code": TOOL_CIRCUIT_OPEN,
            "error": reason,
        }

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        error_code = result.get("error_code") if isinstance(result, dict) else None
        if isinstance(error_code, str) and error_code in {
            WEB_BUDGET_EXHAUSTED,
            DUPLICATE_TOOL_CALL,
            TOOL_CIRCUIT_OPEN,
        }:
            return None
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
    ) -> dict | None:
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
        session_id, agent_name = _session_agent(tool_context)
        total_key = (session_id, agent_name)
        tool_name = strip_tool_name_prefix(tool.name)
        circuit_key = (session_id, agent_name, tool.name)
        pattern_key = (session_id, agent_name, tool_name, error_type)

        if circuit_key in self._open_circuits:
            return

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
            reason = (
                f"The {tool.name} circuit is open for this agent after "
                f"{pattern_total} {error_type} failures. Use another tool, existing "
                "evidence, or finish your assigned role."
            )
            self._open_circuits[circuit_key] = reason
        elif total >= self.max_errors_per_agent:
            reason = (
                f"The {tool.name} circuit is open for this agent after "
                f"{total} tool failures in this run. Use another tool, existing "
                "evidence, or finish your assigned role."
            )
            self._open_circuits[circuit_key] = reason


def _is_error_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    meta = result.get("meta")
    if isinstance(meta, dict) and meta.get(RESCUED_META_KEY):
        return False
    if result.get("isError") is True or result.get("is_error") is True:
        return True
    value = result.get("error")
    return "error" in result and value is not None and value != ""


def _session_agent(tool_context: ToolContext) -> tuple[str, str]:
    invocation = tool_context._invocation_context
    return (
        invocation.session.id,
        invocation.agent.name,  # ty: ignore[unresolved-attribute]
    )


def _error_type_from_result(result: dict) -> str:
    for payload in (
        result,
        *(
            result.get(key)
            for key in ("meta", "_meta", "structuredContent", "structured_content")
        ),
    ):
        if isinstance(payload, dict):
            code = payload.get("error_code")
            if isinstance(code, str) and code.startswith("BROWSER_AGENT_"):
                return code
    value = result.get("error")
    if isinstance(value, dict):
        for key in ("type", "code", "error_type"):
            if value.get(key):
                return str(value[key])
    if isinstance(value, str) and value.strip():
        return value.split(":", 1)[0][:80]
    return "ToolErrorResult"
