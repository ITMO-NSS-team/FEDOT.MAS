from __future__ import annotations

import re
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.mcp import strip_tool_name_prefix
from fedotmas.plugins._research_telemetry import ResearchTelemetry

CODE_AGENT_BUDGET_STATE_KEY = "_fedotmas_code_agent_budget"
TASK_DEADLINE_STATE_KEY = "_fedotmas_task_deadline_monotonic"
GAIA_TASK_FILE_PATH_STATE_KEY = "_fedotmas_gaia_task_file_path"

_DOCUMENT_READING_ACTIONS = re.compile(
    r"\b(?:read|print|dump|show|display|view|inspect)\b", re.IGNORECASE
)
_DOCUMENT_CONTENT_TARGETS = re.compile(
    r"\b(?:file|contents?|rows?|columns?|spreadsheet|csv|json|pdf|document|table)\b",
    re.IGNORECASE,
)
_MATERIAL_COMPUTATION = re.compile(
    r"\b(?:calculat\w*|comput\w*|solv\w*|analy[sz]\w*|count\w*|sum|average|mean|median|"
    r"max(?:imum)?|min(?:imum)?|highest|lowest|where|matching|satisf\w*|"
    r"filter\w*|aggregat\w*|group\w*|join\w*|rank\w*|compar\w*|optimi[sz]\w*|"
    r"convert\w*|transform\w*|derive\w*)\b",
    re.IGNORECASE,
)


def _is_document_reading_call(tool_args: dict[str, Any]) -> bool:
    task = tool_args.get("task", "")
    context = tool_args.get("context", "")
    instruction = " ".join(
        value for value in (task, context) if isinstance(value, str)
    )
    if not isinstance(task, str) or not task.strip():
        return False
    if _MATERIAL_COMPUTATION.search(instruction):
        return False
    has_file = bool(tool_args.get("files")) or bool(
        _DOCUMENT_CONTENT_TARGETS.search(instruction)
    )
    return has_file and bool(_DOCUMENT_READING_ACTIONS.search(instruction))


class CodeAgentBudgetPlugin(BasePlugin):
    """Bound nested stateless code-agent work and reserve outer finalization time."""

    def __init__(
        self,
        *,
        max_calls_per_agent: int,
        total_seconds_per_agent: float,
        max_seconds_per_call: float,
        task_timeout_seconds: float,
        deadline_reserve_seconds: float = 90,
        telemetry: ResearchTelemetry | None = None,
        name: str = "fedotmas_gaia_code_agent_budget",
    ) -> None:
        super().__init__(name=name)
        if min(
            max_calls_per_agent,
            total_seconds_per_agent,
            max_seconds_per_call,
            task_timeout_seconds,
        ) <= 0:
            raise ValueError("code-agent budgets and task timeout must be positive")
        self.max_calls_per_agent = max_calls_per_agent
        self.total_seconds_per_agent = float(total_seconds_per_agent)
        self.max_seconds_per_call = float(max_seconds_per_call)
        self.task_timeout_seconds = float(task_timeout_seconds)
        self.deadline_reserve_seconds = max(0.0, float(deadline_reserve_seconds))
        self.telemetry = telemetry
        self._reservations: dict[tuple[str, str, str], tuple[float, float]] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        state = invocation_context.session.state
        state.setdefault(
            TASK_DEADLINE_STATE_KEY,
            time.monotonic() + self.task_timeout_seconds,
        )
        state.setdefault(CODE_AGENT_BUDGET_STATE_KEY, {})

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        tool_names = {
            strip_tool_name_prefix(name).lower()
            for name in llm_request.tools_dict
        }
        if "solve_with_code" not in tool_names:
            return
        agent_name = callback_context._invocation_context.agent.name
        root = callback_context.state.get(CODE_AGENT_BUDGET_STATE_KEY, {})
        current = root.get(agent_name, {}) if isinstance(root, dict) else {}
        if isinstance(current, dict) and current.get("status") == "exhausted":
            disabled = {
                name
                for name, tool in llm_request.tools_dict.items()
                if strip_tool_name_prefix(name).lower() == "solve_with_code"
            }
            retained = []
            for group in llm_request.config.tools or []:
                declarations = group.function_declarations
                if declarations is None:
                    retained.append(group)
                    continue
                group.function_declarations = [
                    item for item in declarations if item.name not in disabled
                ]
                if group.function_declarations:
                    retained.append(group)
            llm_request.config.tools = retained
            llm_request.append_instructions(
                [
                    "Nested code-agent budget is exhausted and solve_with_code is "
                    "unavailable for this agent. Summarize the best supported state "
                    "now; do not retry the nested computation."
                ]
            )
            return

        path = callback_context.state.get(GAIA_TASK_FILE_PATH_STATE_KEY)
        if isinstance(path, str) and path:
            llm_request.append_instructions(
                [
                    "Original custom task file (HOST path): "
                    f"{path}. For computation, call solve_with_code with "
                    f"files=[{path!r}]. Do not copy its contents into task or context; "
                    "those are instructions, not data transport. The file is staged "
                    "automatically in a fresh independent sandbox. Pass it again on "
                    "any later solve_with_code call."
                ]
            )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        if strip_tool_name_prefix(tool.name).lower() != "solve_with_code":
            return None
        if _is_document_reading_call(tool_args):
            return {
                "status": "blocked",
                "error_code": "CODE_AGENT_DOCUMENT_READING_RECOMMENDED",
                "errors": [
                    (
                        "If document tools are available, use them to read file "
                        "contents. Otherwise, do not retry file inspection. For an "
                        "actual computational task, make one self-contained "
                        "solve_with_code(files=[...]) call that parses the supplied "
                        "file internally and performs the computation."
                    )
                ],
                "converge_now": False,
                "session_persistent": False,
            }
        task, context = tool_args.get("task", ""), tool_args.get("context", "")
        # Let the tool return its actionable argument correction. Rejected data
        # transport does not consume a nested execution slot or time budget.
        if isinstance(task, str) and len(task) > 8_000:
            return None
        if isinstance(context, str) and len(context) > 4_000:
            return None

        invocation = tool_context._invocation_context
        session_id = invocation.session.id
        agent_name = invocation.agent.name  # ty: ignore[unresolved-attribute]
        state = tool_context.state
        root = state.get(CODE_AGENT_BUDGET_STATE_KEY)
        if not isinstance(root, dict):
            root = {}
            state[CODE_AGENT_BUDGET_STATE_KEY] = root
        budget = root.setdefault(
            agent_name, {"calls": 0, "seconds": 0.0, "reserved_seconds": 0.0}
        )
        if not isinstance(budget, dict):
            budget = {"calls": 0, "seconds": 0.0, "reserved_seconds": 0.0}
            root[agent_name] = budget

        exhausted = str(budget.get("status", "")) == "exhausted"
        used_calls = int(budget.get("calls", 0))
        used_seconds = float(budget.get("seconds", 0.0))
        reserved = float(budget.get("reserved_seconds", 0.0))
        deadline = state.get(TASK_DEADLINE_STATE_KEY)
        remaining_task = (
            float(deadline) - time.monotonic()
            if isinstance(deadline, int | float)
            else self.task_timeout_seconds
        )
        remaining_total = self.total_seconds_per_agent - used_seconds - reserved

        reason_code = None
        reason = ""
        if exhausted or used_calls >= self.max_calls_per_agent:
            reason_code = "CODE_AGENT_BUDGET_EXHAUSTED"
            reason = (
                "Nested code-agent budget exhausted. Do not retry this tool; "
                "summarize the best supported state now."
            )
        elif remaining_task <= self.deadline_reserve_seconds + 1:
            reason_code = "TASK_DEADLINE_NEAR"
            reason = (
                "Too little outer task time remains for nested code-agent work. "
                "Converge and summarize the best supported state now."
            )
        else:
            requested = float(
                tool_args.get("max_execution_seconds", self.max_seconds_per_call)
            )
            available = min(
                self.max_seconds_per_call,
                remaining_total,
                remaining_task - self.deadline_reserve_seconds,
            )
            if available < 1:
                reason_code = (
                    "TASK_DEADLINE_NEAR"
                    if remaining_task - self.deadline_reserve_seconds < 1
                    else "CODE_AGENT_BUDGET_EXHAUSTED"
                )
                reason = (
                    "No safe nested execution budget remains. Do not retry this "
                    "tool; summarize the best supported state now."
                )
            else:
                allowed = min(requested, available)
                tool_args["max_execution_seconds"] = allowed
                budget["calls"] = used_calls + 1
                budget["reserved_seconds"] = reserved + allowed
                call_id = getattr(tool_context, "function_call_id", None)
                key = (session_id, agent_name, str(call_id or used_calls + 1))
                self._reservations[key] = (time.monotonic(), allowed)

        budget.update(
            {
                "limit_calls": self.max_calls_per_agent,
                "limit_seconds": self.total_seconds_per_agent,
                "remaining_outer_seconds": max(0.0, remaining_task),
            }
        )
        if reason_code is None:
            return None

        budget["status"] = "exhausted"
        if self.telemetry is not None:
            self.telemetry.budget_blocked(agent_name)
            self.telemetry.record_blocked(
                agent_name,
                tool.name,
                tool_args,
                category=(
                    "task_deadline_near"
                    if reason_code == "TASK_DEADLINE_NEAR"
                    else "code_agent_budget_exhausted"
                ),
                call_id=getattr(tool_context, "function_call_id", None),
                budget={
                    "kind": "code_agent",
                    "limit_calls": self.max_calls_per_agent,
                    "used_calls": used_calls,
                    "limit_seconds": self.total_seconds_per_agent,
                    "used_seconds": round(used_seconds, 3),
                    "remaining_outer_seconds": round(max(0.0, remaining_task), 3),
                    "status": "exhausted",
                },
            )
        return {
            "status": "blocked",
            "error_code": reason_code,
            "errors": [reason],
            "converge_now": True,
            "session_persistent": False,
        }

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> None:
        if strip_tool_name_prefix(tool.name).lower() != "solve_with_code":
            return
        invocation = tool_context._invocation_context
        agent_name = invocation.agent.name  # ty: ignore[unresolved-attribute]
        call_id = getattr(tool_context, "function_call_id", None)
        state = tool_context.state
        root = state.get(CODE_AGENT_BUDGET_STATE_KEY)
        budget = root.get(agent_name) if isinstance(root, dict) else None
        key = (
            invocation.session.id,
            agent_name,
            str(call_id or (budget.get("calls", "") if isinstance(budget, dict) else "")),
        )
        reservation = self._reservations.pop(key, None)
        if reservation is None:
            return
        started, reserved = reservation
        elapsed = max(0.0, time.monotonic() - started)
        if isinstance(budget, dict):
            budget["reserved_seconds"] = max(
                0.0, float(budget.get("reserved_seconds", 0.0)) - reserved
            )
            result_seconds = elapsed
            if isinstance(result, dict):
                payload = result.get("structuredContent") or result
                if isinstance(payload, dict):
                    telemetry = payload.get("telemetry")
                    if isinstance(telemetry, dict):
                        duration = telemetry.get("duration_seconds")
                        if isinstance(duration, int | float):
                            result_seconds = max(result_seconds, float(duration))
            budget["seconds"] = float(budget.get("seconds", 0.0)) + result_seconds
