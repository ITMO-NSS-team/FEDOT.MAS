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
from google.genai import types

from fedotmas.mcp import strip_tool_name_prefix
from fedotmas.plugins._research_telemetry import ResearchTelemetry

CODE_AGENT_BUDGET_STATE_KEY = "_fedotmas_code_agent_budget"
TASK_DEADLINE_STATE_KEY = "_fedotmas_task_deadline_monotonic"
GAIA_TASK_FILE_PATH_STATE_KEY = "_fedotmas_gaia_task_file_path"


def _outcome(payload: Any) -> tuple[str, str]:
    data = payload if isinstance(payload, dict) else {}
    code = str(data.get("error_code") or "")
    status = str(data.get("status") or "")
    errors = data.get("errors") or []
    message = (
        " ".join(str(x) for x in errors)[:500]
        if isinstance(errors, list)
        else str(errors)[:500]
    )
    useful = bool(data.get("answer")) or bool(data.get("evidence"))
    if code == "CODE_AGENT_BUDGET_EXHAUSTED":
        return "budget_exhausted", message
    if status == "blocked" or code == "CODE_AGENT_DOCUMENT_READING_RECOMMENDED":
        return "blocked", message
    if "TIMEOUT" in code or status == "timed_out":
        return "timeout", message
    if any(
        x in code for x in ("PARSE", "INVALID_INPUT", "FILE_NOT_FOUND", "FILE_ACCESS")
    ):
        return "parse_or_validation_error", message
    if status == "completed":
        return ("success_with_result" if useful else "success_without_result"), message
    if status in {"failed", "incomplete"} or code:
        return "execution_error", message
    return "success_without_result", message


CODE_AGENT_POLICY_STATE_KEY = "_fedotmas_code_agent_policy"
CODE_AGENT_PHASE_STATE_KEY = "_fedotmas_code_agent_phase"


def _disable_code_tool(llm_request: LlmRequest) -> None:
    disabled = {
        name
        for name in llm_request.tools_dict
        if strip_tool_name_prefix(name).lower() == "solve_with_code"
    }
    retained = []
    for group in llm_request.config.tools or []:
        if not isinstance(group, types.Tool):
            retained.append(group)
            continue
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


_DOCUMENT_READING_ACTIONS = re.compile(
    r"\b(?:read|print|dump|show|display|view|inspect|output|list|locate|"
    r"reproduce|report)\b",
    re.IGNORECASE,
)
_DOCUMENT_CONTENT_TARGETS = re.compile(
    r"\b(?:file|contents?|rows?|columns?|spreadsheet|csv|json|pdf|document|table)\b",
    re.IGNORECASE,
)
_EXPLICIT_COMPUTE = re.compile(
    r"\b(?:solve|optimi[sz](?:e|ed|es|ing)?|calculat\w*|comput\w*|derive\w*|"
    r"aggregate\w*|transform\w*|rank\w*|join\w*)\b",
    re.IGNORECASE,
)
_CONDITIONAL_COUNT = re.compile(
    r"\bcount\w*\b.{0,100}\b(?:satisf\w*|match\w*|where|condition|"
    r"filter\w*|whose|with|having)\b|"
    r"\b(?:satisf\w*|match\w*|where|condition|filter\w*|whose|having)\b.{0,100}\bcount\w*\b",
    re.IGNORECASE,
)


def _is_document_reading_call(tool_args: dict[str, Any]) -> bool:
    task = tool_args.get("task", "")
    context = tool_args.get("context", "")
    instruction = " ".join(value for value in (task, context) if isinstance(value, str))
    if not isinstance(task, str) or not task.strip():
        return False
    has_file = bool(tool_args.get("files")) or bool(
        _DOCUMENT_CONTENT_TARGETS.search(instruction)
    )
    if not has_file:
        return False
    if _CONDITIONAL_COUNT.search(instruction):
        return False
    inspection = bool(_DOCUMENT_READING_ACTIONS.search(instruction)) or bool(
        re.search(
            r"\b(?:token counts?|field domains?|example rows?|headers?)\b",
            instruction,
            re.IGNORECASE,
        )
    )
    return inspection and not _EXPLICIT_COMPUTE.search(instruction)


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
        if (
            min(
                max_calls_per_agent,
                total_seconds_per_agent,
                max_seconds_per_call,
                task_timeout_seconds,
            )
            <= 0
        ):
            raise ValueError("code-agent budgets and task timeout must be positive")
        self.max_calls_per_agent = max_calls_per_agent
        self.total_seconds_per_agent = float(total_seconds_per_agent)
        self.max_seconds_per_call = float(max_seconds_per_call)
        self.task_timeout_seconds = float(task_timeout_seconds)
        self.deadline_reserve_seconds = max(0.0, float(deadline_reserve_seconds))
        self.telemetry = telemetry
        self._reservations: dict[tuple[str, str, str], tuple[float, float]] = {}
        self._requests: dict[tuple[str, str, str], tuple[str, str]] = {}

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
            strip_tool_name_prefix(name).lower() for name in llm_request.tools_dict
        }
        if "solve_with_code" not in tool_names:
            return
        assert callback_context._invocation_context.agent is not None
        agent_name = callback_context._invocation_context.agent.name
        root = callback_context.state.get(CODE_AGENT_BUDGET_STATE_KEY, {})
        current = root.get(agent_name, {}) if isinstance(root, dict) else {}
        phase_state = callback_context.state.get(CODE_AGENT_PHASE_STATE_KEY, {})
        control_phase = (
            current.get("phase", "primary_available")
            if isinstance(current, dict)
            else "primary_available"
        )
        phase = (
            phase_state.get(agent_name, control_phase)
            if isinstance(phase_state, dict) and isinstance(current, dict)
            else control_phase
        )
        if control_phase in {"complete", "budget_exhausted"}:
            _disable_code_tool(llm_request)
            llm_request.append_instructions(
                [
                    "The code-agent phase is complete and solve_with_code is unavailable. Finalize from the available evidence."
                ]
            )
            return

        llm_request.append_instructions(
            [
                "For exact computational tasks: use document-reading tools for lightweight input/parse inspection. solve_with_code always runs a solver, so call_intent=inspect is redirected without execution. Use call_intent=compute for the primary computation. After an observed failure, a recovery requires call_intent=targeted_recovery and recovery_target equal to the reported failure code/category. Validate parsing with compact structural invariants; make only that targeted correction; do not redesign the solver."
            ]
        )
        outcome = current.get("last_outcome") if isinstance(current, dict) else None
        if phase == "inspection":
            llm_request.append_instructions(
                [
                    "Code-agent phase: inspection. No computation budget has been consumed; proceed to one primary computation only after inspection is complete."
                ]
            )
        if control_phase == "recovery_available":
            llm_request.append_instructions(
                [
                    f"One targeted recovery is available for failure target {current.get('failure_code', 'execution_error') if isinstance(current, dict) else 'execution_error'}. Set call_intent=targeted_recovery and recovery_target to that exact value. Correct that observed failure only; do not redesign the entire solver."
                ]
            )
        elif outcome == "success_with_result" and (
            not isinstance(current, dict)
            or current.get("last_call_phase") != "inspection"
        ):
            llm_request.append_instructions(
                [
                    "The primary computation completed. Synthesize/finalize from the available result. Do not launch a new implementation of the same solve."
                ]
            )
        elif outcome:
            llm_request.append_instructions(
                [
                    f"Previous computation outcome: {outcome}. Concrete failure: {current.get('failure_reason', '')[:300] if isinstance(current, dict) else ''}. Phase: {phase}."
                ]
            )

        path = callback_context.state.get(GAIA_TASK_FILE_PATH_STATE_KEY)
        if isinstance(path, str) and path:
            llm_request.append_instructions(
                [
                    (
                        "Original custom task file (HOST path): "
                        f"{path}. For computation, call solve_with_code with "
                        f"files=[{path!r}]. Do not copy its contents into task or context; "
                        "those are instructions, not data transport. The file is staged "
                        "automatically in a fresh independent sandbox. Pass it again on "
                        "any later solve_with_code call."
                    )
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
        # This tool always invokes a solver. An inspection label must not
        # authorize computation outside the primary/recovery phase checks.
        if (
            _is_document_reading_call(tool_args)
            or tool_args.get("call_intent") == "inspect"
        ):
            invocation = tool_context._invocation_context
            phase_root = tool_context.state.setdefault(CODE_AGENT_PHASE_STATE_KEY, {})
            assert invocation.agent is not None
            if isinstance(phase_root, dict) and invocation.agent.name not in phase_root:
                phase_root[invocation.agent.name] = "inspection"
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
        budget: dict[str, Any] = root.setdefault(
            agent_name,
            {
                "calls": 0,
                "seconds": 0.0,
                "reserved_seconds": 0.0,
                "phase": "primary_available",
            },
        )
        if not isinstance(budget, dict):
            budget = {"calls": 0, "seconds": 0.0, "reserved_seconds": 0.0}
            root[agent_name] = budget

        policies = state.get(CODE_AGENT_POLICY_STATE_KEY, {})
        policy = policies.get(agent_name, {}) if isinstance(policies, dict) else {}
        role_policy = (
            policy.get("mode", "solver") if isinstance(policy, dict) else "solver"
        )
        phase = budget.get("phase", "primary_available")
        if int(budget.get("calls", 0)) >= self.max_calls_per_agent:
            budget["phase"] = "budget_exhausted"
            phase_root = state.setdefault(CODE_AGENT_PHASE_STATE_KEY, {})
            if isinstance(phase_root, dict):
                phase_root[agent_name] = "budget_exhausted"
            if self.telemetry is not None:
                self.telemetry.budget_blocked(agent_name)
                self.telemetry.record_blocked(
                    agent_name,
                    tool.name,
                    tool_args,
                    category="code_agent_budget_exhausted",
                    call_id=getattr(tool_context, "function_call_id", None),
                    budget={
                        "kind": "code_agent",
                        "limit_calls": self.max_calls_per_agent,
                        "used_calls": int(budget.get("calls", 0)),
                        "status": "exhausted",
                    },
                )
            return {
                "status": "blocked",
                "error_code": "CODE_AGENT_BUDGET_EXHAUSTED",
                "errors": [
                    "Nested code-agent budget exhausted. Finalize from existing evidence."
                ],
                "converge_now": True,
                "session_persistent": False,
            }
        if (
            self.total_seconds_per_agent
            - float(budget.get("seconds", 0.0))
            - float(budget.get("reserved_seconds", 0.0))
            < 1
        ):
            budget["phase"] = "budget_exhausted"
            phase_root = state.setdefault(CODE_AGENT_PHASE_STATE_KEY, {})
            if isinstance(phase_root, dict):
                phase_root[agent_name] = "budget_exhausted"
            if self.telemetry is not None:
                self.telemetry.budget_blocked(agent_name)
                self.telemetry.record_blocked(
                    agent_name,
                    tool.name,
                    tool_args,
                    category="code_agent_budget_exhausted",
                    call_id=getattr(tool_context, "function_call_id", None),
                    budget={
                        "kind": "code_agent",
                        "limit_calls": self.max_calls_per_agent,
                        "used_calls": int(budget.get("calls", 0)),
                        "limit_seconds": self.total_seconds_per_agent,
                        "used_seconds": float(budget.get("seconds", 0.0)),
                        "status": "exhausted",
                    },
                )
            return {
                "status": "blocked",
                "error_code": "CODE_AGENT_BUDGET_EXHAUSTED",
                "errors": [
                    "Nested code-agent runtime budget exhausted. Finalize from existing evidence."
                ],
                "converge_now": True,
                "session_persistent": False,
            }
        if phase in {"complete", "budget_exhausted"}:
            return {
                "status": "blocked",
                "error_code": "CODE_AGENT_PHASE_COMPLETE",
                "errors": [
                    "The code-agent execution phase is complete. Finalize from existing evidence."
                ],
                "converge_now": True,
                "session_persistent": False,
            }
        if phase == "recovery_available":
            expected_target = str(
                budget.get("failure_code")
                or budget.get("last_outcome")
                or "execution_error"
            )
            if (
                tool_args.get("call_intent") != "targeted_recovery"
                or tool_args.get("recovery_target") != expected_target
            ):
                return {
                    "status": "blocked",
                    "error_code": "CODE_AGENT_TARGETED_RECOVERY_REQUIRED",
                    "errors": [
                        f"One recovery remains. Use call_intent=targeted_recovery and recovery_target={expected_target!r} to fix the observed failure only."
                    ],
                    "converge_now": False,
                    "session_persistent": False,
                }
            budget["phase"] = "targeted_recovery_running"
            call_phase = "targeted_recovery"
        elif phase == "primary_available":
            budget["phase"] = "primary_running"
            call_phase = "primary_computation"
        else:
            return {
                "status": "blocked",
                "error_code": "CODE_AGENT_PHASE_COMPLETE",
                "errors": [
                    "A substantive computation is already in progress or the recovery allowance is spent."
                ],
                "converge_now": True,
                "session_persistent": False,
            }
        phase_root = state.setdefault(CODE_AGENT_PHASE_STATE_KEY, {})
        if isinstance(phase_root, dict):
            phase_root[agent_name] = budget["phase"]

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
                self._requests[key] = (call_phase, str(phase))
        budget["policy_mode"] = role_policy

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
        budget["phase"] = "budget_exhausted"
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
            str(
                call_id or (budget.get("calls", "") if isinstance(budget, dict) else "")
            ),
        )
        reservation = self._reservations.pop(key, None)
        if reservation is None:
            return
        started, reserved = reservation
        call_phase, previous_phase = self._requests.pop(
            key, ("primary_computation", "primary_available")
        )
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
            payload = (
                result.get("structuredContent") or result
                if isinstance(result, dict)
                else {}
            )
            if (
                isinstance(payload, dict)
                and payload.get("error_code")
                == "CODE_AGENT_DOCUMENT_READING_RECOMMENDED"
            ):
                budget["calls"] = max(0, int(budget.get("calls", 0)) - 1)
                budget["seconds"] = max(
                    0.0, float(budget.get("seconds", 0.0)) - result_seconds
                )
                # An inspection recommendation refunds this call but must not
                # erase a recovery-only phase established by an earlier failure.
                budget["phase"] = previous_phase
                phases = state.setdefault(CODE_AGENT_PHASE_STATE_KEY, {})
                if isinstance(phases, dict):
                    phases[agent_name] = (
                        "inspection"
                        if previous_phase == "primary_available"
                        else previous_phase
                    )
            else:
                category, message = _outcome(payload)
                budget["last_outcome"] = category
                budget["failure_reason"] = message
                budget["failure_code"] = (
                    str(payload.get("error_code") or category)
                    if isinstance(payload, dict)
                    else category
                )
                budget["last_call_phase"] = call_phase
                mode = budget.get("policy_mode", "solver")
                if category == "budget_exhausted":
                    budget["phase"] = "budget_exhausted"
                elif call_phase == "targeted_recovery" or mode == "verify_candidate":
                    budget["phase"] = "complete"
                elif category in {
                    "timeout",
                    "execution_error",
                    "parse_or_validation_error",
                }:
                    budget["phase"] = "recovery_available"
                else:
                    budget["phase"] = "complete"
                phase_root = state.setdefault(CODE_AGENT_PHASE_STATE_KEY, {})
                if isinstance(phase_root, dict):
                    phase_root[agent_name] = budget["phase"]
                # Keep only a concise structured summary in the model context.
                answer = (
                    str(payload.get("answer") or "")[:3000]
                    if isinstance(payload, dict)
                    else ""
                )
                evidence = (
                    payload.get("evidence", []) if isinstance(payload, dict) else []
                )
                summary = {
                    "status": payload.get("status", "failed")
                    if isinstance(payload, dict)
                    else "failed",
                    "outcome": category,
                    "answer": answer,
                    "evidence": [str(item)[:500] for item in evidence[:8]]
                    if isinstance(evidence, list)
                    else [],
                    "error_code": payload.get("error_code")
                    if isinstance(payload, dict)
                    else None,
                    "errors": [message] if message else [],
                }
                result.clear()
                result.update(summary)
