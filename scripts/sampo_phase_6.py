"""Shared policy-neutral Phase 6 experiment definitions and checks."""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.plugins import BasePlugin
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InvocationContext
from google.genai.types import Content, Part
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from fedotmas import MAS
from fedotmas._settings import get_meta_model, resolve_model_config
from fedotmas.common.llm import make_llm
from fedotmas.core.runner import PipelineExecutionError, PipelineResult, run_pipeline
from fedotmas.mas.models import MASConfig
from fedotmas.mcp import StdioMCPServer
from fedotmas.mcp.registry import get_server_descriptions
from fedotmas.maw.builder import frame_instruction

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
OUT_ROOT = ROOT / "artifacts" / "sampo_phase_6"
SERVER_DIR = ROOT / "mcp-servers" / "sampo-phase6"
PREDICTIONS = PUBLIC / "mas_runs"
ARTIFACTS = PUBLIC / "candidate_artifacts"
TASK = (
    "Map each assigned public historical construction work name to three distinct allowed labels. "
    "Produce valid top-3 predictions for every assigned ID. Use only public data and the supplied tools. "
    "Process only the assigned IDs."
)
WORKER_MODEL = os.environ.get("PHASE6_WORKER_MODEL", "openai/gpt-5-mini")
BATCH_SIZE = 20
LIMITS = {
    "timeout_seconds": 300,
    "max_model_calls": 30,
    "max_tool_calls": 60,
    "max_prompt_tokens_per_call": 64000,
}
TOOL_NAMES = {
    "list_methods",
    "prepare_candidates",
    "inspect_candidates",
    "save_default_top3",
    "save_ranked_top3",
    "get_prediction_status",
}
SERVER_NAMES = {"sampo-phase6"}
FORBIDDEN_MANUAL_PATTERNS = [
    r"complementary\s+retrieval",
    r"disagreement",
    r"uncertainty",
    r"semantic\s+(?:review|rerank(?:ing)?)",
    r"candidate[- ]generation\s*(?:→|->|to)\s*(?:reasoning|review)",
    r"fallback\s+policy",
    r"review\s+only",
    r"selective\s+(?:review|reasoning|inspection)",
    r"recommended\s+(?:call|tool)\s+order",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def pilot_rows() -> list[dict[str, str]]:
    return read_rows(PUBLIC / "pilot_inputs.csv")


def labels() -> list[str]:
    return [row["target_label"] for row in read_rows(PUBLIC / "allowed_target_labels.csv")]


def assigned_ids(offset: int, limit: int = BATCH_SIZE) -> list[str]:
    return [row["example_id"] for row in pilot_rows()[offset : offset + limit]]


def prediction_path(run_id: str) -> Path:
    return PREDICTIONS / f"{run_id}.jsonl"


def read_stored(run_id: str) -> list[dict[str, str]]:
    path = prediction_path(run_id)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compose_task(run_id: str, ids: list[str], offset: int) -> str:
    return (
        f"{TASK}\n"
        f"Run ID: {run_id}\n"
        f"Assigned IDs: {json.dumps(ids, ensure_ascii=False)}\n"
        f"Batch offset: {offset}\n"
        "Schema requirements: each assigned ID has exactly three distinct labels from the public allowed-label list.\n"
        "Limits: wall-clock 300 seconds; at most 30 model calls, 60 tool calls, "
        "and 64,000 prompt tokens per call."
    )


def manual_input_audit_text(
    system_prompt: str,
    task: str,
    server_description: str,
    tools: list[dict[str, Any]],
    harness_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "meta_system_prompt": system_prompt,
        "task": task,
        "mcp_servers": [{"name": "sampo-phase6", "description": server_description}],
        "tools": tools,
        "harness_added_instructions": harness_messages,
        "fedotmas_execution_framing": frame_instruction("<generated agent instruction>"),
        "single_agent_instruction": frame_instruction(
            "Complete the supplied task using the available tools."
        ),
    }


def assert_neutral_manual_inputs(payload: dict[str, Any]) -> None:
    """Reject explicit solving-policy language in manually supplied inputs."""
    texts = [
        payload["meta_system_prompt"],
        payload["task"],
        *(server["description"] for server in payload["mcp_servers"]),
        *(tool.get("description", "") for tool in payload["tools"]),
        *(json.dumps(tool.get("inputSchema", {}), ensure_ascii=False) for tool in payload["tools"]),
        *(item["message"] for item in payload["harness_added_instructions"]),
        payload["fedotmas_execution_framing"],
        payload["single_agent_instruction"],
    ]
    for pattern in FORBIDDEN_MANUAL_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [text for text in texts if regex.search(text)]
        if matches:
            raise ValueError(f"Manual Phase 6 input contains forbidden policy language: {pattern}")


def _uv_environment() -> dict[str, str]:
    from mcp.client.stdio import get_default_environment

    return {
        **get_default_environment(),
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        },
    }


async def exposed_tools() -> list[dict[str, Any]]:
    """Read the actual Phase 6 MCP tool names, descriptions, and schemas."""
    uv = shutil.which("uv") or "uv"
    params = StdioServerParameters(
        command=uv,
        args=["run", "--directory", str(SERVER_DIR), "mcp-sampo-phase6"],
        env=_uv_environment(),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
    tools = [item.model_dump(mode="json", exclude_none=True) for item in result.tools]
    found = {item["name"] for item in tools}
    if found != TOOL_NAMES:
        raise RuntimeError(f"Unexpected Phase 6 tool surface: {sorted(found)}")
    return tools


def phase6_registry(offset: int, ids: list[str], run_id: str) -> dict[str, Any]:
    mas = MAS(mcp_servers=["sampo-phase6"], worker_models=[WORKER_MODEL])
    registry = mas.mcp_registry
    current = registry["sampo-phase6"]
    if not isinstance(current, StdioMCPServer):
        raise TypeError("sampo-phase6 must be a local stdio server")
    registry["sampo-phase6"] = replace(
        current,
        env={
            **current.env,
            "PHASE6_ASSIGNED_IDS": ",".join(ids),
            "PHASE6_OFFSET": str(offset),
            "PHASE6_RUN_ID": run_id,
        },
    )
    return registry


class BatchTrace(BasePlugin):
    """Record and enforce Phase 6 infrastructure budgets for one batch."""

    def __init__(
        self,
        ids: list[str],
        offset: int,
        run_id: str,
        delegate_names: set[str] | None = None,
    ):
        super().__init__(name="phase6_batch_trace")
        self.ids = set(ids)
        self.offset = offset
        self.run_id = run_id
        self.delegate_names = delegate_names or set()
        self.model_calls: list[dict[str, Any]] = []
        self.model_request_attempts = 0
        self.blocked_model_attempts = 0
        self.prompt_estimates: list[int] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.persistence_results: list[dict[str, Any]] = []
        self.tool_errors: list[dict[str, Any]] = []
        self.agents: list[str] = []
        self.failures: list[str] = []
        self.containment_violations: list[dict[str, Any]] = []
        self.idempotent_write_retries: list[str] = []
        self.conflicting_rewrites: list[str] = []
        self.successful_conflicting_rewrites: list[str] = []
        self.write_intents: dict[str, tuple[str, str, str]] = {}
        self.pending_write_intents: dict[str, dict[str, tuple[str, str, str]]] = {}
        self.pending_tool_names: dict[str, str] = {}
        self.blocked_tool_attempts = 0
        self.prepared_artifacts: set[str] = set()
        self.blocked_reason: str | None = None
        self.terminal = False
        self.terminal_reason: str | None = None
        self.terminal_snapshot: dict[str, int] | None = None
        self.terminal_suppressed_model_attempts = 0
        self.terminal_suppressed_tool_attempts = 0
        self.invocation_contexts: dict[str, InvocationContext] = {}

    async def before_agent_callback(self, *, agent, callback_context):
        self.agents.append(agent.name)

    async def before_model_callback(self, *, callback_context, llm_request):
        invocation_context = self._context_from(callback_context)
        if self._durable_batch_complete():
            self._mark_terminal(invocation_context)
        if self.terminal:
            self.terminal_suppressed_model_attempts += 1
            return self._terminal_response()
        if self.blocked_reason:
            self.blocked_model_attempts += 1
            return self._budget_response(self.blocked_reason)
        if self.model_request_attempts >= LIMITS["max_model_calls"]:
            self.failures.append("model_call_limit")
            self.blocked_reason = "model_call_limit"
            self.blocked_model_attempts += 1
            return self._budget_response(self.blocked_reason)
        estimate = _estimate_request_tokens(llm_request)
        if estimate > LIMITS["max_prompt_tokens_per_call"]:
            self.failures.append("prompt_token_limit")
            self.blocked_reason = "prompt_token_limit"
            self.prompt_estimates.append(estimate)
            self.blocked_model_attempts += 1
            return self._budget_response(self.blocked_reason)
        self.prompt_estimates.append(estimate)
        self.model_request_attempts += 1

    @staticmethod
    def _budget_response(reason: str) -> LlmResponse:
        return LlmResponse(
            content=Content(
                role="model",
                parts=[Part(text=f"Phase 6 execution limit reached ({reason}). Stop execution.")],
            ),
            turn_complete=True,
        )

    @staticmethod
    def _terminal_response() -> LlmResponse:
        return LlmResponse(
            content=Content(
                role="model",
                parts=[Part(text="The assigned batch is complete.")],
            ),
            turn_complete=True,
        )

    async def after_model_callback(
        self, *, callback_context, llm_response: LlmResponse
    ):
        usage = llm_response.usage_metadata
        prompt = (usage.prompt_token_count if usage else 0) or 0
        completion = (usage.candidates_token_count if usage else 0) or 0
        self.model_calls.append(
            {
                "agent": callback_context.agent_name,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            }
        )
        if prompt > LIMITS["max_prompt_tokens_per_call"]:
            if "prompt_token_limit" not in self.failures:
                self.failures.append("prompt_token_limit")
            self.blocked_reason = "prompt_token_limit"
            return self._budget_response(self.blocked_reason)

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        invocation_context = self._context_from(tool_context)
        if self._durable_batch_complete():
            self._mark_terminal(invocation_context)
        if self.terminal:
            self.terminal_suppressed_tool_attempts += 1
            return {"status": "assigned_batch_complete"}
        if self.blocked_reason:
            self.blocked_tool_attempts += 1
            return {"error": f"Phase 6 execution limit reached ({self.blocked_reason}). Stop execution."}
        if len(self.tool_calls) >= LIMITS["max_tool_calls"]:
            self.failures.append("tool_call_limit")
            self.blocked_reason = "tool_call_limit"
            self.blocked_tool_attempts += 1
            return {"error": "Phase 6 tool-call limit reached. Stop execution."}

        name = tool.name or ""
        args = tool_args or {}
        if name not in self.delegate_names and name not in TOOL_NAMES:
            self.failures.append("inaccessible_tool_name")
            return {"error": f"Tool is not in the Phase 6 catalogue: {name}"}
        self._record_tool_call(name, args)
        call_id = getattr(tool_context, "function_call_id", None)
        if call_id:
            self.pending_write_intents[call_id] = _write_intents(name, args)
            self.pending_tool_names[call_id] = name

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        call_id = getattr(tool_context, "function_call_id", None)
        name = self.pending_tool_names.pop(call_id, tool.name or "unknown") if call_id else (tool.name or "unknown")
        intents = self.pending_write_intents.pop(call_id, {}) if call_id else {}
        error = _find_tool_error(result)
        if error is not None:
            self._record_tool_error(name, tool_args or {}, error)
        status = _find_write_status(result)
        if status is not None:
            self.persistence_results.append({"name": name, **status})
            stored_ids = set(status.get("stored_ids", []))
            identical_ids = set(status.get("already_identical_ids", []))
            for example_id in identical_ids:
                self.idempotent_write_retries.append(example_id)
            for example_id in stored_ids | identical_ids:
                if example_id in intents:
                    previous = self.write_intents.get(example_id)
                    if previous is not None and previous != intents[example_id]:
                        self.successful_conflicting_rewrites.append(example_id)
                    self.write_intents[example_id] = intents[example_id]
        if error is None:
            artifact_id = None
            if name == "prepare_candidates":
                artifact_id = _find_artifact_id(result)
            elif name in {"inspect_candidates", "save_default_top3", "save_ranked_top3"}:
                artifact_id = (tool_args or {}).get("artifact_id")
            if artifact_id:
                self.prepared_artifacts.add(artifact_id)

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        name = tool.name or "unknown"
        self._record_tool_error(name, tool_args or {}, str(error))

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ):
        self._remember_invocation_context(invocation_context)
        if event.partial:
            return
        for function_response in event.get_function_responses():
            response = function_response.response or {}
            response_name = getattr(function_response, "name", None) or "unknown"
            error = _find_tool_error(response)
            if error is not None:
                self._record_tool_error(response_name, {}, error)
            artifact_id = _find_artifact_id(response) if response_name == "prepare_candidates" else None
            if error is None and artifact_id:
                self.prepared_artifacts.add(artifact_id)
        if self._durable_batch_complete():
            self._mark_terminal(invocation_context)

    @staticmethod
    def _context_from(context: Any) -> InvocationContext | None:
        if context is None:
            return None
        getter = getattr(context, "get_invocation_context", None)
        if callable(getter):
            try:
                return getter()
            except (AttributeError, RuntimeError):
                return None
        return getattr(context, "_invocation_context", None)

    def _remember_invocation_context(self, invocation_context: InvocationContext | None) -> None:
        if invocation_context is None:
            return
        key = str(getattr(invocation_context, "invocation_id", id(invocation_context)))
        self.invocation_contexts[key] = invocation_context

    def _durable_batch_complete(self) -> bool:
        try:
            rows = read_stored(self.run_id)
        except (OSError, json.JSONDecodeError):
            return False
        return not _schema_failures(rows, sorted(self.ids))

    def _mark_terminal(self, invocation_context: InvocationContext | None = None) -> None:
        self._remember_invocation_context(invocation_context)
        if not self.terminal:
            self.terminal = True
            self.terminal_reason = "assigned_batch_complete"
            self.terminal_snapshot = {
                "model_calls": len(self.model_calls),
                "prompt_tokens": sum(item["prompt_tokens"] for item in self.model_calls),
                "completion_tokens": sum(item["completion_tokens"] for item in self.model_calls),
                "worker_delegations": sum(
                    item.get("kind") == "worker_call" for item in self.tool_calls
                ),
                "mcp_tool_calls": sum(
                    item.get("kind") != "worker_call" for item in self.tool_calls
                ),
            }
        for context in self.invocation_contexts.values():
            context.end_invocation = True

    def _record_tool_error(self, name: str, args: dict[str, Any], error: str) -> None:
        message = error[:1000]
        entry = {"name": name, "error": message}
        if entry not in self.tool_errors:
            self.tool_errors.append(entry)
        if "conflicts with an existing stored top-3" not in message.casefold():
            return
        if name == "save_default_top3":
            ids = list(args.get("example_ids", []))
        elif name == "save_ranked_top3":
            ids = [
                item.get("example_id")
                for item in args.get("rankings", [])
                if isinstance(item, dict)
            ]
        else:
            ids = []
        matched = [
            example_id
            for example_id in ids
            if example_id is not None
            and re.search(rf"(?<![\w]){re.escape(str(example_id))}(?![\w])", message)
        ]
        self.conflicting_rewrites.extend(matched)

    def _record_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if name in self.delegate_names:
            self.tool_calls.append({"name": name, "kind": "worker_call", "ids": []})
            return

        ids: list[str] = []
        if name in {"inspect_candidates", "save_default_top3", "get_prediction_status"}:
            ids = list(args.get("example_ids") or [])
        elif name == "save_ranked_top3":
            ids = [
                item.get("example_id") if isinstance(item, dict) else None
                for item in args.get("rankings", [])
            ]
        if ids and not set(ids) <= self.ids:
            self.containment_violations.append(
                {"tool": name, "ids": ids, "reason": "IDs outside assigned batch"}
            )
        if name == "prepare_candidates" and (
            args.get("offset") != self.offset or args.get("limit") != len(self.ids)
        ):
            self.containment_violations.append(
                {
                    "tool": name,
                    "offset": args.get("offset"),
                    "limit": args.get("limit"),
                    "reason": "Batch parameters outside assigned scope",
                }
            )
        if name in {"save_default_top3", "save_ranked_top3", "get_prediction_status"} and args.get("run_id") != self.run_id:
            self.containment_violations.append(
                {"tool": name, "reason": "Run ID outside assigned scope"}
            )
        artifact_id = args.get("artifact_id")
        self.tool_calls.append(
            {
                "name": name,
                "ids": ids,
                "artifact_id": artifact_id,
                "arguments": args,
            }
        )


def _schema_failures(rows: list[dict[str, str]], expected: list[str]) -> list[str]:
    allowed = set(labels())
    by_id: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    for row in rows:
        example_id = row.get("example_id", "")
        values = [row.get(f"top_{rank}", "") for rank in range(1, 4)]
        if not example_id or example_id in by_id:
            failures.append("invalid_prediction_schema")
            continue
        if len(set(values)) != 3 or not set(values) <= allowed:
            failures.append("invalid_prediction_schema")
        by_id[example_id] = row
    stored = set(by_id)
    assigned = set(expected)
    if stored - assigned:
        failures.append("successful_write_outside_assigned_batch")
    if assigned - stored:
        failures.append("incomplete_assigned_batch_coverage")
    return sorted(set(failures))


def _find_artifact_id(value: Any) -> str | None:
    if isinstance(value, dict):
        artifact_id = value.get("artifact_id")
        if isinstance(artifact_id, str):
            return artifact_id
        for item in value.values():
            found = _find_artifact_id(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_artifact_id(item)
            if found:
                return found
    return None


def _find_tool_error(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("isError") is True or value.get("is_error") is True:
            texts = _collect_text(value)
            return "; ".join(texts)[:1000] or "MCP tool returned an error"
        error = value.get("error")
        if error:
            return str(error)[:1000]
        for item in value.values():
            found = _find_tool_error(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_tool_error(item)
            if found is not None:
                return found
    elif isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            return _find_tool_error(json.loads(value))
        except json.JSONDecodeError:
            return None
    return None


def _collect_text(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, str):
            found.append(text_value)
        for key, item in value.items():
            if key != "text":
                found.extend(_collect_text(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_text(item))
    return found


def _estimate_request_tokens(llm_request: Any) -> int:
    """Conservatively estimate the full prepared request before sending it."""
    try:
        payload = llm_request.model_dump_json(exclude_none=True)
    except Exception:
        payload = str(llm_request)
    model_name = getattr(llm_request, "model", None) or WORKER_MODEL
    try:
        from litellm import token_counter

        return int(token_counter(model=model_name, text=payload))
    except Exception:
        # JSON punctuation is counted too, making this fallback conservative.
        return (len(payload) + 2) // 3


def _write_intents(name: str, args: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    """Resolve requested writes to labels for retry/conflict telemetry only."""
    artifact_id = args.get("artifact_id")
    if not isinstance(artifact_id, str):
        return {}
    path = ARTIFACTS / f"{artifact_id}.json"
    if not path.exists():
        return {}
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    examples = {row.get("example_id"): row for row in artifact.get("examples", [])}
    result: dict[str, tuple[str, str, str]] = {}
    if name == "save_default_top3":
        for example_id in args.get("example_ids", []):
            candidates = examples.get(example_id, {}).get("fused_candidates", [])
            if len(candidates) >= 3:
                result[example_id] = tuple(candidate["label"] for candidate in candidates[:3])
    elif name == "save_ranked_top3":
        for ranking in args.get("rankings", []):
            if not isinstance(ranking, dict):
                continue
            example_id = ranking.get("example_id")
            indices = ranking.get("candidate_indices", [])
            candidates = examples.get(example_id, {}).get("fused_candidates", [])
            by_index = {row.get("candidate_index"): row.get("label") for row in candidates}
            if len(indices) == 3 and all(index in by_index for index in indices):
                result[example_id] = tuple(by_index[index] for index in indices)
    return result


def _find_write_status(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "stored_ids" in value or "already_identical_ids" in value:
            return {
                key: value[key]
                for key in ("stored", "stored_ids", "already_identical", "already_identical_ids", "total")
                if key in value
            }
        for item in value.values():
            found = _find_write_status(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_write_status(item)
            if found is not None:
                return found
    return None


def _artifact_integrity(trace: BatchTrace) -> list[str]:
    failures: list[str] = []
    for artifact_id in trace.prepared_artifacts:
        path = ARTIFACTS / f"{artifact_id}.json"
        if not path.exists():
            failures.append("artifact_missing")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("artifact_unreadable")
            continue
        artifact_ids = {row.get("example_id") for row in data.get("examples", [])}
        parameters = data.get("parameters", {})
        if (
            data.get("artifact_id") != artifact_id
            or artifact_ids != trace.ids
            or parameters.get("offset") != trace.offset
            or parameters.get("limit") != len(trace.ids)
        ):
            failures.append("artifact_lineage_integrity")
    return failures


def qualify_trace(trace: BatchTrace, ids: list[str], runtime_seconds: float) -> dict[str, Any]:
    rows = read_stored(trace.run_id)
    failures = list(trace.failures)
    failures.extend(_schema_failures(rows, ids))
    if trace.successful_conflicting_rewrites:
        failures.append("conflicting_write_succeeded")
    if runtime_seconds >= LIMITS["timeout_seconds"]:
        failures.append("runtime_limit")
    failures.extend(_artifact_integrity(trace))
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "assigned_ids": ids,
        "stored_ids": sorted(row["example_id"] for row in rows),
        "model_calls": len(trace.model_calls),
        "model_call_attempts": trace.model_request_attempts,
        "blocked_model_attempts": trace.blocked_model_attempts,
        "terminal_suppressed_model_attempts": trace.terminal_suppressed_model_attempts,
        "tool_calls": len(trace.tool_calls),
        "blocked_tool_attempts": trace.blocked_tool_attempts,
        "terminal_suppressed_tool_attempts": trace.terminal_suppressed_tool_attempts,
        "mcp_tool_calls": sum(item.get("kind") != "worker_call" for item in trace.tool_calls),
        "worker_delegations": sum(item.get("kind") == "worker_call" for item in trace.tool_calls),
        "prompt_tokens": sum(item["prompt_tokens"] for item in trace.model_calls),
        "completion_tokens": sum(item["completion_tokens"] for item in trace.model_calls),
        "max_prompt_tokens": max((item["prompt_tokens"] for item in trace.model_calls), default=0),
        "max_estimated_prompt_tokens": max(trace.prompt_estimates, default=0),
        "runtime_seconds": runtime_seconds,
        "agent_names": trace.agents,
        "tool_trace": trace.tool_calls,
        "persistence_results": trace.persistence_results,
        "tool_errors": trace.tool_errors,
        "batch_containment_violations": trace.containment_violations,
        "idempotent_write_retries": trace.idempotent_write_retries,
        "conflicting_rewrites": sorted(set(trace.conflicting_rewrites)),
        "successful_conflicting_rewrites": sorted(set(trace.successful_conflicting_rewrites)),
        "terminal": trace.terminal,
        "terminal_reason": trace.terminal_reason,
        "model_calls_at_completion": (trace.terminal_snapshot or {}).get("model_calls"),
        "prompt_tokens_at_completion": (trace.terminal_snapshot or {}).get("prompt_tokens"),
        "completion_tokens_at_completion": (trace.terminal_snapshot or {}).get("completion_tokens"),
        "worker_delegations_at_completion": (trace.terminal_snapshot or {}).get("worker_delegations"),
        "mcp_tool_calls_at_completion": (trace.terminal_snapshot or {}).get("mcp_tool_calls"),
        "private_ground_truth_used": False,
        "private_ground_truth_accesses": 0,
    }


async def run_generated_batch(
    config: MASConfig, ids: list[str], offset: int, run_id: str
) -> dict[str, Any]:
    trace = BatchTrace(
        ids,
        offset,
        run_id,
        delegate_names={worker.name for worker in config.workers},
    )
    registry = phase6_registry(offset, ids, run_id)
    mas = MAS(
        worker_models=[WORKER_MODEL],
        mcp_servers=registry,
        plugins=[trace],
        max_retries=0,
    )
    started = time.monotonic()
    try:
        await mas.build_and_run(
            config,
            compose_task(run_id, ids, offset),
            timeout=LIMITS["timeout_seconds"],
        )
    except Exception as exc:  # returned traces determine qualification
        trace.failures.append(f"execution_error:{type(exc).__name__}")
    result = qualify_trace(trace, ids, time.monotonic() - started)
    result["run_id"] = run_id
    return result


async def run_single_agent_batch(ids: list[str], offset: int, run_id: str) -> dict[str, Any]:
    trace = BatchTrace(ids, offset, run_id)
    registry = phase6_registry(offset, ids, run_id)
    model = make_llm(resolve_model_config(WORKER_MODEL))
    agent = LlmAgent(
        name="phase6_single_agent",
        model=model,
        instruction=frame_instruction("Complete the supplied task using the available tools."),
        tools=[_toolset(registry)],
    )
    app = App(name="phase6_single_agent", root_agent=agent, plugins=[trace])
    started = time.monotonic()
    try:
        result: PipelineResult = await run_pipeline(
            app,
            compose_task(run_id, ids, offset),
            session_service=InMemorySessionService(),
            timeout=LIMITS["timeout_seconds"],
        )
    except PipelineExecutionError as exc:
        trace.failures.append(f"execution_error:{type(exc).__name__}")
    report = qualify_trace(trace, ids, time.monotonic() - started)
    report["run_id"] = run_id
    return report


def _toolset(registry: dict[str, Any]):
    from fedotmas.mcp import create_toolset

    return create_toolset("sampo-phase6", registry=registry)


def format_tool_catalogue(
    server_description: str, tools: list[dict[str, Any]]
) -> str:
    blocks = [f"MCP SERVER: sampo-phase6\nDescription: {server_description}"]
    for tool in tools:
        blocks.append(
            "TOOL: "
            + tool["name"]
            + "\nDescription: "
            + tool.get("description", "")
            + "\nInput schema: "
            + json.dumps(tool.get("inputSchema", {}), ensure_ascii=False, sort_keys=True)
        )
    return "\n\n".join(blocks)


async def build_audit_inputs(
    system_prompt: str,
    *,
    run_prefix: str,
    smoke_runs: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = await exposed_tools()
    server_description = get_server_descriptions(MAS(mcp_servers=["sampo-phase6"]).mcp_registry)["sampo-phase6"]
    messages: list[dict[str, Any]] = []
    qual_ids = assigned_ids(0, 5)
    for index in range(1, 6):
        run_id = f"{run_prefix}_qual_{index:02d}"
        messages.append(
            {
                "purpose": f"behavioral qualification config {index:02d}",
                "message": compose_task(run_id, qual_ids, 0),
            }
        )
    if smoke_runs:
        for system in ("single", "mas"):
            for offset in (0, 20, 40):
                run_id = f"{run_prefix}_{system}_{offset:04d}"
                messages.append(
                    {
                        "purpose": f"{system} smoke offset {offset}",
                        "message": compose_task(run_id, assigned_ids(offset), offset),
                    }
                )
    payload = manual_input_audit_text(
        system_prompt, TASK, server_description, tools, messages
    )
    assert_neutral_manual_inputs(payload)
    return payload, tools


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
