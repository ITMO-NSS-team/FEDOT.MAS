"""Structured, per-agent research metrics for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.mcp import strip_tool_name_prefix
from fedotmas.mcp.capabilities import (
    ToolCapability,
    is_inspection_tool,
    normalize_tool_name,
    tool_capability,
)
from fedotmas.plugins._tool_error_circuit_breaker import (
    DUPLICATE_TOOL_CALL,
    TOOL_CIRCUIT_OPEN,
    WEB_BUDGET_EXHAUSTED,
)

CONTROL_CODES = frozenset(
    {
        DUPLICATE_TOOL_CALL,
        WEB_BUDGET_EXHAUSTED,
        TOOL_CIRCUIT_OPEN,
        "INSPECT_CANDIDATES_FIRST",
        "DISCOVERY_FANOUT_LIMIT",
        "SOURCE_CANDIDATES_READY",
        "RESEARCH_CONVERGENCE_REQUIRED",
        "EVIDENCE_FIRST_SEARCH_DISABLED",
        "INSPECTION_ONLY_DISCOVERY_DISABLED",
        "DISCOVERY_ONLY_INSPECTION_DISABLED",
    }
)
RESEARCH_GATE_STATE_KEY = "__fedotmas_research_gate"
RESEARCH_MODE_STATE_KEY = "__fedotmas_research_modes"
RESEARCH_POLICY_STATE_KEY = "__fedotmas_research_policies"
RESEARCH_CANDIDATE_LEDGER_KEY = "__fedotmas_research_candidates"
RESEARCH_INSPECTED_SOURCES_KEY = "__fedotmas_inspected_sources"
RESEARCH_PROGRESS_STATE_KEY = "__fedotmas_research_progress"
RESEARCH_TURN_STATE_KEY = "__fedotmas_research_turns"
MAX_CANDIDATES_PER_AGENT = 24
MAX_CANDIDATE_TITLE_CHARS = 180
MAX_CANDIDATE_SNIPPET_CHARS = 420
MAX_CANDIDATE_INSPECTION_PROGRESS = 3


def _read_gate_state(state: Any, agent: str) -> dict[str, Any]:
    root = state.get(RESEARCH_GATE_STATE_KEY) if hasattr(state, "get") else None
    gate = root.get(agent) if isinstance(root, dict) else None
    if not isinstance(gate, dict):
        return {}
    # Read old snapshots from the preceding threshold-based implementation safely.
    return {**gate, "phase": gate.get("phase", "inspect" if gate.get("gated") else "discover")}


def _tool_failed(result: dict[str, Any]) -> bool:
    return bool(
        result.get("isError") is True
        or result.get("is_error") is True
        or result.get("error")
    )


def _inspection_has_content(value: Any, *, depth: int = 0) -> bool:
    """Whether a successful inspection returned text rather than an empty shell."""
    if depth > 5:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_inspection_has_content(item, depth=depth + 1) for item in value[:12])
    if not isinstance(value, dict):
        return False
    content_keys = {
        "content",
        "text",
        "markdown",
        "body",
        "excerpt",
        "title",
        "name",
        "structuredcontent",
        "structured_content",
        "result",
    }
    return any(
        _inspection_has_content(nested, depth=depth + 1)
        for key, nested in value.items()
        if str(key).casefold() in content_keys
    )


class ResearchTelemetry(BasePlugin):
    """Collect tool outcomes and token usage without inspecting log text."""

    def __init__(self, name: str = "fedotmas_research_telemetry") -> None:
        super().__init__(name=name)
        self._agents: dict[str, dict[str, Any]] = defaultdict(self._new_agent)
        self._queries: dict[str, set[str]] = defaultdict(set)
        self._query_fingerprints: dict[str, set[str]] = defaultdict(set)
        self._discovered: dict[str, set[str]] = defaultdict(set)
        self._inspected: dict[str, set[str]] = defaultdict(set)
        self._search_query_seen: dict[str, set[str]] = defaultdict(set)
        self._calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.max_call_diagnostics = 300

    @staticmethod
    def _new_agent() -> dict[str, Any]:
        return {
            "attempted_calls": 0,
            "blocked_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "search_calls": 0,
            "successful_searches": 0,
            "unique_queries": 0,
            "duplicate_blocks": 0,
            "circuit_open_blocks": 0,
            "zero_result_searches": 0,
            "backend_errors": 0,
            "urls_discovered": 0,
            "urls_inspected": 0,
            "discovery_calls": 0,
            "candidate_urls_discovered": 0,
            "candidate_urls_inspected": 0,
            "discovery_calls_yielding_new_candidates": 0,
            "repeated_query_search_no_new_evidence_events": 0,
            "searches_with_no_new_candidates": 0,
            "no_progress_turns": 0,
            "discovery_gated": 0,
            "discovery_reopened": 0,
            "repeated_discovery_without_inspection": 0,
            "scraping_extraction_calls": 0,
            "search_exhaustion": 0,
            "scraping_exhaustion": 0,
            "browser_agent_calls": 0,
            "browser_agent_exhaustion": 0,
            "browser_agent_prompt_tokens": 0,
            "browser_agent_completion_tokens": 0,
            "browser_agent_total_tokens": 0,
            "browser_agent_llm_invocations": 0,
            "browser_agent_steps": 0,
            "browser_agent_usage_missing": 0,
            "code_agent_calls": 0,
            "code_agent_completed_calls": 0,
            "code_agent_incomplete_calls": 0,
            "code_agent_blocked_calls": 0,
            "code_agent_failed_calls": 0,
            "code_agent_steps": 0,
            "code_agent_execution_failures": 0,
            "code_agent_timeouts": 0,
            "code_agent_files_accessed": 0,
            "code_agent_prompt_tokens": 0,
            "code_agent_completion_tokens": 0,
            "code_agent_total_tokens": 0,
            "code_agent_llm_invocations": 0,
            "code_agent_cost_usd": 0.0,
            "code_agent_usage_missing": 0,
            "code_agent_duration_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def attempt(self, agent: str, kind: str, args: dict[str, Any]) -> None:
        metrics = self._agents[agent]
        metrics["attempted_calls"] += 1
        if kind == "search":
            metrics["search_calls"] += 1
            query = args.get("query")
            if isinstance(query, str) and query.strip():
                normalized_query = query.strip().casefold()
                self._queries[agent].add(normalized_query)
                self._query_fingerprints[agent].add(
                    hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
                )
                metrics["unique_queries"] = len(self._queries[agent])
        elif kind == "browser_agent":
            metrics["browser_agent_calls"] += 1
        elif kind == "code_agent":
            metrics["code_agent_calls"] += 1
        else:
            metrics["scraping_extraction_calls"] += 1

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        kind = _research_tool_kind(tool.name)
        if kind is None and tool_capability(tool.name, description=getattr(tool, "description", "") or "") == ToolCapability.DISCOVERY:
            kind = "search"
        if kind is None:
            return None
        agent = tool_context._invocation_context.agent.name
        self.attempt(agent, kind, tool_args)
        calls = self._calls[agent]
        calls.append(
            {
                "agent": agent,
                "tool": strip_tool_name_prefix(tool.name),
                "query": _sanitize_query(tool_args.get("query")),
                "url": _sanitize_url(tool_args.get("url") or tool_args.get("uri")),
                "attempted": True,
                "status": "attempted",
                "error_category": None,
                "started": time.monotonic(),
                "elapsed_ms": None,
                "result_chars": None,
                "result_count": None,
                "_call_id": _tool_call_id(tool_context),
            }
        )
        if len(calls) > self.max_call_diagnostics:
            del calls[: len(calls) - self.max_call_diagnostics]
        if kind == "search":
            metrics = self._agents[agent]
            metrics["discovery_calls"] += 1
            gate = _read_gate_state(tool_context.state, agent)
            if _research_mode(tool_context.state, agent) != "mixed":
                return None
            if gate.get("phase") == "inspect":
                metrics["repeated_discovery_without_inspection"] += 1
                metrics["blocked_calls"] += 1
                calls[-1]["status"] = "blocked"
                calls[-1]["error_category"] = "inspect_candidates_first"
                return {
                    "isError": True,
                    "error_code": "INSPECT_CANDIDATES_FIRST",
                    "error": (
                        "Discovery is paused. Inspect one pending candidate URL "
                        "before searching again."
                    ),
                }
        return None

    def record_blocked(
        self,
        agent: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        category: str,
        budget: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> None:
        call_id = call_id if isinstance(call_id, str) else None
        item = self._pending_call(agent, tool_name, args, call_id=call_id)
        if item is not None:
            item["status"] = "blocked"
            item["error_category"] = category
            item["elapsed_ms"] = max(
                0,
                int((time.monotonic() - item.pop("started", time.monotonic())) * 1000),
            )
            if budget is not None:
                item["budget"] = dict(budget)

    def record_budget(
        self,
        agent: str,
        tool_name: str,
        args: dict[str, Any],
        budget: dict[str, Any],
        call_id: str | None = None,
    ) -> None:
        call_id = call_id if isinstance(call_id, str) else None
        item = self._pending_call(agent, tool_name, args, call_id=call_id)
        if item is not None:
            item["budget"] = dict(budget)

    def duplicate(self, agent: str) -> None:
        metrics = self._agents[agent]
        metrics["duplicate_blocks"] += 1
        metrics["blocked_calls"] += 1

    def exhausted(self, agent: str, kind: str) -> None:
        metrics = self._agents[agent]
        metrics[f"{kind}_exhaustion"] += 1
        metrics["blocked_calls"] += 1

    def budget_blocked(self, agent: str) -> None:
        self._agents[agent]["blocked_calls"] += 1

    def blocked(self, agent: str) -> None:
        self._agents[agent]["blocked_calls"] += 1

    def circuit_blocked(self, agent: str, tool_name: str) -> None:
        del tool_name
        metrics = self._agents[agent]
        metrics["blocked_calls"] += 1
        metrics["circuit_open_blocks"] += 1

    def inspected(self, agent: str, url: str) -> None:
        candidate = _sanitize_url(url) or url
        if candidate not in self._discovered[agent]:
            return
        self._inspected[agent].add(candidate)
        self._agents[agent]["urls_inspected"] = len(self._inspected[agent])
        self._agents[agent]["candidate_urls_inspected"] = len(self._inspected[agent])

    def _add_candidates(
        self,
        state: Any,
        agent: str,
        tool_name: str,
        query: Any,
        results: list[Any],
    ) -> tuple[list[str], bool]:
        if not hasattr(state, "get") or not hasattr(state, "__setitem__"):
            return [], False
        root = state.get(RESEARCH_CANDIDATE_LEDGER_KEY)
        if not isinstance(root, dict):
            root = {}
        existing = root.get(agent)
        ledger = [
            dict(item)
            for item in existing
            if isinstance(item, dict) and isinstance(item.get("url"), str)
        ] if isinstance(existing, list) else []
        by_url = {item["url"]: item for item in ledger}
        new_urls: list[str] = []
        next_update = max(
            (item.get("updated", -1) for item in ledger if isinstance(item.get("updated"), int)),
            default=-1,
        ) + 1
        normalized_tool = normalize_tool_name(strip_tool_name_prefix(tool_name))
        safe_query = _sanitize_query(query)
        for raw in results:
            if not isinstance(raw, dict) or not isinstance(raw.get("url"), str):
                continue
            url = _sanitize_url(raw["url"])
            if not url:
                continue
            item = by_url.get(url)
            if item is None:
                item = {
                    "url": url,
                    "title": "",
                    "snippet": "",
                    "source_tool": normalized_tool[:80],
                    "query": safe_query,
                    "inspected": False,
                    "inspection_status": "uninspected",
                    "updated": len(ledger) + len(new_urls),
                }
                by_url[url] = item
                new_urls.append(url)
            elif item in ledger:
                ledger.remove(item)
            item["updated"] = next_update
            next_update += 1
            ledger.append(item)
            if not item.get("title"):
                item["title"] = str(raw.get("title") or "")[:MAX_CANDIDATE_TITLE_CHARS]
            if not item.get("snippet"):
                snippet = raw.get("snippet") or raw.get("content") or raw.get("description") or ""
                item["snippet"] = str(snippet)[:MAX_CANDIDATE_SNIPPET_CHARS]
            item["source_tool"] = normalized_tool[:80]
            item["query"] = safe_query
        if len(ledger) > MAX_CANDIDATES_PER_AGENT:
            # Retain inspected evidence and the most recent uninspected sources.
            inspected = [item for item in ledger if item.get("inspected") is True]
            uninspected = [item for item in ledger if item.get("inspected") is not True]
            ledger = (inspected[-8:] + uninspected[-(MAX_CANDIDATES_PER_AGENT - min(8, len(inspected))):])
            ledger.sort(key=lambda item: int(item.get("updated", 0)))
        root[agent] = ledger
        state[RESEARCH_CANDIDATE_LEDGER_KEY] = root
        return new_urls, bool(ledger)

    def _mark_source_inspected(self, state: Any, agent: str, url: str) -> bool:
        if not hasattr(state, "get") or not hasattr(state, "__setitem__"):
            return False
        root = state.get(RESEARCH_INSPECTED_SOURCES_KEY)
        if not isinstance(root, dict):
            root = {}
        urls = root.get(agent)
        urls = urls if isinstance(urls, list) else []
        normalized = _sanitize_url(url)
        if not normalized or normalized in urls:
            return False
        root[agent] = [*urls, normalized][-64:]
        state[RESEARCH_INSPECTED_SOURCES_KEY] = root
        return True

    def _mark_candidate_inspected(self, state: Any, agent: str, url: str, *, status: str = "success", tool: str = "", error: str = "") -> bool:
        root = state.get(RESEARCH_CANDIDATE_LEDGER_KEY) if hasattr(state, "get") else None
        ledger = root.get(agent) if isinstance(root, dict) else None
        if not isinstance(ledger, list):
            return False
        candidate_url = _sanitize_url(url)
        for item in ledger:
            if isinstance(item, dict) and item.get("url") == candidate_url:
                was_usefully_inspected = item.get("inspection_status") == "success"
                item["inspected"] = True
                item["inspection_status"] = status
                item["inspection_tool"] = tool[:80]
                if error:
                    item["inspection_error"] = error[:240]
                else:
                    item.pop("inspection_error", None)
                return not was_usefully_inspected
        return False

    def _mark_progress(self, state: Any, agent: str, kind: str) -> None:
        if not hasattr(state, "get") or not hasattr(state, "__setitem__"):
            return
        root = state.get(RESEARCH_PROGRESS_STATE_KEY)
        if not isinstance(root, dict):
            root = {}
        progress = root.get(agent)
        if not isinstance(progress, dict):
            progress = {"version": 0, "progress_events": []}
        progress["version"] = int(progress.get("version", 0)) + 1
        events = progress.get("progress_events")
        events = events if isinstance(events, list) else []
        events.append(kind)
        progress["progress_events"] = events[-20:]
        root[agent] = progress
        state[RESEARCH_PROGRESS_STATE_KEY] = root

    def _mark_candidate_inspection_progress(
        self, state: Any, agent: str, url: str
    ) -> bool:
        root = state.get(RESEARCH_PROGRESS_STATE_KEY)
        if not isinstance(root, dict):
            root = {}
        progress = root.get(agent)
        if not isinstance(progress, dict):
            progress = {"version": 0, "progress_events": []}
        count = progress.get("candidate_inspection_progress_count", 0)
        count = count if isinstance(count, int) and not isinstance(count, bool) else 0
        urls = progress.get("candidate_inspection_progress_urls", [])
        urls = urls if isinstance(urls, list) else []
        candidate_url = _sanitize_url(url)
        if (
            not candidate_url
            or candidate_url in urls
            or count >= MAX_CANDIDATE_INSPECTION_PROGRESS
        ):
            return False
        progress["candidate_inspection_progress_count"] = count + 1
        progress["candidate_inspection_progress_urls"] = [*urls, candidate_url][
            -MAX_CANDIDATE_INSPECTION_PROGRESS:
        ]
        root[agent] = progress
        state[RESEARCH_PROGRESS_STATE_KEY] = root
        self._mark_progress(state, agent, "candidate_inspection")
        return True

    def _sync_gate_state(
        self,
        state: Any,
        agent: str,
        *,
        discovered: list[str] | None = None,
        inspected_url: str | None = None,
    ) -> None:
        root = state.get(RESEARCH_GATE_STATE_KEY) if hasattr(state, "get") else None
        if not isinstance(root, dict):
            root = {}
        old = root.get(agent)
        old = old if isinstance(old, dict) else {}
        if _research_mode(state, agent) != "mixed":
            root.pop(agent, None)
            if hasattr(state, "__setitem__"):
                state[RESEARCH_GATE_STATE_KEY] = root
            return
        candidates = {
            item for item in old.get("candidate_urls", []) if isinstance(item, str)
        } if isinstance(old.get("candidate_urls", []), list) else set()
        pending = {
            item for item in old.get("pending_urls", []) if isinstance(item, str)
        } if isinstance(old.get("pending_urls", []), list) else set()
        phase = "inspect" if old.get("phase") == "inspect" else "discover"
        if discovered:
            candidates.update(discovered)
            pending.update(
                item for item in discovered if item not in self._inspected[agent]
            )
            if pending:
                phase = "inspect"
        if inspected_url and inspected_url in pending:
            pending.remove(inspected_url)
            phase = "discover"
        metrics = self._agents[agent]
        if phase == "inspect" and old.get("phase") != "inspect":
            metrics["discovery_gated"] += 1
        elif old.get("phase") == "inspect" and phase == "discover":
            metrics["discovery_reopened"] += 1
        root[agent] = {
            "phase": phase,
            "gated": phase == "inspect",
            "candidate_urls": sorted(candidates),
            "pending_urls": sorted(pending),
        }
        if hasattr(state, "__setitem__"):
            state[RESEARCH_GATE_STATE_KEY] = root

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        kind = _research_tool_kind(tool.name)
        if kind is None and tool_capability(tool.name, description=getattr(tool, "description", "") or "") == ToolCapability.DISCOVERY:
            kind = "search"
        agent = tool_context._invocation_context.agent.name
        if normalize_tool_name(strip_tool_name_prefix(tool.name)) == "get_next_action":
            action = _find_controller_action(result)
            if action:
                _record_controller_recommendation(tool_context.state, agent, action)
        if kind is None:
            return
        if kind in {"scraping", "browser_agent"}:
            url = _inspection_candidate_url(tool.name, tool_args, tool_context.state)
            if url:
                # An attempted inspection counts as inspection even if the source
                # is unavailable; discovery can resume after failed candidates.
                self.inspected(agent, url.strip())
                inspection_has_content = (
                    not _tool_failed(result) and _inspection_has_content(result)
                )
                candidate_progress = self._mark_candidate_inspected(
                    tool_context.state, agent, url,
                    status=(
                        "failed"
                        if _tool_failed(result)
                        else "success" if inspection_has_content else "empty"
                    ),
                    tool=tool.name,
                    error=_bounded_inspection_error(result),
                )
                self._mark_source_inspected(
                    tool_context.state, agent, url
                )
                if (
                    candidate_progress
                    and inspection_has_content
                ):
                    self._mark_candidate_inspection_progress(
                        tool_context.state, agent, url
                    )
                if _research_mode(tool_context.state, agent) == "mixed":
                    self._sync_gate_state(tool_context.state, agent, inspected_url=url)
        self._record_call_result(
            agent,
            tool.name,
            tool_args,
            result,
            call_id=_tool_call_id(tool_context),
            is_discovery=kind == "search",
        )
        if kind == "search":
            payload = _search_payload(result) if not _tool_failed(result) else None
            candidates = []
            if payload and isinstance(payload.get("results"), list):
                new_urls, had_ledger = self._add_candidates(
                    tool_context.state,
                    agent,
                    tool.name,
                    tool_args.get("query"),
                    payload["results"],
                )
                candidates = [
                    _sanitize_url(item["url"])
                    for item in payload["results"]
                    if isinstance(item, dict)
                    and isinstance(item.get("url"), str)
                    and _sanitize_url(item["url"])
                ]
                metrics = self._agents[agent]
                if new_urls:
                    metrics["discovery_calls_yielding_new_candidates"] += 1
                    evidence_urls = {
                        _sanitize_url(item.get("url"))
                        for item in payload["results"]
                        if isinstance(item, dict)
                        and isinstance(item.get("url"), str)
                        and str(item.get("title") or "").strip()
                        and str(item.get("snippet") or item.get("content") or "").strip()
                    }
                    if set(new_urls) & evidence_urls:
                        self._mark_progress(tool_context.state, agent, "candidate_evidence")
                    progress_root = tool_context.state.get(RESEARCH_PROGRESS_STATE_KEY, {})
                    if not isinstance(progress_root, dict):
                        progress_root = {}
                    progress = progress_root.setdefault(agent, {})
                    progress["productive_discovery_waves"] = min(3, int(progress.get("productive_discovery_waves", 0)) + 1)
                    tool_context.state[RESEARCH_PROGRESS_STATE_KEY] = progress_root
                else:
                    metrics["searches_with_no_new_candidates"] += 1
                    metrics["repeated_query_search_no_new_evidence_events"] += 1
                query = _query_key(tool_args.get("query"))
                if query:
                    if query in self._search_query_seen[agent]:
                        metrics["repeated_query_search_no_new_evidence_events"] += 1
                    self._search_query_seen[agent].add(query)
                if _research_mode(tool_context.state, agent) == "mixed":
                    self._sync_gate_state(
                        tool_context.state, agent, discovered=candidates
                    )
                metrics["candidate_urls_discovered"] = len(
                    tool_context.state.get(RESEARCH_CANDIDATE_LEDGER_KEY, {}).get(agent, [])
                ) if had_ledger else metrics["candidate_urls_discovered"]
        if (
            isinstance(result.get("error_code"), str)
            and result["error_code"] in CONTROL_CODES
        ):
            _record_control_block(
                tool_context.state,
                agent,
                tool.name,
                result["error_code"],
                _tool_call_id(tool_context),
            )
            return
        if kind == "browser_agent":
            payload = _browser_payload(result)
            usage = payload.get("usage", {}) if payload else {}
            if not isinstance(usage, dict):
                usage = {}
            metrics = self._agents[agent]
            fields = (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "llm_invocations",
            )
            for field in fields:
                value = usage.get(field)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    metrics[f"browser_agent_{field}"] += value
            steps = payload.get("steps_taken") if payload else None
            if isinstance(steps, int) and not isinstance(steps, bool) and steps >= 0:
                metrics["browser_agent_steps"] += steps
            if any(usage.get(field) is None for field in fields):
                metrics["browser_agent_usage_missing"] += 1
        elif kind == "code_agent":
            payload = _code_agent_payload(result)
            if payload is not None:
                metrics = self._agents[agent]
                usage = payload.get("usage", {})
                if isinstance(usage, dict):
                    for field in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "llm_invocations",
                    ):
                        value = usage.get(field)
                        if (
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 0
                        ):
                            metrics[f"code_agent_{field}"] += value
                    cost = usage.get("cost_usd")
                    if isinstance(cost, int | float) and not isinstance(cost, bool):
                        metrics["code_agent_cost_usd"] += cost
                    if usage.get("available") is not True:
                        metrics["code_agent_usage_missing"] += 1
                telemetry = payload.get("telemetry", {})
                if isinstance(telemetry, dict):
                    for source, target in (
                        ("execution_failures", "code_agent_execution_failures"),
                        ("timeouts", "code_agent_timeouts"),
                        ("files_accessed", "code_agent_files_accessed"),
                    ):
                        value = telemetry.get(source)
                        if isinstance(value, int) and not isinstance(value, bool):
                            metrics[target] += value
                    duration = telemetry.get("duration_seconds")
                    if isinstance(duration, int | float) and not isinstance(
                        duration, bool
                    ):
                        metrics["code_agent_duration_seconds"] += duration
                steps = payload.get("steps_taken")
                if isinstance(steps, int) and not isinstance(steps, bool):
                    metrics["code_agent_steps"] += steps
                status = payload.get("status")
                if status in {
                    "completed",
                    "incomplete",
                    "blocked",
                    "failed",
                }:
                    metrics[f"code_agent_{status}_calls"] += 1
                    if status == "failed":
                        metrics["failed_calls"] += 1
                        return
                    if status == "blocked":
                        metrics["blocked_calls"] += 1
        if (
            result.get("isError") is True
            or result.get("is_error") is True
            or result.get("error")
        ):
            metrics = self._agents[agent]
            metrics["failed_calls"] += 1
            if kind == "search":
                metrics["backend_errors"] += 1
            return
        metrics = self._agents[agent]
        metrics["successful_calls"] += 1
        if kind == "browser_agent":
            return
        if kind == "scraping":
            return

        payload = _search_payload(result)
        if payload is None:
            return
        results = payload.get("results")
        if isinstance(results, list):
            metrics["successful_searches"] += 1
            if not results:
                self._agents[agent]["zero_result_searches"] += 1
            for item in results:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    url = item["url"]
                    self._discovered[agent].add(_sanitize_url(url) or url)
            self._agents[agent]["urls_discovered"] = len(self._discovered[agent])
            self._agents[agent]["candidate_urls_discovered"] = len(
                self._discovered[agent]
            )

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> None:
        kind = _research_tool_kind(tool.name)
        if kind is not None:
            agent = tool_context._invocation_context.agent.name
            metrics = self._agents[agent]
            metrics["failed_calls"] += 1
            if kind == "code_agent":
                metrics["code_agent_failed_calls"] += 1
            if kind == "search":
                metrics["backend_errors"] += 1
            if kind in {"scraping", "browser_agent"}:
                url = _inspection_candidate_url(tool.name, tool_args, tool_context.state)
                if url:
                    self.inspected(agent, url)
                    self._mark_candidate_inspected(
                        tool_context.state, agent, url,
                        status="failed",
                        tool=tool.name,
                        error=str(error),
                    )
                    self._mark_source_inspected(
                        tool_context.state, agent, url
                    )
                    if _research_mode(tool_context.state, agent) == "mixed":
                        self._sync_gate_state(
                            tool_context.state, agent, inspected_url=url
                        )
            item = self._pending_call(
                agent,
                tool.name,
                tool_args,
                call_id=_tool_call_id(tool_context),
            )
            if item is not None:
                item["status"] = "executed_error"
                item["error_category"] = _error_category(str(error))

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Any
    ) -> None:
        agent = event.author or invocation_context.agent.name
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            response_part = getattr(part, "function_response", None)
            if response_part is not None and isinstance(response_part.response, dict):
                self._record_call_result(
                    agent,
                    response_part.name or "",
                    {},
                    response_part.response,
                    call_id=response_part.id,
                )
        if event.partial or not event.usage_metadata:
            return
        metrics = self._agents[agent]
        metrics["prompt_tokens"] += event.usage_metadata.prompt_token_count or 0
        metrics["completion_tokens"] += event.usage_metadata.candidates_token_count or 0
        state = getattr(invocation_context.session, "state", {})
        progress_root = state.get(RESEARCH_PROGRESS_STATE_KEY, {})
        progress = progress_root.get(agent) if isinstance(progress_root, dict) else None
        no_progress = (
            progress.get("no_progress_turns") if isinstance(progress, dict) else None
        )
        if isinstance(no_progress, int) and not isinstance(no_progress, bool):
            metrics["no_progress_turns"] = max(metrics["no_progress_turns"], no_progress)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        snapshot = {agent: dict(metrics) for agent, metrics in self._agents.items()}
        for agent, metrics in snapshot.items():
            metrics["tool_calls"] = [
                {
                    key: value
                    for key, value in call.items()
                    if key not in {"started", "_call_id"}
                }
                for call in self._calls[agent]
            ]
            metrics["_query_fingerprints"] = sorted(self._query_fingerprints[agent])
            metrics["_discovered_urls"] = sorted(self._discovered[agent])
            metrics["_inspected_urls"] = sorted(self._inspected[agent])
        return snapshot

    def _pending_call(
        self,
        agent: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        call_id: str | None = None,
    ) -> dict[str, Any] | None:
        name = strip_tool_name_prefix(tool_name)
        candidates = []
        for item in reversed(self._calls[agent]):
            if item.get("status") != "attempted" or item.get("tool") != name:
                continue
            if call_id is not None and item.get("_call_id") != call_id:
                continue
            if args.get("query") and item.get("query") != _sanitize_query(
                args.get("query")
            ):
                continue
            if args.get("url") and item.get("url") != _sanitize_url(args.get("url")):
                continue
            candidates.append(item)
        ambiguous = (
            call_id is None
            and not (args.get("query") or args.get("url"))
            and len(candidates) > 1
        )
        if not candidates or ambiguous:
            return None
        return candidates[0]

    def _record_call_result(
        self,
        agent: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        *,
        call_id: str | None = None,
        is_discovery: bool = False,
    ) -> None:
        item = self._pending_call(agent, tool_name, args, call_id=call_id)
        if item is None:
            return
        started = item.pop("started", time.monotonic())
        item["elapsed_ms"] = max(0, int((time.monotonic() - started) * 1000))
        item["result_chars"] = len(json.dumps(result, ensure_ascii=False, default=str))
        item["result_count"] = _result_count(result)
        code = result.get("error_code")
        error = str(result.get("error", result.get("message", "")))
        is_error = (
            result.get("isError") is True
            or result.get("is_error") is True
            or bool(error)
        )
        if item.get("tool", "").lower() == "complete_browser_task":
            browser_result = _browser_payload(result)
            if browser_result and browser_result.get("status") in {"failed", "blocked"}:
                is_error = True
                error = str(
                    browser_result.get("error", browser_result.get("message", ""))
                )
        if isinstance(code, str) and code in CONTROL_CODES:
            item["status"] = "blocked"
            item["error_category"] = _error_category(code)
        elif is_error:
            item["status"] = "executed_error"
            item["error_category"] = _error_category(error)
        elif (
            (is_discovery or tool_capability(item.get("tool", "")) == ToolCapability.DISCOVERY)
            and item["result_count"] == 0
        ):
            item["status"] = "executed_empty"
            item["error_category"] = "empty_results"
        else:
            item["status"] = "executed"


def _search_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(result.get("results"), list):
        return result
    for key in ("structuredContent", "structured_content", "result"):
        value = result.get(key)
        if isinstance(value, dict):
            found = _search_payload(value)
            if found is not None:
                return found
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                try:
                    value = json.loads(item["text"])
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict):
                    found = _search_payload(value)
                    if found is not None:
                        return found
    return None


def _research_tool_kind(name: str) -> str | None:
    capability = tool_capability(name)
    if capability == ToolCapability.BROWSER_NAVIGATION:
        return "browser_agent"
    if capability == ToolCapability.COMPUTATION:
        return "code_agent"
    if capability == ToolCapability.DISCOVERY:
        return "search"
    if is_inspection_tool(name):
        return "scraping"
    return None


def _sanitize_query(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", value.strip())
    text = re.sub(
        r"(?i)(api[_-]?key|token|password)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    return text[:240]


def _sanitize_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        if not parsed.scheme or not parsed.netloc:
            return value.strip()[:300]
        host = (parsed.hostname or "").lower()
        path = parsed.path[:220]
        query_source = parsed.query
        if host == "m.youtube.com":
            host = "youtube.com"
        if host == "youtu.be" and path.strip("/"):
            video_id = path.strip("/").split("/", 1)[0]
            host = "youtube.com"
            path = "/watch"
            query_source = urlencode({"v": video_id})
        if parsed.port:
            host += f":{parsed.port}"
        secret_params = {"api_key", "key", "token", "access_token", "password", "auth"}
        tracking_params = {"fbclid", "gclid", "ref", "source"}
        query = [
            (key[:80], item[:160])
            for key, item in parse_qsl(query_source, keep_blank_values=True)[:20]
            if key.casefold() not in secret_params
            and not key.casefold().startswith("utm_")
            and key.casefold() not in tracking_params
            and key.casefold() not in {"feature", "ab_channel", "si"}
        ]
        query.sort()
        return urlunsplit(
            (parsed.scheme.lower(), host, path, urlencode(query), "")
        )[:500]
    except ValueError:
        return "[invalid-url]"


def _research_mode(state: Any, agent: str) -> str:
    root = state.get(RESEARCH_MODE_STATE_KEY) if hasattr(state, "get") else None
    mode = root.get(agent) if isinstance(root, dict) else None
    return mode if mode in {"discovery_only", "mixed", "inspection_only"} else "mixed"


def _query_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()[:240]


def _record_control_block(
    state: Any,
    agent: str,
    tool: str,
    reason: str,
    call_id: str | None = None,
) -> None:
    metadata = state.get("_fedotmas_execution") if hasattr(state, "get") else None
    traces = metadata.get("turn_observability") if isinstance(metadata, dict) else None
    events = traces.get(agent) if isinstance(traces, dict) else None
    if not isinstance(events, list) or not events:
        return
    blocked = events[-1].setdefault("calls_blocked", [])
    if not isinstance(blocked, list):
        return
    if any(
        isinstance(item, dict)
        and (
            (call_id and item.get("call_id") == call_id)
            or (
                not call_id
                and item.get("tool") == tool[:120]
                and item.get("reason") == reason[:120]
            )
        )
        for item in blocked
    ):
        return
    if len(blocked) < 40:
        record = {"tool": tool[:120], "reason": reason[:120]}
        if isinstance(call_id, str):
            record["call_id"] = call_id[:120]
        blocked.append(record)


def _find_controller_action(value: Any, depth: int = 0) -> str | None:
    if depth > 4:
        return None
    if isinstance(value, dict):
        action = value.get("action")
        if action in {"continue_search", "change_strategy", "strategy_blocked", "synthesize"}:
            return action
        for nested in value.values():
            if found := _find_controller_action(nested, depth + 1):
                return found
    elif isinstance(value, list):
        for nested in value[:8]:
            if found := _find_controller_action(nested, depth + 1):
                return found
    elif isinstance(value, str) and value.strip().startswith("{"):
        try:
            return _find_controller_action(json.loads(value), depth + 1)
        except (ValueError, TypeError):
            return None
    return None


def _record_controller_recommendation(state: Any, agent: str, action: str) -> None:
    metadata = state.get("_fedotmas_execution") if hasattr(state, "get") else None
    traces = metadata.get("turn_observability") if isinstance(metadata, dict) else None
    events = traces.get(agent) if isinstance(traces, dict) else None
    if isinstance(events, list) and events:
        events[-1]["controller_recommendation"] = action
    if isinstance(metadata, dict):
        recommendations = metadata.setdefault("controller_recommendations", {})
        if isinstance(recommendations, dict):
            recommendations[agent] = action


def _inspection_candidate_url(name: str, args: dict[str, Any], state: Any) -> str | None:
    if not is_inspection_tool(name):
        return None
    for key in (
        "url",
        "uri",
        "href",
        "target",
        "video_url",
        "source_url",
        "start_url",
        "file_path",
        "path",
    ):
        value = args.get(key)
        if isinstance(value, str) and (normalized := _sanitize_url(value)):
            return normalized
    video_id = args.get("video_id") or args.get("id")
    if isinstance(video_id, str) and video_id.strip():
        return _sanitize_url(f"https://www.youtube.com/watch?v={video_id.strip()}")
    # Browser agents often accept a natural-language task rather than a URL
    # argument. Match an explicit candidate URL mentioned in that task.
    task = args.get("task") or args.get("instruction") or args.get("prompt")
    if isinstance(task, str):
        for candidate in re.findall(r"https?://[^\s\]\[<>\"']+", task):
            if normalized := _sanitize_url(candidate.rstrip(".,;)")):
                return normalized
    return None


def _error_category(error: str) -> str:
    lowered = error.lower()
    if any(term in lowered for term in ("dns", "name or service", "getaddrinfo")):
        return "dns_failure"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "circuit" in lowered or "tool_circuit_open" in lowered:
        return "circuit_breaker"
    if any(term in lowered for term in ("blocked", "budget", "disabled", "duplicate")):
        return "blocked_call"
    if any(term in lowered for term in ("searx", "backend", "http", "connection")):
        return "backend_error"
    return "tool_error"


def _tool_call_id(tool_context: ToolContext) -> str | None:
    value = getattr(tool_context, "function_call_id", None)
    return value if isinstance(value, str) else None


def _result_count(result: Any) -> int | None:
    if isinstance(result, dict):
        for key in ("results", "items", "content"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        for key in ("structuredContent", "structured_content", "result"):
            count = _result_count(result.get(key))
            if count is not None:
                return count
    return None


def _browser_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Read either MCP SDK spelling, choosing one copy of structured/text data."""
    if isinstance(result.get("usage"), dict) and "status" in result:
        return result
    for key in ("structuredContent", "structured_content", "result"):
        value = result.get(key)
        if isinstance(value, dict):
            found = _browser_payload(value)
            if found is not None:
                return found
    content = result.get("content")
    for item in content if isinstance(content, list) else []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            try:
                value = json.loads(item["text"])
            except ValueError:
                continue
            if isinstance(value, dict):
                found = _browser_payload(value)
                if found is not None:
                    return found
    return None


def _code_agent_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Find the compact code-agent result across MCP SDK result wrappers."""
    if isinstance(result.get("usage"), dict) and isinstance(result.get("status"), str):
        return result
    for key in ("structuredContent", "structured_content", "result"):
        value = result.get(key)
        if isinstance(value, dict):
            found = _code_agent_payload(value)
            if found is not None:
                return found
    content = result.get("content")
    for item in content if isinstance(content, list) else []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            try:
                value = json.loads(item["text"])
            except ValueError:
                continue
            if isinstance(value, dict):
                found = _code_agent_payload(value)
                if found is not None:
                    return found
    return None


def _bounded_inspection_error(result: Any) -> str:
    if not _tool_failed(result):
        return ""
    raw = result.get("error") or result.get("message") or result.get("error_code") or "inspection failed" if isinstance(result, dict) else "inspection failed"
    return " ".join(str(raw).split())[:240]
