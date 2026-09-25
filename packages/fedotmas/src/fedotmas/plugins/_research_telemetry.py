"""Structured, per-agent research metrics for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.mcp import strip_tool_name_prefix
from fedotmas.plugins._tool_error_circuit_breaker import (
    DUPLICATE_TOOL_CALL,
    TOOL_CIRCUIT_OPEN,
    WEB_BUDGET_EXHAUSTED,
)

SEARCH_TOOLS = frozenset(
    {
        "search",
        "web_search",
        "web-search",
        "websearch",
        "google_search",
        "searxng_search",
    }
)
SCRAPING_TOOLS = frozenset(
    {"goto", "markdown", "extract", "links", "eval", "evaluate", "screenshot", "status"}
)
URL_INSPECTION_TOOLS = SCRAPING_TOOLS - {"status"}
CONTROL_CODES = frozenset(
    {DUPLICATE_TOOL_CALL, WEB_BUDGET_EXHAUSTED, TOOL_CIRCUIT_OPEN}
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
    ) -> None:
        kind = _research_tool_kind(tool.name)
        if kind is None:
            return
        agent = tool_context._invocation_context.agent.name
        self.attempt(agent, kind, tool_args)
        calls = self._calls[agent]
        calls.append(
            {
                "agent": agent,
                "tool": strip_tool_name_prefix(tool.name),
                "query": _sanitize_query(tool_args.get("query")),
                "url": _sanitize_url(tool_args.get("url")),
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
        self._inspected[agent].add(url)
        self._agents[agent]["urls_inspected"] = len(self._inspected[agent])

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        kind = _research_tool_kind(tool.name)
        if kind is None:
            return
        agent = tool_context._invocation_context.agent.name
        self._record_call_result(
            agent,
            tool.name,
            tool_args,
            result,
            call_id=_tool_call_id(tool_context),
        )
        if (
            isinstance(result.get("error_code"), str)
            and result["error_code"] in CONTROL_CODES
        ):
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
            url = tool_args.get("url")
            if (
                strip_tool_name_prefix(tool.name).lower() in URL_INSPECTION_TOOLS
                and isinstance(url, str)
                and url.strip()
            ):
                self.inspected(agent, url.strip())
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
                    self._discovered[agent].add(item["url"])
            self._agents[agent]["urls_discovered"] = len(self._discovered[agent])

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
        elif item.get("tool", "").lower() in SEARCH_TOOLS and item["result_count"] == 0:
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
    normalized = strip_tool_name_prefix(name).lower()
    if normalized == "complete_browser_task":
        return "browser_agent"
    if normalized == "solve_with_code":
        return "code_agent"
    if normalized in SEARCH_TOOLS:
        return "search"
    short_name = normalized.rsplit("_", 1)[-1]
    if normalized in SCRAPING_TOOLS or short_name in SCRAPING_TOOLS:
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
        host = parsed.hostname or ""
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path[:180], "", ""))[:300]
    except ValueError:
        return "[invalid-url]"


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
