"""Structured, per-agent research metrics for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

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
        del tool_args, error
        kind = _research_tool_kind(tool.name)
        if kind is not None:
            metrics = self._agents[tool_context._invocation_context.agent.name]
            metrics["failed_calls"] += 1
            if kind == "search":
                metrics["backend_errors"] += 1

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Any
    ) -> None:
        if event.partial or not event.usage_metadata:
            return
        agent = event.author or invocation_context.agent.name
        metrics = self._agents[agent]
        metrics["prompt_tokens"] += event.usage_metadata.prompt_token_count or 0
        metrics["completion_tokens"] += event.usage_metadata.candidates_token_count or 0

    def snapshot(self) -> dict[str, dict[str, Any]]:
        snapshot = {agent: dict(metrics) for agent, metrics in self._agents.items()}
        for agent, metrics in snapshot.items():
            metrics["_query_fingerprints"] = sorted(self._query_fingerprints[agent])
            metrics["_discovered_urls"] = sorted(self._discovered[agent])
            metrics["_inspected_urls"] = sorted(self._inspected[agent])
        return snapshot


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
    if normalized in SEARCH_TOOLS:
        return "search"
    short_name = normalized.rsplit("_", 1)[-1]
    if normalized in SCRAPING_TOOLS or short_name in SCRAPING_TOOLS:
        return "scraping"
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
