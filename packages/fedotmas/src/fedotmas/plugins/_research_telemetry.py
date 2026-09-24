"""Structured, per-agent research metrics for benchmark artifacts."""

from __future__ import annotations

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
    WEB_BUDGET_EXHAUSTED,
)


class ResearchTelemetry(BasePlugin):
    """Collect tool outcomes and token usage without inspecting log text."""

    def __init__(self, name: str = "fedotmas_research_telemetry") -> None:
        super().__init__(name=name)
        self._agents: dict[str, dict[str, Any]] = defaultdict(self._new_agent)
        self._queries: dict[str, set[str]] = defaultdict(set)
        self._discovered: dict[str, set[str]] = defaultdict(set)
        self._inspected: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def _new_agent() -> dict[str, Any]:
        return {
            "search_calls": 0,
            "unique_queries": 0,
            "duplicate_blocks": 0,
            "zero_result_searches": 0,
            "backend_errors": 0,
            "urls_discovered": 0,
            "urls_inspected": 0,
            "scraping_extraction_calls": 0,
            "search_exhaustion": 0,
            "scraping_exhaustion": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def attempt(self, agent: str, kind: str, args: dict[str, Any]) -> None:
        metrics = self._agents[agent]
        if kind == "search":
            metrics["search_calls"] += 1
            query = args.get("query")
            if isinstance(query, str) and query.strip():
                self._queries[agent].add(query.strip().casefold())
                metrics["unique_queries"] = len(self._queries[agent])
        else:
            metrics["scraping_extraction_calls"] += 1

    def duplicate(self, agent: str) -> None:
        self._agents[agent]["duplicate_blocks"] += 1

    def exhausted(self, agent: str, kind: str) -> None:
        self._agents[agent][f"{kind}_exhaustion"] += 1

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
        del tool_args
        if strip_tool_name_prefix(tool.name).lower() not in {
            "search",
            "web_search",
            "web-search",
            "websearch",
            "google_search",
            "searxng_search",
        }:
            return
        agent = tool_context._invocation_context.agent.name
        if result.get("error_code") in {DUPLICATE_TOOL_CALL, WEB_BUDGET_EXHAUSTED}:
            return
        if result.get("isError") is True or result.get("error"):
            self._agents[agent]["backend_errors"] += 1
            return
        payload = _search_payload(result)
        if payload is None:
            return
        results = payload.get("results")
        if isinstance(results, list):
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
        if strip_tool_name_prefix(tool.name).lower() in {
            "search",
            "web_search",
            "web-search",
            "websearch",
            "google_search",
            "searxng_search",
        }:
            self._agents[tool_context._invocation_context.agent.name][
                "backend_errors"
            ] += 1

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
        return {agent: dict(metrics) for agent, metrics in self._agents.items()}


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
