from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fedotmas.plugins._research_telemetry import ResearchTelemetry


def _context(agent: str = "researcher") -> MagicMock:
    context = MagicMock()
    context._invocation_context.agent.name = agent
    return context


@pytest.mark.asyncio
async def test_research_telemetry_records_structured_outcomes_and_serializes():
    telemetry = ResearchTelemetry()
    tool = MagicMock(name="search")
    tool.name = "search"
    telemetry.attempt("researcher", "search", {"query": "Example"})
    telemetry.attempt("researcher", "search", {"query": "example"})
    telemetry.duplicate("researcher")
    telemetry.exhausted("researcher", "search")
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={"results": [{"url": "https://example.com", "title": "Example"}]},
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={"results": []},
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={"isError": True, "error": "backend down"},
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={
            "isError": True,
            "error_code": "WEB_BUDGET_EXHAUSTED",
            "error": "budget",
        },
    )
    telemetry.attempt("researcher", "scraping", {"url": "https://example.com"})
    telemetry.inspected("researcher", "https://example.com")
    metrics = json.loads(json.dumps(telemetry.snapshot()))["researcher"]
    assert metrics["search_calls"] == 2
    assert metrics["unique_queries"] == 1
    assert metrics["duplicate_blocks"] == 1
    assert metrics["zero_result_searches"] == 1
    assert metrics["backend_errors"] == 1
    assert metrics["urls_discovered"] == 1
    assert metrics["urls_inspected"] == 1
    assert metrics["scraping_extraction_calls"] == 1
    assert metrics["search_exhaustion"] == 1
