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
    for query in ("Example", "example"):
        await telemetry.before_tool_callback(
            tool=tool,
            tool_args={"query": query},
            tool_context=_context(),
        )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={"results": [{"url": "https://example.com", "title": "Example"}]},
    )
    await telemetry.before_tool_callback(
        tool=tool,
        tool_args={"query": "zero"},
        tool_context=_context(),
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={"results": []},
    )
    await telemetry.before_tool_callback(
        tool=tool,
        tool_args={"query": "failure"},
        tool_context=_context(),
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={"isError": True, "error": "backend down"},
    )
    telemetry.duplicate("researcher")
    telemetry.exhausted("researcher", "search")
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
    scraping_tool = MagicMock(name="markdown")
    scraping_tool.name = "markdown"
    for result in (
        {"isError": True, "error": "navigation failed"},
        {"content": "page text"},
    ):
        await telemetry.before_tool_callback(
            tool=scraping_tool,
            tool_args={"url": "https://example.com"},
            tool_context=_context(),
        )
        await telemetry.after_tool_callback(
            tool=scraping_tool,
            tool_args={"url": "https://example.com"},
            tool_context=_context(),
            result=result,
        )
        if result.get("isError"):
            assert telemetry.snapshot()["researcher"]["urls_inspected"] == 0

    status_tool = MagicMock(name="status")
    status_tool.name = "status"
    await telemetry.before_tool_callback(
        tool=status_tool,
        tool_args={"url": "https://example.org"},
        tool_context=_context(),
    )
    await telemetry.after_tool_callback(
        tool=status_tool,
        tool_args={"url": "https://example.org"},
        tool_context=_context(),
        result={"status": "ready"},
    )

    telemetry.circuit_blocked("researcher", "search")
    metrics = json.loads(json.dumps(telemetry.snapshot()))["researcher"]
    assert metrics["attempted_calls"] == 7
    assert metrics["blocked_calls"] == 3
    assert metrics["successful_calls"] == 4
    assert metrics["failed_calls"] == 2
    assert metrics["search_calls"] == 4
    assert metrics["successful_searches"] == 2
    assert metrics["unique_queries"] == 3
    assert metrics["duplicate_blocks"] == 1
    assert metrics["zero_result_searches"] == 1
    assert metrics["backend_errors"] == 1
    assert metrics["urls_discovered"] == 1
    assert metrics["urls_inspected"] == 1
    assert metrics["scraping_extraction_calls"] == 3
    assert metrics["search_exhaustion"] == 1
