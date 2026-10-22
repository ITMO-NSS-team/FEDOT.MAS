"""Lightpanda discovery must participate in research progress and budgets."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fedotmas.mcp.capabilities import ToolCapability, tool_capability
from fedotmas.plugins import WebSearchLimitPlugin
from fedotmas.plugins._research_telemetry import ResearchTelemetry

DESCRIPTION = "Run a web search and return results as markdown."
# Format returned by the installed Lightpanda MCP search, including links in snippets.
RESULT = {
    "content": [
        {
            "type": "text",
            "text": (
                "1. **First source** — https://example.org/first\n"
                "   Evidence with an [incidental link](https://example.org/incidental).\n\n"
                "2. **Second source** — https://example.org/second\n"
                "   Another piece of evidence.\n\n"
            ),
        }
    ],
    "isError": False,
}


def context():
    ctx = MagicMock()
    ctx._invocation_context.agent.name = "researcher"
    ctx._invocation_context.session.id = "session"
    ctx.state = {}
    return ctx


@pytest.mark.parametrize(
    "name", ["search", "web_scraping_search", "web-scraping__search"]
)
async def test_search_candidates_inspection_progress_and_budget(name):
    tool = SimpleNamespace(name=name, description=DESCRIPTION)
    assert tool_capability(name, description=DESCRIPTION) == ToolCapability.DISCOVERY
    telemetry = ResearchTelemetry()
    ctx = context()
    args = {"query": "a factual question"}
    assert (
        await telemetry.before_tool_callback(
            tool=tool, tool_args=args, tool_context=ctx
        )
        is None
    )
    await telemetry.after_tool_callback(
        tool=tool, tool_args=args, tool_context=ctx, result=RESULT
    )
    metrics = telemetry.snapshot()["researcher"]
    assert metrics["search_calls"] == 1
    assert metrics["candidate_urls_discovered"] == 2
    assert metrics["tool_calls"][0]["result_count"] == 2
    assert metrics["tool_calls"][0]["status"] == "executed"
    candidates = ctx.state["__fedotmas_research_candidates"]["researcher"]
    assert [item["url"] for item in candidates] == [
        "https://example.org/first",
        "https://example.org/second",
    ]
    progress = ctx.state["__fedotmas_research_progress"]["researcher"]
    assert "candidate_evidence" in progress["progress_events"]

    read = SimpleNamespace(name="web_scraping_markdown", description="Read a page")
    read_args = {"url": "https://example.org/first"}
    await telemetry.before_tool_callback(
        tool=read, tool_args=read_args, tool_context=ctx
    )
    await telemetry.after_tool_callback(
        tool=read,
        tool_args=read_args,
        tool_context=ctx,
        result={"content": [{"type": "text", "text": "Source body"}]},
    )
    assert candidates[0]["inspection_status"] == "success"
    assert (
        "candidate_inspection"
        in ctx.state["__fedotmas_research_progress"]["researcher"]["progress_events"]
    )

    limit = WebSearchLimitPlugin(max_calls_per_agent=1)
    assert (
        await limit.before_tool_callback(tool=tool, tool_args=args, tool_context=ctx)
        is None
    )
    blocked = await limit.before_tool_callback(
        tool=tool, tool_args={"query": "different"}, tool_context=ctx
    )
    assert blocked["error_code"] == "WEB_BUDGET_EXHAUSTED"


@pytest.mark.parametrize("name", ["search", "web_scraping_search"])
async def test_search_exception_is_recorded(name):
    telemetry = ResearchTelemetry()
    tool = SimpleNamespace(name=name, description=DESCRIPTION)
    ctx = context()
    args = {"query": "question"}
    await telemetry.before_tool_callback(tool=tool, tool_args=args, tool_context=ctx)
    await telemetry.on_tool_error_callback(
        tool=tool,
        tool_args=args,
        tool_context=ctx,
        error=RuntimeError("connection failed"),
    )
    metrics = telemetry.snapshot()["researcher"]
    assert metrics["failed_calls"] == 1
    assert metrics["backend_errors"] == 1
    assert metrics["tool_calls"][0]["status"] == "executed_error"


async def test_non_web_search_does_not_consume_budget_or_create_candidates():
    tool = SimpleNamespace(name="memory_search", description="Search local memory")
    assert (
        tool_capability(tool.name, description=tool.description) == ToolCapability.OTHER
    )
    telemetry = ResearchTelemetry()
    limit = WebSearchLimitPlugin(max_calls_per_agent=1)
    ctx = context()
    for query in ("first", "second"):
        args = {"query": query}
        assert (
            await limit.before_tool_callback(
                tool=tool, tool_args=args, tool_context=ctx
            )
            is None
        )
        await telemetry.before_tool_callback(
            tool=tool, tool_args=args, tool_context=ctx
        )
        await telemetry.after_tool_callback(
            tool=tool, tool_args=args, tool_context=ctx, result=RESULT
        )
    assert "__fedotmas_research_candidates" not in ctx.state


async def test_error_response_does_not_become_evidence():
    tool = SimpleNamespace(name="web_scraping_search", description=DESCRIPTION)
    telemetry = ResearchTelemetry()
    ctx = context()
    args = {"query": "question"}
    await telemetry.before_tool_callback(tool=tool, tool_args=args, tool_context=ctx)
    await telemetry.after_tool_callback(
        tool=tool, tool_args=args, tool_context=ctx, result={**RESULT, "isError": True}
    )
    assert "__fedotmas_research_candidates" not in ctx.state
    assert "__fedotmas_research_progress" not in ctx.state
