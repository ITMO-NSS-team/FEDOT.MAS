from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fedotmas.plugins._research_telemetry import ResearchTelemetry
from google.genai import types


def _context(agent: str = "researcher") -> MagicMock:
    context = MagicMock()
    context._invocation_context.agent.name = agent
    return context


@pytest.mark.asyncio
async def test_browser_usage_is_recorded_even_for_mcp_errors():
    telemetry = ResearchTelemetry()
    tool = MagicMock()
    tool.name = "complete_browser_task"
    result = {
        "is_error": True,
        "structured_content": {
            "status": "failed",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "llm_invocations": 2,
            },
            "steps_taken": 4,
        },
    }
    await telemetry.before_tool_callback(
        tool=tool, tool_args={"task": "first"}, tool_context=_context()
    )
    await telemetry.after_tool_callback(
        tool=tool, tool_args={}, tool_context=_context(), result=result
    )

    metrics = telemetry.snapshot()["researcher"]
    assert metrics["browser_agent_calls"] == 1
    assert metrics["failed_calls"] == 1
    assert metrics["browser_agent_prompt_tokens"] == 120
    assert metrics["browser_agent_completion_tokens"] == 30
    assert metrics["browser_agent_total_tokens"] == 150
    assert metrics["browser_agent_llm_invocations"] == 2
    assert metrics["browser_agent_steps"] == 4
    assert metrics["browser_agent_usage_missing"] == 0


@pytest.mark.asyncio
async def test_code_agent_usage_steps_and_execution_metrics_are_recorded():
    telemetry = ResearchTelemetry()
    tool = MagicMock()
    tool.name = "solve_with_code"
    await telemetry.before_tool_callback(
        tool=tool,
        tool_args={"task": "filter rows", "files": ["data.csv"]},
        tool_context=_context(),
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args={},
        tool_context=_context(),
        result={
            "status": "completed",
            "steps_taken": 2,
            "usage": {
                "available": True,
                "llm_invocations": 3,
                "prompt_tokens": 90,
                "completion_tokens": 20,
                "total_tokens": 110,
                "cost_usd": 0.004,
            },
            "telemetry": {
                "duration_seconds": 4.5,
                "execution_failures": 1,
                "timeouts": 0,
                "files_accessed": 1,
            },
        },
    )

    metrics = telemetry.snapshot()["researcher"]
    assert metrics["code_agent_calls"] == 1
    assert metrics["code_agent_completed_calls"] == 1
    assert metrics["code_agent_steps"] == 2
    assert metrics["code_agent_execution_failures"] == 1
    assert metrics["code_agent_files_accessed"] == 1
    assert metrics["code_agent_llm_invocations"] == 3
    assert metrics["code_agent_total_tokens"] == 110
    assert metrics["code_agent_cost_usd"] == 0.004
    assert metrics["code_agent_duration_seconds"] == 4.5


@pytest.mark.asyncio
async def test_discovery_pauses_until_candidate_inspection_and_resumes_after_failure():
    telemetry = ResearchTelemetry()
    search = MagicMock(name="search")
    search.name = "search"
    args = {"query": "broad topic"}
    allowed = await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=_context()
    )
    assert allowed is None
    await telemetry.after_tool_callback(
        tool=search,
        tool_args=args,
        tool_context=_context(),
        result={
            "results": [
                {"url": f"https://example.org/{number}"} for number in range(4)
            ]
        },
    )

    blocked = await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=_context()
    )
    assert blocked["error_code"] == "INSPECT_CANDIDATES_FIRST"
    targeted = await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": 'site:example.org "specific claim"'},
        tool_context=_context(),
    )
    assert targeted is None

    inspect = MagicMock(name="markdown")
    inspect.name = "markdown"
    await telemetry.before_tool_callback(
        tool=inspect,
        tool_args={"url": "https://example.org/0"},
        tool_context=_context(),
    )
    await telemetry.after_tool_callback(
        tool=inspect,
        tool_args={"url": "https://example.org/0"},
        tool_context=_context(),
        result={"isError": True, "error": "source unavailable"},
    )
    resumed = await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=_context()
    )
    assert resumed is None

    metrics = telemetry.snapshot()["researcher"]
    assert metrics["discovery_calls"] == 4
    assert metrics["candidate_urls_discovered"] == 4
    assert metrics["candidate_urls_inspected"] == 1
    assert metrics["repeated_discovery_without_inspection"] == 1


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
            assert telemetry.snapshot()["researcher"]["urls_inspected"] == 1

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


@pytest.mark.asyncio
async def test_per_call_diagnostics_distinguish_infra_failures_and_blocks():
    telemetry = ResearchTelemetry()

    async def record(name, args, result):
        tool = MagicMock()
        tool.name = name
        await telemetry.before_tool_callback(
            tool=tool, tool_args=args, tool_context=_context()
        )
        await telemetry.after_tool_callback(
            tool=tool, tool_args=args, tool_context=_context(), result=result
        )

    await record(
        "search",
        {
            "query": "lookup token=secret@example.com",
            "url": "https://user:pass@example.org/search?token=secret",
        },
        {"isError": True, "error": "HTTP backend unavailable in SearXNG"},
    )
    await record(
        "search",
        {"query": "dns check"},
        {"isError": True, "error": "DNS lookup failed: getaddrinfo"},
    )
    await record(
        "complete_browser_task",
        {"task": "open the source"},
        {"isError": True, "error": "OperationTimedout: browser timeout"},
    )
    await record("search", {"query": "empty"}, {"results": []})
    search_tool = MagicMock()
    search_tool.name = "search"
    await telemetry.before_tool_callback(
        tool=search_tool,
        tool_args={"query": "blocked"},
        tool_context=_context(),
    )
    telemetry.record_blocked(
        "researcher",
        "search",
        {"query": "blocked"},
        category="budget_exhausted",
        budget={"kind": "search", "limit": 1, "used": 1, "remaining": 0},
    )
    markdown_tool = MagicMock()
    markdown_tool.name = "markdown"
    await telemetry.before_tool_callback(
        tool=markdown_tool,
        tool_args={"url": "https://example.org"},
        tool_context=_context(),
    )
    telemetry.record_blocked(
        "researcher",
        "markdown",
        {"url": "https://example.org"},
        category="circuit_breaker",
    )

    calls = telemetry.snapshot()["researcher"]["tool_calls"]
    assert [call["error_category"] for call in calls] == [
        "backend_error",
        "dns_failure",
        "timeout",
        "empty_results",
        "budget_exhausted",
        "circuit_breaker",
    ]
    assert all(call["elapsed_ms"] is not None for call in calls)
    assert calls[0]["result_chars"] > 0
    assert calls[0]["query"] == "lookup token=[redacted]"
    assert calls[0]["url"] == "https://example.org/search"
    assert calls[4]["budget"]["remaining"] == 0
    assert calls[3]["result_count"] == 0


@pytest.mark.asyncio
async def test_event_diagnostics_match_parallel_tool_results_by_call_id():
    telemetry = ResearchTelemetry()
    tool = MagicMock()
    tool.name = "search"
    first_context = _context()
    first_context.function_call_id = "call-1"
    second_context = _context()
    second_context.function_call_id = "call-2"

    for query, context in (("first", first_context), ("second", second_context)):
        await telemetry.before_tool_callback(
            tool=tool, tool_args={"query": query}, tool_context=context
        )

    event = SimpleNamespace(
        author="researcher",
        partial=False,
        usage_metadata=None,
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call-1",
                        name="search",
                        response={
                            "isError": True,
                            "error_code": "WEB_BUDGET_EXHAUSTED",
                            "error": "search budget exhausted",
                        },
                    )
                )
            ],
        ),
    )
    invocation = SimpleNamespace(agent=SimpleNamespace(name="researcher"))
    await telemetry.on_event_callback(invocation_context=invocation, event=event)

    calls = telemetry.snapshot()["researcher"]["tool_calls"]
    assert [(call["query"], call["status"]) for call in calls] == [
        ("first", "blocked"),
        ("second", "attempted"),
    ]
