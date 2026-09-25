from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fedotmas.plugins._research_telemetry import ResearchTelemetry
from fedotmas.plugins._tool_result_truncation import ToolResultTruncationPlugin
from google.adk.models.llm_request import LlmRequest
from google.genai import types


def _context(agent: str = "researcher") -> MagicMock:
    context = MagicMock()
    context._invocation_context.agent.name = agent
    context.state = {}
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
async def test_discovery_gate_is_stateful_and_reopens_after_failed_candidate_inspection():
    telemetry = ResearchTelemetry()
    context = _context()
    search = MagicMock(name="search")
    search.name = "search"
    args = {"query": "broad topic"}
    allowed = await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=context
    )
    assert allowed is None
    await telemetry.after_tool_callback(
        tool=search,
        tool_args=args,
        tool_context=context,
        result={
            "results": [
                {"url": f"https://example.org/{number}"} for number in range(4)
            ]
        },
    )

    quoted_broad = await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": '"general topic"'},
        tool_context=context,
    )
    assert quoted_broad["error_code"] == "INSPECT_CANDIDATES_FIRST"
    unrelated_site = await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": "site:unrelated.example example.org specific claim"},
        tool_context=context,
    )
    assert unrelated_site["error_code"] == "INSPECT_CANDIDATES_FIRST"
    targeted = await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": 'site:example.org "specific claim"'},
        tool_context=context,
    )
    assert targeted["error_code"] == "INSPECT_CANDIDATES_FIRST"
    candidate_domain = await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": "example.org focused recovery"},
        tool_context=context,
    )
    assert candidate_domain["error_code"] == "INSPECT_CANDIDATES_FIRST"

    inspect = MagicMock(name="markdown")
    inspect.name = "markdown"
    await telemetry.before_tool_callback(
        tool=inspect,
        tool_args={"url": "https://example.org/0"},
        tool_context=context,
    )
    await telemetry.after_tool_callback(
        tool=inspect,
        tool_args={"url": "https://example.org/0"},
        tool_context=context,
        result={"isError": True, "error": "source unavailable"},
    )
    resumed = await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=context
    )
    assert resumed is None

    metrics = telemetry.snapshot()["researcher"]
    assert metrics["discovery_calls"] == 6
    assert metrics["candidate_urls_discovered"] == 4
    assert metrics["candidate_urls_inspected"] == 1
    assert metrics["repeated_discovery_without_inspection"] == 4
    assert metrics["discovery_gated"] == 1
    assert metrics["discovery_reopened"] == 1


@pytest.mark.asyncio
async def test_unrelated_scrape_does_not_count_as_candidate_inspection():
    telemetry = ResearchTelemetry()
    context = _context()
    search = MagicMock(name="search")
    search.name = "search"
    await telemetry.before_tool_callback(
        tool=search, tool_args={"query": "topic"}, tool_context=context
    )
    await telemetry.after_tool_callback(
        tool=search,
        tool_args={},
        tool_context=context,
        result={"results": [{"url": "https://example.org/paper"}]},
    )
    scrape = MagicMock(name="markdown")
    scrape.name = "markdown"
    args = {"url": "https://unrelated.example/page"}
    await telemetry.before_tool_callback(
        tool=scrape, tool_args=args, tool_context=context
    )
    await telemetry.after_tool_callback(
        tool=scrape,
        tool_args=args,
        tool_context=context,
        result={"content": "unrelated"},
    )

    assert telemetry.snapshot()["researcher"]["candidate_urls_inspected"] == 0
    assert context.state["__fedotmas_research_gate"]["researcher"]["phase"] == "inspect"


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
    assert metrics["attempted_calls"] == 6
    assert metrics["blocked_calls"] == 3
    assert metrics["successful_calls"] == 3
    assert metrics["failed_calls"] == 2
    assert metrics["search_calls"] == 4
    assert metrics["successful_searches"] == 2
    assert metrics["unique_queries"] == 3
    assert metrics["duplicate_blocks"] == 1
    assert metrics["zero_result_searches"] == 1
    assert metrics["backend_errors"] == 1
    assert metrics["urls_discovered"] == 1
    assert metrics["urls_inspected"] == 1
    assert metrics["scraping_extraction_calls"] == 2
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


@pytest.mark.asyncio
async def test_youtube_candidate_inspection_reopens_mixed_discovery():
    telemetry = ResearchTelemetry()
    context = _context()
    search = MagicMock(name="searxng_search")
    search.name = "searxng_search"
    await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": "named video"},
        tool_context=context,
    )
    await telemetry.after_tool_callback(
        tool=search,
        tool_args={"query": "named video"},
        tool_context=context,
        result={
            "results": [
                {
                    "url": "https://www.youtube.com/watch?v=video123&feature=shared",
                    "title": "Target interview",
                    "snippet": "The speaker explains the requested event.",
                }
            ]
        },
    )

    info = MagicMock(name="get_video_info")
    info.name = "get_video_info"
    assert await telemetry.before_tool_callback(
        tool=info,
        tool_args={"video_id": "video123"},
        tool_context=context,
    ) is None
    await telemetry.after_tool_callback(
        tool=info,
        tool_args={"video_id": "video123"},
        tool_context=context,
        result={"title": "Target interview", "duration": 120},
    )

    candidate = context.state["__fedotmas_research_candidates"]["researcher"][0]
    assert candidate["inspected"] is True
    assert context.state["__fedotmas_research_gate"]["researcher"]["phase"] == "discover"
    assert telemetry.snapshot()["researcher"]["candidate_urls_inspected"] == 1


@pytest.mark.asyncio
async def test_inspection_of_supplied_document_source_counts_as_progress():
    telemetry = ResearchTelemetry()
    context = _context()
    context.state["__fedotmas_research_modes"] = {
        "researcher": "inspection_only"
    }
    tool = MagicMock(name="read_document")
    tool.name = "read_document"
    args = {"file_path": "/tmp/records.xml"}

    await telemetry.before_tool_callback(
        tool=tool, tool_args=args, tool_context=context
    )
    await telemetry.after_tool_callback(
        tool=tool,
        tool_args=args,
        tool_context=context,
        result={"content": "<record>evidence</record>"},
    )

    progress = context.state["__fedotmas_research_progress"]["researcher"]
    assert progress["version"] == 1
    assert progress["progress_events"] == ["source_inspected"]


@pytest.mark.asyncio
async def test_candidate_ledger_deduplicates_and_records_no_progress_searches():
    telemetry = ResearchTelemetry()
    context = _context()
    search = MagicMock(name="search")
    search.name = "search"
    result = {
        "results": [
            {
                "url": "https://example.org/source?utm_source=search",
                "title": "The source",
                "snippet": "A useful source excerpt.",
            }
        ]
    }

    for query in ("first search", "same-source again"):
        await telemetry.before_tool_callback(
            tool=search,
            tool_args={"query": query},
            tool_context=context,
        )
        await telemetry.after_tool_callback(
            tool=search,
            tool_args={"query": query},
            tool_context=context,
            result=result,
        )

    ledger = context.state["__fedotmas_research_candidates"]["researcher"]
    progress = context.state["__fedotmas_research_progress"]["researcher"]
    metrics = telemetry.snapshot()["researcher"]
    assert len(ledger) == 1
    assert ledger[0]["title"] == "The source"
    assert progress["version"] == 1
    assert metrics["discovery_calls_yielding_new_candidates"] == 1
    assert metrics["searches_with_no_new_candidates"] == 1
    assert metrics["repeated_query_search_no_new_evidence_events"] >= 1


@pytest.mark.asyncio
async def test_candidate_ledger_survives_rolling_tool_result_compaction():
    telemetry = ResearchTelemetry()
    context = _context()
    context.state["__fedotmas_research_modes"] = {"researcher": "discovery_only"}
    search = MagicMock(name="search")
    search.name = "search"
    await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": "large result"},
        tool_context=context,
    )
    await telemetry.after_tool_callback(
        tool=search,
        tool_args={"query": "large result"},
        tool_context=context,
        result={
            "results": [
                {
                    "url": "https://example.org/source",
                    "title": "Preserved title",
                    "snippet": "Preserved evidence snippet",
                }
            ]
        },
    )
    before = context.state["__fedotmas_research_candidates"]["researcher"]
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name="search",
                    response={"content": "old evidence " + "x" * 900},
                )
            ],
        )
        for _ in range(4)
    ]
    await ToolResultTruncationPlugin(max_agent_total_chars=700).before_model_callback(
        callback_context=context,
        llm_request=LlmRequest(contents=contents),
    )

    assert context.state["__fedotmas_research_candidates"]["researcher"] == before
    assert before[0]["title"] == "Preserved title"
    assert before[0]["snippet"] == "Preserved evidence snippet"
