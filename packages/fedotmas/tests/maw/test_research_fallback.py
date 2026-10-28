import json
from unittest.mock import MagicMock

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool
from google.genai import types

from fedotmas.maw import builder
from fedotmas.maw._research_fallback import allow_fallback, record_retrieval
from fedotmas.maw.handoffs import is_unresolved_answer
from fedotmas.maw.models import MAWAgentConfig
from fedotmas.plugins._research_telemetry import (
    MAX_WEB_FRAGMENT_PROGRESS,
    RESEARCH_CANDIDATE_LEDGER_KEY,
    RESEARCH_PROGRESS_STATE_KEY,
    RESEARCH_TURN_STATE_KEY,
    ResearchTelemetry,
)


def context(state):
    ctx = MagicMock()
    ctx.state = state
    ctx._invocation_context.agent.name = "reader"
    return ctx


def tool(name):
    t = FunctionTool(lambda url: "")
    t.name = name
    if name == "search":
        t.description = "Search the web with Tavily"
    return t


def request(*tools):
    return LlmRequest(
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name=t.name) for t in tools
                    ]
                )
            ]
        ),
        tools_dict={t.name: t for t in tools},
    )


@pytest.mark.asyncio
async def test_real_reading_counts_without_search_ledger_and_deduplicates():
    state = {}
    ctx = context(state)
    telemetry = ResearchTelemetry()
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="reader", instruction="Read", output_key="out"), None, None
    )
    await agent.before_agent_callback(ctx)
    for name, response in [
        ("web_scraping_goto", {"content": [{"text": "Navigated successfully."}]}),
        (
            "web_scraping_markdown",
            {"content": [{"text": "A page with candidate evidence."}]},
        ),
        ("web_scraping_evaluate", {"content": [{"text": '["Raymond (odc. 44),"]'}]}),
    ]:
        t = tool(name)
        await agent.before_model_callback(ctx, request(t))
        args = (
            {"url": "https://example.org/page"}
            if name != "web_scraping_evaluate"
            else {"script": "read section"}
        )
        await telemetry.after_tool_callback(
            tool=t, tool_args=args, tool_context=ctx, result=response
        )
        await agent.after_tool_callback(t, args, ctx, response)
    assert state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["progress_events"] == [
        "web_fragment",
        "web_fragment",
    ]
    await agent.before_model_callback(ctx, request(tool("markdown")))
    assert not state[RESEARCH_TURN_STATE_KEY]["reader"]["force_converge"]
    # Repeated content on a new URL or via another reader earns no progress.
    version = state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["version"]
    await telemetry.after_tool_callback(
        tool=tool("html"),
        tool_args={"url": "https://example.org/mirror"},
        tool_context=ctx,
        result={"content": [{"text": "A page with candidate evidence."}]},
    )
    assert state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["version"] == version


@pytest.mark.asyncio
async def test_web_progress_cap_and_empty_error_wrappers():
    state = {}
    ctx = context(state)
    p = ResearchTelemetry()
    t = tool("extract")
    for response in [
        {"content": [{"text": ""}]},
        {"content": [{"text": '{"field":null}'}]},
        {"content": [{"text": '{"error":"denied","content":"error page"}'}]},
        {"isError": True, "content": [{"text": "denied"}]},
    ]:
        await p.after_tool_callback(
            tool=t, tool_args={}, tool_context=ctx, result=response
        )
    assert not state.get(RESEARCH_PROGRESS_STATE_KEY)
    for i in range(MAX_WEB_FRAGMENT_PROGRESS + 2):
        await p.after_tool_callback(
            tool=t,
            tool_args={},
            tool_context=ctx,
            result={"content": [{"text": json.dumps({"value": i})}]},
        )
    assert (
        state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["version"]
        == MAX_WEB_FRAGMENT_PROGRESS
    )


@pytest.mark.asyncio
async def test_stalled_reader_can_change_known_source_or_backend_but_not_loop():
    urls = [
        "https://example.org/failed",
        "https://example.org/alternative",
        "https://example.org/third",
    ]
    state = {RESEARCH_CANDIDATE_LEDGER_KEY: {"reader": [{"url": u} for u in urls]}}
    ctx = context(state)
    agent = builder._build_llm_agent(
        MAWAgentConfig(
            name="reader",
            instruction="Read",
            output_key="out",
            research_mode="inspection_only",
        ),
        None,
        None,
    )
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("markdown"), {"url": urls[0]}, ctx, {"isError": True}
    )
    state[RESEARCH_TURN_STATE_KEY] = {"reader": {"force_converge": True}}
    # Same URL/reader is blocked, even if a different selector is invented.
    assert (
        await agent.before_tool_callback(
            tool("markdown"), {"url": urls[0], "selector": "new"}, ctx
        )
    )["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"
    # A direct document read is a different backend for the failed URL.
    assert (
        await agent.before_tool_callback(
            tool("read_document"), {"file_path": urls[0]}, ctx
        )
        is None
    )
    assert (
        await agent.before_tool_callback(
            tool("find_document"), {"file_path": urls[0], "query": "needle"}, ctx
        )
        is None
    )
    assert (
        await agent.before_tool_callback(
            tool("find_document"), {"file_path": urls[0], "query": "different"}, ctx
        )
    )["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"
    # One further known candidate has a navigation + read allowance.
    assert (
        await agent.before_tool_callback(
            tool("web_scraping_goto"), {"url": urls[1]}, ctx
        )
        is None
    )
    assert (
        await agent.before_tool_callback(
            tool("web_scraping_markdown"), {"url": urls[1]}, ctx
        )
        is None
    )
    assert (await agent.before_tool_callback(tool("html"), {"url": urls[2]}, ctx))[
        "error_code"
    ] == "RESEARCH_CONVERGENCE_REQUIRED"
    assert (
        await agent.before_tool_callback(
            tool("html"), {"url": "https://unknown.org"}, ctx
        )
    )["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"
    # Persisted state enforces the same quota across another recovery instance.
    restored = json.loads(json.dumps(state))
    assert not allow_fallback(
        restored,
        "reader",
        "html",
        state[RESEARCH_CANDIDATE_LEDGER_KEY]["reader"],
        args={"url": urls[2]},
        reserve=True,
    )


@pytest.mark.asyncio
async def test_blocked_methods_hidden_while_alternative_remains_visible():
    url = "https://example.org/a"
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="reader", instruction="Read", output_key="out"), None, None
    )
    await agent.before_agent_callback(ctx)
    record_retrieval(state, "reader", "markdown", {"url": url})
    state[RESEARCH_PROGRESS_STATE_KEY] = {
        "reader": {"version": 0, "no_progress_turns": 2}
    }
    req = request(tool("markdown"), tool("read_document"), tool("search"))
    await agent.before_model_callback(ctx, req)
    visible = {
        d.name for g in req.config.tools or [] for d in g.function_declarations or []
    }
    assert "markdown" not in visible and "search" not in visible
    assert "read_document" in visible
    assert "two alternative routes" in str(req.config.system_instruction)


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            {
                "first_date": None,
                "first_year": None,
                "evidence_url": ["https://example.org"],
                "notes": "Abstained: access blocked.",
            },
            True,
        ),
        ({"decision": "unresolved", "notes": "Missing source"}, True),
        (
            {
                "answer": "2018",
                "optional": None,
                "notes": "Could not read another source",
            },
            False,
        ),
        ({"value": 0, "optional": None, "notes": "Cannot find more records"}, False),
        ({"first_year": 2018, "notes": "Abstained: earlier attempt"}, False),
        ({"optional": None, "notes": "Read the complete source."}, False),
    ],
)
def test_structured_refusal_without_fixed_answer_key(value, expected):
    assert is_unresolved_answer(value) == expected
    assert is_unresolved_answer(json.dumps(value)) == expected


@pytest.mark.asyncio
async def test_download_is_a_distinct_route_and_real_file_reopens_reading():
    url = "https://example.org/data.zip"
    state = {}
    ctx = context(state)
    record_retrieval(state, "reader", "read_document", {"file_path": url})
    assert allow_fallback(
        state, "reader", "download", [], args={"urls": [url]}, reserve=True
    )
    telemetry = ResearchTelemetry()
    download = tool("download")
    for result in [
        {"success": False, "file_size": 0, "file_path": ""},
        {"success": True, "file_size": 0, "file_path": "/tmp/empty"},
    ]:
        await telemetry.after_tool_callback(
            tool=download,
            tool_args={"urls": [url]},
            tool_context=ctx,
            result={"structuredContent": result},
        )
    assert not state.get(RESEARCH_PROGRESS_STATE_KEY)
    result = {
        "structuredContent": {
            "results": [
                {"success": True, "file_size": 120, "file_path": "/tmp/data.zip"}
            ],
            "success_count": 1,
            "failed_count": 0,
        }
    }
    await telemetry.after_tool_callback(
        tool=download, tool_args={"urls": [url]}, tool_context=ctx, result=result
    )
    assert state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["progress_events"] == [
        "downloaded_file"
    ]
    # A different downloaded filename for the same URL earns no second credit.
    result["structuredContent"]["results"][0]["file_path"] = "/tmp/data_2.zip"
    await telemetry.after_tool_callback(
        tool=download, tool_args={"urls": [url]}, tool_context=ctx, result=result
    )
    assert state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["version"] == 1
