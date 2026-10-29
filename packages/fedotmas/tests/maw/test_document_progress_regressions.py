import json
from unittest.mock import MagicMock

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool
from google.adk.sessions.state import State
from google.genai import types

from fedotmas.maw import builder
from fedotmas.maw.models import ArtifactContract, ArtifactRequirement, MAWAgentConfig
from fedotmas.plugins import ResearchTelemetry
from fedotmas.plugins._research_telemetry import (
    MAX_DOCUMENT_FRAGMENT_PROGRESS,
    RESEARCH_PROGRESS_STATE_KEY,
)


def request(*tools):
    return LlmRequest(
        tools_dict={t.name: t for t in tools},
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name=t.name) for t in tools
                    ]
                )
            ]
        ),
    )


@pytest.mark.asyncio
async def test_find_and_new_document_ranges_continue_but_repeated_content_stops():
    read = FunctionTool(lambda file_path, start_line=0: {})
    read.name = "read_document"
    find = FunctionTool(lambda file_path, query: {})
    find.name = "find_document"
    url = "https://example.org/paper?b=two&a=one"
    cfg = MAWAgentConfig(
        name="reader",
        instruction="Read",
        output_key="out",
        research_mode="inspection_only",
        max_llm_turns=20,
        output_contract=ArtifactContract(required_fields=["chemicals"]),
        input_requirements=[
            ArtifactRequirement(source_key="source", required_fields=["url"])
        ],
    )
    agent = builder._build_llm_agent(cfg, [read, find], None, autonomous=False)
    ctx = MagicMock()
    state = {"source": {"url": url}}
    ctx.state = state
    ctx._invocation_context.session.state = state
    ctx._invocation_context.agent.name = "reader"
    telemetry = ResearchTelemetry()
    await agent.before_agent_callback(ctx)

    async def turn(tool, args, result):
        await agent.before_model_callback(ctx, request(read, find))
        block = await agent.before_tool_callback(tool, args, ctx)
        assert block is None
        await telemetry.before_tool_callback(
            tool=tool, tool_args=args, tool_context=ctx
        )
        await telemetry.after_tool_callback(
            tool=tool, tool_args=args, tool_context=ctx, result=result
        )
        await agent.after_tool_callback(tool, args, ctx, result)

    await turn(
        read,
        {"file_path": url},
        {
            "content": "Introduction\n",
            "start_line": 0,
            "next_start_char": 13,
            "total_chars": 1000,
            "truncated": True,
        },
    )
    await turn(
        find,
        {"file_path": url, "query": "Materials and Methods"},
        {"matches": [], "total_matches": 0},
    )
    await turn(
        find,
        {"file_path": url, "query": "Materials"},
        {
            "matches": [{"line": 100, "text": "Materials and methods"}],
            "total_matches": 1,
        },
    )
    for line in [99, 100, 101, 102]:
        await turn(
            read,
            {"file_path": url, "start_line": line},
            {"content": f"Distinct evidence on line {line}\n", "start_line": line},
        )
    # Merely changing the requested cursor cannot make an identical reply progress.
    last = {"content": "Distinct evidence on line 102\n", "start_line": 102}
    for line in [103, 104]:
        await turn(read, {"file_path": url, "start_line": line}, last)
    await agent.before_model_callback(ctx, request(read, find))
    assert (
        await agent.before_tool_callback(
            read, {"file_path": url, "start_line": 105}, ctx
        )
    )["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["html", "web_scraping_html", "web-scraping__html"])
async def test_html_obeys_inspection_policy_and_convergence(name):
    tool = FunctionTool(lambda url: "page")
    tool.name = name
    for mode, force, code in [
        ("mixed", True, "RESEARCH_CONVERGENCE_REQUIRED"),
        ("discovery_only", False, "DISCOVERY_ONLY_INSPECTION_DISABLED"),
    ]:
        agent = builder._build_llm_agent(
            MAWAgentConfig(
                name="reader", instruction="Read", output_key="out", research_mode=mode
            ),
            [tool],
            None,
            autonomous=False,
        )
        ctx = MagicMock()
        ctx.state = {}
        ctx._invocation_context.session.state = ctx.state
        await agent.before_agent_callback(ctx)
        ctx.state[builder.RESEARCH_TURN_STATE_KEY] = {
            "reader": {"force_converge": force}
        }
        assert (
            await agent.before_tool_callback(tool, {"url": "https://example.org"}, ctx)
        )["error_code"] == code


async def record_document(state, result, name="read_document", **args):
    ctx = MagicMock()
    ctx.state = state
    ctx._invocation_context.agent.name = "reader"
    tool = FunctionTool(lambda file_path: {})
    tool.name = name
    # A recreated plugin must retain deduplication through persisted ADK state.
    await ResearchTelemetry().after_tool_callback(
        tool=tool,
        tool_args={"file_path": "/tmp/paper.pdf", **args},
        tool_context=ctx,
        result=result,
    )


def progress(state):
    return (
        state.get(RESEARCH_PROGRESS_STATE_KEY, {}).get("reader", {}).get("version", 0)
    )


@pytest.mark.asyncio
async def test_character_pages_persist_deduplication_and_have_bounded_progress():
    state = State({}, {})
    for page in range(MAX_DOCUMENT_FRAGMENT_PROGRESS + 3):
        await record_document(
            state,
            {
                "structuredContent": {
                    "content": f"page {page:02d}",
                    "next_start_char": (page + 1) * 7,
                    "truncated": True,
                }
            },
        )
        assert progress(state) == min(page + 1, MAX_DOCUMENT_FRAGMENT_PROGRESS)
        if page < MAX_DOCUMENT_FRAGMENT_PROGRESS:
            assert RESEARCH_PROGRESS_STATE_KEY in state._delta
        state = State(json.loads(json.dumps(state.to_dict())), {})
    assert (
        len(state[RESEARCH_PROGRESS_STATE_KEY]["reader"]["document_fragments"])
        == MAX_DOCUMENT_FRAGMENT_PROGRESS
    )


@pytest.mark.asyncio
async def test_overlap_changed_query_and_error_wrappers_do_not_count_as_progress():
    state = {}
    await record_document(state, {"content": "abcdefghij", "next_start_char": 10})
    assert progress(state) == 1
    # Same text at a different claimed cursor, and different text in an already read range.
    for payload in [
        {"content": "abcdefghij", "next_start_char": 20},
        {"content": "changed", "next_start_char": 9},
        {"content": "", "start_line": 40},
        {"content": "failure", "start_line": 40, "error": "unavailable"},
        {
            "isError": True,
            "structuredContent": {"content": "failure", "start_line": 40},
        },
        {"content": [{"type": "text", "text": '{"matches":[],"total_matches":0}'}]},
    ]:
        await record_document(state, payload)
        assert progress(state) == 1
    for query in ["first", "rephrased"]:
        await record_document(
            state,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "matches": [{"line": 45, "text": "Methods"}],
                                "total_matches": 1,
                            }
                        ),
                    }
                ]
            },
            name="find_document",
            query=query,
        )
        assert progress(state) == 2
    # A genuinely new match and an extending page remain useful.
    await record_document(
        state, {"matches": [{"line": 46, "text": "Reagents"}]}, name="find_document"
    )
    await record_document(state, {"content": "hijklmnopq", "next_start_char": 17})
    assert progress(state) == 4


@pytest.mark.asyncio
async def test_retry_after_empty_find_cannot_credit_the_same_candidate_again():
    url = "https://example.org/paper?b=two&a=one"
    state = {
        builder.RESEARCH_CANDIDATE_LEDGER_KEY: {
            "reader": [{"url": url, "inspection_status": "uninspected"}]
        }
    }
    page = {"content": "same page", "start_line": 0}
    await record_document(state, page, file_path=url)
    assert progress(state) == 1
    assert (
        state[builder.RESEARCH_CANDIDATE_LEDGER_KEY]["reader"][0]["inspection_status"]
        == "success"
    )
    await record_document(state, {"matches": []}, name="find_document", file_path=url)
    await record_document(state, page, file_path=url)
    assert progress(state) == 1
