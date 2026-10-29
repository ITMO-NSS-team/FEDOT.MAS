"""Offline regressions for extraction claims without executed source reads."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import Field

from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw._file_evidence import (
    FILE_EVIDENCE_STATE_KEY,
    record_file_evidence,
    reject_file_artifact,
    remember_file_artifact,
    unsupported_file_artifact,
)
from fedotmas.maw.handoffs import is_unresolved_answer
from fedotmas.maw.models import ArtifactContract, MAWAgentConfig, MAWConfig


def context(state):
    ctx = MagicMock()
    ctx.state = state
    return ctx


def tool(name):
    t = FunctionTool(lambda file_path: "")
    t.name = name
    return t


def config(**kwargs):
    return MAWAgentConfig(
        name="reader",
        instruction="Extract the records.",
        output_key="records",
        output_contract=ArtifactContract(required_fields=["records"]),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_response",
    [
        None,
        {"isError": True, "content": [{"text": "Access denied"}]},
        {"content": [{"text": '{"error":"denied","content":"invented"}'}]},
        {"structuredContent": {"content": "", "total_chars": 500}},
        {"structuredContent": {"matches": [], "total_matches": 0}},
    ],
)
async def test_preparation_and_failed_reads_cannot_validate_fabricated_json(
    monkeypatch, read_response
):
    repair = AsyncMock(
        side_effect=AssertionError("Must not repair fabricated extraction")
    )
    monkeypatch.setattr(builder, "_repair_contract_once", repair)
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(config(), None, None)
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("extract_zip"),
        {"file_path": "/tmp/data.zip"},
        ctx,
        {"structuredContent": {"files": ["Records.xlsx"], "total_extracted": 1}},
    )
    if read_response is not None:
        await agent.after_tool_callback(
            tool("read_document"), {"file_path": "Records.xlsx"}, ctx, read_response
        )
    state["records"] = (
        'read_document({"file_path":"Records.xlsx"})\n{"records":[{"name":"Invented person"}]}'
    )
    await agent.after_agent_callback(ctx)
    assert is_unresolved_answer(state["records"])
    assert "Invented person" not in state["records"]
    repair.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,response",
    [
        (
            "read_document",
            {"structuredContent": {"content": "Ada,42", "total_lines": 1}},
        ),
        (
            "read_document",
            {"content": [{"type": "text", "text": '{"content":"Ada,42"}'}]},
        ),
        (
            "find_document",
            {"structuredContent": {"matches": [{"line": 1, "text": "Ada,42"}]}},
        ),
    ],
)
async def test_executed_nonempty_reads_create_runtime_receipts(name, response):
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(config(), None, None)
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("extract_zip"), {}, ctx, {"files": ["records.csv"]}
    )
    await agent.after_tool_callback(
        tool(name), {"file_path": "records.csv", "start_line": 0}, ctx, response
    )
    value = '{"records":[{"name":"Ada","value":42}]}'
    state["records"] = value
    await agent.after_agent_callback(ctx)
    assert state["records"] == value
    root = state[FILE_EVIDENCE_STATE_KEY]
    assert root["agents"]["reader"]["reads"][0]["range"] == {"start_line": 0}
    assert "records" in root["artifacts"]
    assert "Ada" not in json.dumps(root)


@pytest.mark.asyncio
async def test_locator_can_pass_actual_paths_without_claiming_contents():
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(config(), None, None)
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("extract_zip"),
        {"file_path": "/tmp/archive.zip"},
        ctx,
        {
            "structuredContent": {
                "output_dir": "/tmp/extracted",
                "files": ["a.csv", "b.pdf"],
            }
        },
    )
    state["records"] = (
        '{"records":["/tmp/extracted/a.csv","/tmp/extracted/b.pdf","/tmp/archive.zip"]}'
    )
    await agent.after_agent_callback(ctx)
    assert not is_unresolved_answer(state["records"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,args,response",
    [
        (
            "markdown",
            {"url": "https://example.org/file"},
            {"content": [{"text": "Ada"}]},
        ),
        (
            "solve_with_code",
            {"files": ["records.csv"]},
            {"status": "completed", "answer": "Ada", "files_used": ["records.csv"]},
        ),
    ],
)
async def test_other_executed_file_reading_routes_remain_usable(name, args, response):
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(config(), None, None)
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("download"), {}, ctx, {"success": True, "file_path": "records.csv"}
    )
    await agent.after_tool_callback(tool(name), args, ctx, response)
    state["records"] = '{"records":["Ada"]}'
    await agent.after_agent_callback(ctx)
    assert not is_unresolved_answer(state["records"])


def test_only_unchanged_upstream_artifact_with_read_receipt_can_supply_evidence():
    state = {"records": '{"records":["Ada"]}'}
    source = config()
    record_file_evidence(
        state,
        source.name,
        "read_document",
        {"file_path": "records.csv"},
        {"content": "Ada"},
    )
    remember_file_artifact(state, source)
    consumer = MAWAgentConfig(
        name="counter",
        instruction="Count the records",
        output_key="count",
        input_requirements=[{"source_key": "records", "required_fields": ["records"]}],
        output_contract=ArtifactContract(required_fields=["count"]),
    )
    record_file_evidence(state, consumer.name, "list_zip_contents", {}, {"entries": []})
    assert not unsupported_file_artifact(state, consumer, {"count": 1})
    # A later unverified model overwrite cannot reuse a receipt for old data.
    state["records"] = '{"records":["Invented"]}'
    assert unsupported_file_artifact(state, consumer, {"count": 1})


@pytest.mark.asyncio
async def test_loop_invocation_cannot_borrow_previous_read_receipts():
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(config(), None, None)
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(tool("read_document"), {}, ctx, {"content": "Ada"})
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("extract_zip"), {}, ctx, {"files": ["other.csv"]}
    )
    state["records"] = '{"records":["Invented"]}'
    await agent.after_agent_callback(ctx)
    assert is_unresolved_answer(state["records"])


@pytest.mark.asyncio
async def test_policy_block_is_not_an_executed_read():
    state = {}
    ctx = context(state)
    agent = builder._build_llm_agent(config(), None, None)
    await agent.before_agent_callback(ctx)
    await agent.after_tool_callback(
        tool("extract_zip"), {}, ctx, {"files": ["records.csv"]}
    )
    await agent.after_tool_callback(
        tool("read_document"),
        {},
        ctx,
        {
            "error_code": "RESEARCH_CONVERGENCE_REQUIRED",
            "content": "Policy explanation",
        },
    )
    state["records"] = '{"records":["Invented"]}'
    await agent.after_agent_callback(ctx)
    assert is_unresolved_answer(state["records"])


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["{records}", "{records.records}"])
async def test_legacy_instruction_dependency_cannot_consume_rejected_artifact(
    reference,
):
    state = {}
    reject_file_artifact(state, "records", "No executed source reads")
    agent = builder._build_llm_agent(
        MAWAgentConfig(
            name="counter", instruction="Count " + reference, output_key="answer"
        ),
        None,
        None,
    )
    assert await agent.before_agent_callback(context(state)) is not None
    assert is_unresolved_answer(state["answer"])


class Scripted(BaseLlm):
    model: str = "offline"
    outputs: list = Field(default_factory=list)
    calls: int = 0

    async def generate_content_async(self, request, stream=False):
        output = self.outputs[self.calls]
        self.calls += 1
        part = (
            types.Part(text=output)
            if isinstance(output, str)
            else types.Part.from_function_call(**output)
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [False, True])
async def test_pipeline_blocks_dependent_calculation_only_when_source_was_not_read(
    monkeypatch, read
):
    outputs = [{"name": "extract_zip", "args": {}}]
    if read:
        outputs.append({"name": "read_document", "args": {}})
    outputs.append('Simulated call read_document()\n{"records":["Ada"]}')
    reader = Scripted(outputs=outputs)
    counter = Scripted(outputs=["1"])
    models = iter([reader, counter])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: next(models))
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                config().model_dump(),
                {
                    "name": "counter",
                    "instruction": "Count {records}.",
                    "output_key": "answer",
                    "input_requirements": [
                        {"source_key": "records", "required_fields": ["records"]}
                    ],
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [{"agent_name": "reader"}, {"agent_name": "counter"}],
            },
        }
    )
    root = builder.build(cfg, autonomous=False)

    def extract_zip() -> dict:
        """Extract the test archive."""
        return {"files": ["records.csv"]}

    def read_document() -> dict:
        """Read the extracted records."""
        return {"content": "Ada"}

    root.sub_agents[0].tools = [FunctionTool(extract_zip), FunctionTool(read_document)]
    result = await run_pipeline(root, "Count the supplied records.")
    if read:
        assert result.status == "completed"
        assert result.state["answer"] == "1"
        assert counter.calls == 1
    else:
        assert result.status == "incomplete"
        assert is_unresolved_answer(result.state["answer"])
        assert counter.calls == 0
