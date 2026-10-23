"""Nullable tool arguments mean no change, never removal of requirements."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fedotmas._settings import ModelConfig
from fedotmas.control import (
    Controller,
    RunError,
    fix_agent_configs,
    repair_agent_configs,
)
from fedotmas.control.fixes._fix_agent_configs import (
    AVAILABLE_TOOLS_STATE_KEY,
    AgentConfigRepair,
)
from fedotmas.maw.models import MAWConfig
from fedotmas.mcp import StdioMCPServer
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

FIXTURES = Path(__file__).parent / "fixtures"
CASE = json.loads((FIXTURES / "recovery_cases.json").read_text())[0]
RECORDED = json.loads((FIXTURES / "repair_with_nulls.json").read_text())
TOOLS = ["document", "sandbox-light", "web-scraping", "websearch-tavily"]
OPTIONAL = set(AgentConfigRepair.model_fields) - {"agent_name"}


def config():
    return MAWConfig.model_validate(CASE["config"])


def context():
    return SimpleNamespace(
        state={
            "config": config().model_dump_json(),
            AVAILABLE_TOOLS_STATE_KEY: TOOLS,
        }
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This regression must not use the network")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_model", [False, True])
async def test_recorded_partial_contract_repair_accepts_null_placeholders(as_model):
    ctx = context()
    original = config()
    updates = deepcopy(RECORDED["updates"])
    if as_model:
        updates = [AgentConfigRepair.model_validate(u) for u in updates]
    result = await repair_agent_configs(ctx, updates, RECORDED["reasoning"])
    assert result.startswith("Updated"), result
    after = MAWConfig.model_validate_json(ctx.state["config"])
    for old, new in zip(original.agents, after.agents):
        assert old.instruction == new.instruction
        assert old.tools == new.tools
        assert old.input_requirements == new.input_requirements
        if old.name != CASE["failed_agent"]:
            assert old == new
    contract = next(
        a for a in after.agents if a.name == CASE["failed_agent"]
    ).output_contract
    assert "total_usd" in contract.required_fields
    assert "included_items[].item_name" in contract.identity_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("as_model", [False, True])
async def test_instruction_edit_with_all_other_fields_null_preserves_contract(as_model):
    ctx = context()
    original = config()
    old = original.agents[1]
    update = {name: None for name in OPTIONAL}
    update.update(agent_name=old.name, instruction=old.instruction + "\nKeep evidence.")
    if as_model:
        update = AgentConfigRepair.model_validate(update)
    result = await repair_agent_configs(ctx, [update], "Only instruction changes")
    assert result.startswith("Updated"), result
    after = MAWConfig.model_validate_json(ctx.state["config"])
    expected = original.model_dump()
    expected["agents"][1]["instruction"] = old.instruction + "\nKeep evidence."
    assert after.model_dump() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("field", sorted(OPTIONAL) + ["all"])
async def test_null_only_update_is_rejected_without_changes(field):
    ctx = context()
    before = ctx.state["config"]
    update = {name: None for name in OPTIONAL} if field == "all" else {field: None}
    update["agent_name"] = CASE["failed_agent"]
    result = await repair_agent_configs(ctx, [update], "No actual change")
    assert result.startswith("Error") and ctx.state["config"] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("output_contract", {}),
        ("input_requirements", []),
        ("tools", ["unavailable"]),
        ("tools", "sandbox-light"),
        ("unknown_field", None),
        ("output_contract", {"required_fields": None}),
        (
            "output_field_renames",
            [{"old_path": "items[].item_name", "new_path": "other"}],
        ),
    ],
)
async def test_invalid_non_null_update_still_rejects_entire_batch(field, value):
    ctx = context()
    before = ctx.state["config"]
    update = {name: None for name in OPTIONAL}
    update.update(agent_name=CASE["failed_agent"])
    update[field] = value
    result = await repair_agent_configs(
        ctx,
        [
            {"agent_name": config().agents[0].name, "instruction": "Must roll back"},
            update,
        ],
        "Invalid second edit",
    )
    assert result.startswith("Error") and ctx.state["config"] == before


@pytest.mark.asyncio
async def test_empty_tools_is_an_explicit_edit_not_a_null_placeholder():
    ctx = context()
    old = config().agents[1]
    assert old.tools
    result = await repair_agent_configs(
        ctx,
        [
            {"agent_name": old.name, "tools": [], "output_contract": None},
        ],
        "Remove tool access but keep requirements",
    )
    assert result.startswith("Updated"), result
    after = MAWConfig.model_validate_json(ctx.state["config"]).agents[1]
    assert after.tools == []
    assert after.output_contract == old.output_contract
    assert after.input_requirements == old.input_requirements


@pytest.mark.asyncio
async def test_legacy_explicit_contract_deletion_remains_an_error():
    ctx = context()
    before = ctx.state["config"]
    result = await fix_agent_configs(
        ctx,
        json.dumps(
            [
                {"agent_name": CASE["failed_agent"], "output_contract": None},
            ]
        ),
        "Attempt removal",
    )
    assert result.startswith("Error") and ctx.state["config"] == before


@pytest.mark.asyncio
async def test_recorded_null_arguments_work_through_real_adk_debugger(monkeypatch):
    import fedotmas.control._controller as module

    class Stub(BaseLlm):
        calls: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.calls += 1
            if self.calls == 1:
                part = types.Part(
                    function_call=types.FunctionCall(
                        name="repair_agent_configs",
                        args=deepcopy(RECORDED),
                    )
                )
            else:
                responses = [
                    part.function_response
                    for content in llm_request.contents
                    for part in content.parts or []
                    if part.function_response
                ]
                assert responses and "Updated agents" in str(responses[-1].response)
                part = types.Part(text="Done")
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    stub = Stub(model="offline")
    monkeypatch.setattr(module, "make_llm", lambda _: stub)
    maw = SimpleNamespace(
        meta_model=ModelConfig(model="offline"),
        _session_service=None,
        mcp_servers={name: StdioMCPServer(command="unused", args=()) for name in TOOLS},
    )
    original = config()
    updated = await Controller(maw)._run_debugger(
        task="Offline replay",
        error=RunError(agent_name=CASE["failed_agent"], message="incomplete artifact"),
        current_config=original,
        state={},
    )
    assert stub.calls == 2
    assert updated != original
    for old, new in zip(original.agents, updated.agents):
        assert old.instruction == new.instruction
        assert old.tools == new.tools
        assert old.input_requirements == new.input_requirements
