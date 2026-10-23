"""Recovery must not manufacture success by weakening an output contract."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fedotmas import MAW
from fedotmas._settings import ModelConfig
from fedotmas.control import (
    Controller,
    RunError,
    fix_agent_configs,
    repair_agent_configs,
)
from fedotmas.control.fixes._fix_agent_configs import AVAILABLE_TOOLS_STATE_KEY
from fedotmas.maw.handoffs import validate_output_contract
from fedotmas.maw.models import MAWConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def config():
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "solver",
                    "instruction": "Return answer and evidence",
                    "output_key": "answer",
                    "output_contract": {
                        "required_fields": ["answer", "evidence"],
                        "identity_fields": ["source_id"],
                    },
                }
            ],
            "pipeline": {"agent_name": "solver"},
        }
    )


def context(cfg=None):
    return SimpleNamespace(
        state={
            "config": (cfg or config()).model_dump_json(),
            AVAILABLE_TOOLS_STATE_KEY: [],
        }
    )


async def repair(ctx, updates, legacy):
    if legacy:
        return await fix_agent_configs(ctx, json.dumps(updates), "Offline regression")
    return await repair_agent_configs(ctx, updates, "Offline regression")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "contract",
    [
        None,
        {},
        {"description": "no checks"},
        {"required_fields": [], "identity_fields": []},
        {"required_fields": ["answer"], "identity_fields": ["source_id"]},
        {"required_fields": ["answer", "evidence"], "identity_fields": []},
        {"required_fields": ["answer", "citation"], "identity_fields": ["source_id"]},
        {"required_fields": ["answer", "evidence", "source_id"], "identity_fields": []},
    ],
)
@pytest.mark.asyncio
async def test_rejects_contract_removal_or_weakening_without_consumers(
    contract, legacy
):
    ctx = context()
    before = ctx.state["config"]
    result = await repair(
        ctx,
        [
            {
                "agent_name": "solver",
                "output_contract": contract,
                "instruction": "This edit must also roll back",
            }
        ],
        legacy,
    )
    assert result.startswith("Error applying repair")
    assert ctx.state["config"] == before


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_strengthening_and_explicit_rename_preserve_obligations(legacy):
    ctx = context()
    updates = [
        {
            "agent_name": "solver",
            "output_contract": {
                "required_fields": ["answer", "citation.url"],
                "identity_fields": ["source_id"],
            },
            "output_field_renames": [{"old_path": "evidence", "new_path": "citation"}],
        }
    ]
    assert (await repair(ctx, updates, legacy)).startswith("Updated")
    fixed = MAWConfig.model_validate_json(ctx.state["config"])
    assert validate_output_contract(
        {"answer": "x", "source_id": "s"}, fixed.agents[0].output_contract
    ) == ["citation.url"]


@pytest.mark.asyncio
@pytest.mark.parametrize("new_path", ["answer", "answer.detail"])
async def test_rename_cannot_merge_evidence_into_answer(new_path):
    ctx = context()
    before = ctx.state["config"]
    result = await repair_agent_configs(
        ctx,
        [
            {
                "agent_name": "solver",
                "output_contract": {
                    "required_fields": [new_path],
                    "identity_fields": ["source_id"],
                },
                "output_field_renames": [
                    {"old_path": "evidence", "new_path": new_path}
                ],
            }
        ],
        "Merge",
    )
    assert "must not merge" in result and ctx.state["config"] == before


@pytest.mark.asyncio
async def test_repair_can_add_contract_and_preserve_existing_with_unrelated_edit():
    raw = config().model_dump()
    raw["agents"][0]["output_contract"] = None
    ctx = context(MAWConfig.model_validate(raw))
    assert (
        await repair_agent_configs(
            ctx,
            [
                {
                    "agent_name": "solver",
                    "output_contract": {"required_fields": ["answer", "evidence"]},
                }
            ],
            "Add validation",
        )
    ).startswith("Updated")
    before = MAWConfig.model_validate_json(ctx.state["config"])
    assert (
        await repair_agent_configs(
            ctx, [{"agent_name": "solver", "instruction": "Better instructions"}], "Fix"
        )
    ).startswith("Updated")
    after = MAWConfig.model_validate_json(ctx.state["config"])
    assert after.agents[0].output_contract == before.agents[0].output_contract


@pytest.mark.asyncio
async def test_failed_batch_repair_is_atomic_for_other_agents():
    raw = config().model_dump()
    raw["agents"].insert(
        0, {"name": "source", "instruction": "Read", "output_key": "data"}
    )
    raw["pipeline"] = {
        "type": "sequential",
        "children": [{"agent_name": "source"}, {"agent_name": "solver"}],
    }
    ctx = context(MAWConfig.model_validate(raw))
    before = ctx.state["config"]
    result = await repair_agent_configs(
        ctx,
        [
            {"agent_name": "source", "instruction": "Changed"},
            {"agent_name": "solver", "output_contract": None},
        ],
        "Invalid transaction",
    )
    assert result.startswith("Error") and ctx.state["config"] == before


@pytest.mark.asyncio
async def test_real_debugger_can_rename_producer_and_consumer_together(monkeypatch):
    import fedotmas.control._controller as module

    raw = config().model_dump()
    raw["agents"].append(
        {
            "name": "consumer",
            "instruction": "Use the evidence",
            "output_key": "final",
            "input_requirements": [
                {"source_key": "answer", "required_fields": ["evidence"]}
            ],
        }
    )
    raw["pipeline"] = {
        "type": "sequential",
        "children": [{"agent_name": "solver"}, {"agent_name": "consumer"}],
    }
    cfg = MAWConfig.model_validate(raw)
    updates = [
        {
            "agent_name": "solver",
            "output_contract": {
                "required_fields": ["answer", "citation"],
                "identity_fields": ["source_id"],
            },
            "output_field_renames": [{"old_path": "evidence", "new_path": "citation"}],
        },
        {
            "agent_name": "consumer",
            "input_requirements": [
                {"source_key": "answer", "required_fields": ["citation"]}
            ],
        },
    ]

    class Stub(BaseLlm):
        calls: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.calls += 1
            part = (
                types.Part(
                    function_call=types.FunctionCall(
                        name="repair_agent_configs",
                        args={"updates": updates, "reasoning": "Align explicit rename"},
                    )
                )
                if self.calls == 1
                else types.Part(text="Done")
            )
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    stub = Stub(model="offline")
    monkeypatch.setattr(module, "make_llm", lambda _: stub)
    maw = SimpleNamespace(
        meta_model=ModelConfig(model="offline"), _session_service=None, mcp_servers={}
    )
    fixed = await Controller(maw)._run_debugger(
        task="Keep answer and evidence",
        error=RunError(agent_name="solver", message="Wrong field name"),
        current_config=cfg,
        state={},
    )
    assert stub.calls == 2
    assert fixed.agents[0].output_contract.required_fields == ["answer", "citation"]
    assert fixed.agents[1].input_requirements[0].required_fields == ["citation"]


@pytest.mark.asyncio
async def test_missing_evidence_remains_incomplete_after_attempted_contract_removal():
    class Stub(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text='{"answer":"unverified","source_id":"s"}')],
                )
            )

    cfg = config()
    ctx = context(cfg)
    with (
        patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")),
        patch("fedotmas.maw.builder._resolve_llm", return_value=Stub(model="offline")),
        patch(
            "fedotmas.maw.builder._repair_contract_once",
            AsyncMock(
                return_value=(
                    None,
                    {"prompt_tokens": 0, "completion_tokens": 0},
                    "stop",
                )
            ),
        ),
    ):
        maw = MAW(mcp_servers=[], plugins=[])
        first = await Controller(maw).run("Offline regression", config=cfg)
        assert first.status == "error" and maw.last_result.status == "incomplete"
        result = await repair_agent_configs(
            ctx,
            [{"agent_name": "solver", "output_contract": None}],
            "Remove validation",
        )
        assert result.startswith("Error")
        fixed = MAWConfig.model_validate_json(ctx.state["config"])
        last = await Controller(maw).run("Offline regression", config=fixed)
        assert last.status == "error" and maw.last_result.status == "incomplete"
        assert first.state["answer"] == last.state["answer"]
