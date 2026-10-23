"""Repairs preserve each input obligation, including entity continuity."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fedotmas import MAW
from fedotmas.control import (
    Controller,
    Strategy,
    fix_agent_configs,
    repair_agent_configs,
)
from fedotmas.control.fixes._fix_agent_configs import AVAILABLE_TOOLS_STATE_KEY
from fedotmas.maw.handoffs import unresolved_execution_issues
from fedotmas.maw.models import MAWConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def config():
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "source",
                    "instruction": "Read",
                    "model": "stub/source",
                    "output_key": "data",
                    "output_contract": {
                        "required_fields": ["answer", "evidence.url"],
                        "identity_fields": ["entity", "year"],
                    },
                },
                {
                    "name": "worker",
                    "instruction": "Preserve entity and year",
                    "model": "stub/worker",
                    "output_key": "result",
                    "input_requirements": [
                        {
                            "source_key": "data",
                            "required_fields": ["answer", "evidence.url"],
                            "identity_fields": ["entity", "year"],
                        }
                    ],
                },
                {
                    "name": "final",
                    "instruction": "Answer from {result}",
                    "model": "stub/final",
                    "output_key": "answer",
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [{"agent_name": n} for n in ("source", "worker", "final")],
            },
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
        return await fix_agent_configs(ctx, json.dumps(updates), "Regression")
    return await repair_agent_configs(ctx, updates, "Regression")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"required_fields": ["answer"]},
        {"identity_fields": ["year"]},
        {"required_fields": ["answer", "evidence"]},
        {
            "identity_fields": ["year"],
            "required_fields": ["answer", "evidence.url", "entity"],
        },
        {"source_key": "result"},
    ],
)
async def test_partial_deletion_weakening_or_relocation_rolls_back_batch(
    legacy, change
):
    ctx = context()
    before = ctx.state["config"]
    requirement = config().agents[1].input_requirements[0].model_dump()
    requirement.update(change)
    result = await repair(
        ctx,
        [
            {"agent_name": "source", "instruction": "Must roll back too"},
            {"agent_name": "worker", "input_requirements": [requirement]},
        ],
        legacy,
    )
    assert result.startswith("Error")
    assert ctx.state["config"] == before


@pytest.mark.parametrize("legacy", [False, True])
async def test_explicit_coordinated_rename_and_split_requirements(legacy):
    ctx = context()
    result = await repair(
        ctx,
        [
            {
                "agent_name": "source",
                "output_contract": {
                    "required_fields": ["answer", "citation.url"],
                    "identity_fields": ["entity", "year"],
                },
                "output_field_renames": [
                    {"old_path": "evidence.url", "new_path": "citation.url"}
                ],
            },
            {
                "agent_name": "worker",
                "input_requirements": [
                    {
                        "source_key": "data",
                        "required_fields": ["answer"],
                        "identity_fields": ["entity"],
                    },
                    {
                        "source_key": "data",
                        "required_fields": ["citation.url"],
                        "identity_fields": ["year"],
                    },
                ],
                "input_field_renames": [
                    {
                        "source_key": "data",
                        "old_path": "evidence.url",
                        "new_path": "citation.url",
                    }
                ],
            },
        ],
        legacy,
    )
    assert result.startswith("Updated")


@pytest.mark.parametrize(
    "renames",
    [
        [{"source_key": "missing", "old_path": "evidence.url", "new_path": "answer"}],
        [{"source_key": "data", "old_path": "missing", "new_path": "answer"}],
        [{"source_key": "data", "old_path": "evidence.url", "new_path": "answer"}],
        [
            {
                "source_key": "data",
                "old_path": "evidence.url",
                "new_path": "answer.detail",
            }
        ],
        [{"source_key": "data", "old_path": "evidence.url", "new_path": "evidence.url"}]
        * 2,
        [{"source_key": "data", "old_path": "evidence.url", "new_path": ""}],
    ],
)
async def test_bad_or_merging_renames_are_rejected(renames):
    ctx = context()
    before = ctx.state["config"]
    result = await repair_agent_configs(
        ctx,
        [
            {
                "agent_name": "worker",
                "instruction": "Changed",
                "input_requirements": [
                    r.model_dump() for r in config().agents[1].input_requirements
                ],
                "input_field_renames": renames,
            }
        ],
        "Bad mapping",
    )
    assert result.startswith("Error") and ctx.state["config"] == before


async def test_null_rename_is_ignored_and_rename_without_requirements_rejected():
    ctx = context()
    result = await repair_agent_configs(
        ctx,
        [
            {
                "agent_name": "worker",
                "instruction": "Changed",
                "input_field_renames": None,
            }
        ],
        "Change instruction",
    )
    assert result.startswith("Updated")
    before = ctx.state["config"]
    result = await repair_agent_configs(
        ctx, [{"agent_name": "worker", "input_field_renames": []}], "No requirements"
    )
    assert result.startswith("Error") and ctx.state["config"] == before


class Stub(BaseLlm):
    answer: str
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.answer)])
        )


@pytest.mark.parametrize("strategy", [Strategy.RESTART_ALL, Strategy.RESTART_AFTER])
async def test_real_recovery_cannot_hide_entity_change(strategy):
    cfg = config()
    stubs = {
        "stub/source": Stub(
            model="offline",
            answer='{"entity":"A","year":2020,"answer":42,"evidence":{"url":"source"}}',
        ),
        "stub/worker": Stub(model="offline", answer='{"entity":"B","year":2020}'),
        "stub/final": Stub(model="offline", answer="B"),
    }
    with (
        patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")),
        patch(
            "socket.socket.connect_ex", side_effect=AssertionError("Network forbidden")
        ),
        patch("fedotmas.core.base.setup_logging"),
        patch(
            "fedotmas.maw.builder._resolve_llm",
            side_effect=lambda model, _: stubs[model],
        ),
    ):
        ctrl = Controller(MAW(mcp_servers=[], plugins=[]))
        first = await ctrl.run("Find A", config=cfg)
        assert first.status == "error"
        assert (
            unresolved_execution_issues(first.state)[0]["kind"]
            == "entity_continuity_mismatch"
        )
        ctx = context(cfg)
        requirement = cfg.agents[1].input_requirements[0].model_dump()
        requirement["identity_fields"] = ["year"]
        result = await repair_agent_configs(
            ctx,
            [{"agent_name": "worker", "input_requirements": [requirement]}],
            "Hide mismatch",
        )
        assert result.startswith("Error")
        unchanged = MAWConfig.model_validate_json(ctx.state["config"])
        last = await ctrl.resume(unchanged, strategy=strategy)
        assert last.status == "error"
        assert any(
            i["kind"] == "entity_continuity_mismatch"
            for i in unresolved_execution_issues(last.state)
        )

        stubs["stub/worker"].answer = '{"entity":"A","year":2020}'
        result = await repair_agent_configs(
            ctx,
            [{"agent_name": "worker", "instruction": "Use the original entity"}],
            "Fix worker",
        )
        assert result.startswith("Updated")
        last = await ctrl.resume(
            MAWConfig.model_validate_json(ctx.state["config"]), strategy=strategy
        )
        assert last.status == "success"
        assert json.loads(last.state["data"])["entity"] == "A"
        assert not unresolved_execution_issues(last.state)
        assert stubs["stub/source"].calls == (
            1 if strategy == Strategy.RESTART_AFTER else 3
        )
