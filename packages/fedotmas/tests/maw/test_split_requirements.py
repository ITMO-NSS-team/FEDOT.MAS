"""Split requirements must retain the semantics of one combined requirement."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fedotmas import MAW
from fedotmas.control import Controller, Strategy, repair_agent_configs
from fedotmas.maw import builder
from fedotmas.maw.handoffs import unresolved_execution_issues
from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class Stub(BaseLlm):
    answer: str

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.answer)])
        )


@pytest.mark.parametrize("strategy", [Strategy.RESTART_ALL, Strategy.RESTART_AFTER])
@pytest.mark.parametrize("reverse", [False, True])
async def test_split_repair_cannot_hide_identity_mismatch(strategy, reverse):
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "source",
                    "instruction": "Read",
                    "model": "stub/source",
                    "output_key": "data",
                    "output_contract": {"identity_fields": ["entity", "year"]},
                },
                {
                    "name": "worker",
                    "instruction": "Preserve entity and year",
                    "model": "stub/worker",
                    "output_key": "result",
                    "input_requirements": [
                        {"source_key": "data", "identity_fields": ["entity", "year"]}
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
    stubs = {
        "stub/source": Stub(model="offline", answer='{"entity":"A","year":2020}'),
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
        before = await ctrl.run("Find A", config=cfg)
        assert before.status == "error"
        assert (
            unresolved_execution_issues(before.state)[0]["kind"]
            == "entity_continuity_mismatch"
        )
        ctx = SimpleNamespace(state={"config": cfg.model_dump_json()})
        fields = ["year", "entity"] if reverse else ["entity", "year"]
        result = await repair_agent_configs(
            ctx,
            [
                {
                    "agent_name": "worker",
                    "input_requirements": [
                        {"source_key": "data", "identity_fields": [field]}
                        for field in fields
                    ],
                }
            ],
            "Split without removing obligations",
        )
        assert result.startswith("Updated"), result
        repaired = MAWConfig.model_validate_json(ctx.state["config"])
        after = await ctrl.resume(repaired, strategy=strategy)
        assert json.loads(after.state["result"])["entity"] == "B"
        assert after.status == "error"
        assert unresolved_execution_issues(after.state)[0]["fields"] == ["entity"]

        # A real correction must still resolve the combined issue on another resume.
        stubs["stub/worker"].answer = '{"entity":"A","year":2020}'
        corrected = await ctrl.resume(repaired, strategy=strategy)
        assert corrected.status == "success"
        assert not unresolved_execution_issues(corrected.state)


@pytest.mark.parametrize("reverse", [False, True])
async def test_split_required_fields_remain_missing_until_all_are_recovered(reverse):
    fields = ["answer", "evidence"] if reverse else ["evidence", "answer"]
    cfg = MAWAgentConfig(
        name="worker",
        instruction="Recover missing evidence",
        model="stub/worker",
        output_key="result",
        research_policy="targeted_recovery",
        input_requirements=[
            {
                "source_key": "data",
                "required_fields": [field],
                "purpose": f"Need {field}",
            }
            for field in fields
        ],
        output_contract={"required_fields": ["answer"]},
    )
    original = cfg.model_dump()
    worker = builder._build_llm_agent(cfg, {}, {})
    state = {"data": {"answer": 42}, "result": {"answer": 42}}
    ctx = SimpleNamespace(state=state)
    await worker.before_agent_callback(ctx)
    assert unresolved_execution_issues(state)[0]["missing_fields"] == ["evidence"]
    await worker.after_agent_callback(ctx)
    assert unresolved_execution_issues(state)[0]["kind"] == "incomplete_handoff"
    state["result"] = {"answer": 42, "evidence": "source"}
    await worker.after_agent_callback(ctx)
    assert not unresolved_execution_issues(state)
    assert cfg.model_dump() == original
