import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fedotmas.control import Controller, fix_agent_configs, fix_instruction
from fedotmas.control.fixes._fix_agent_configs import AVAILABLE_TOOLS_STATE_KEY
from fedotmas.core.runner import PipelineResult
from fedotmas.maw.contracts import handoff_contract_errors
from fedotmas.maw.models import MAWConfig


def config():
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "source",
                    "instruction": "Read the file",
                    "output_key": "locations",
                    "tools": ["sandbox-light"],
                    "output_contract": {"required_fields": ["locations"]},
                },
                {
                    "name": "consumer",
                    "instruction": "Map locations",
                    "output_key": "answer",
                    "input_requirements": [
                        {"source_key": "locations", "required_fields": ["place_name"]}
                    ],
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [{"agent_name": "source"}, {"agent_name": "consumer"}],
            },
            "final_answer_agent": "consumer",
        }
    )


def context():
    return SimpleNamespace(
        state={
            "config": config().model_dump_json(),
            AVAILABLE_TOOLS_STATE_KEY: ["sandbox-light", "document"],
        }
    )


def coherent_updates():
    return [
        {
            "agent_name": "source",
            "tools": ["document"],
            "output_contract": {"required_fields": ["locations[].place_name"]},
        },
        {
            "agent_name": "consumer",
            "input_requirements": [
                {
                    "source_key": "locations",
                    "required_fields": ["locations[].place_name"],
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_related_tool_and_contract_fix_commits_atomically():
    ctx = context()
    result = await fix_agent_configs(
        ctx, json.dumps(coherent_updates()), "Match list shape"
    )
    assert result.startswith("Updated agents")
    fixed = MAWConfig.model_validate_json(ctx.state["config"])
    assert fixed.agents[0].tools == ["document"]
    assert not handoff_contract_errors(fixed)
    assert fixed.final_answer_agent == "consumer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        coherent_updates()[:1],  # Partial repair leaves consumer inconsistent.
        [{"agent_name": "source", "tools": ["invented-reader"]}],
        [{"agent_name": "missing", "instruction": "Read"}],
        [{"agent_name": "source", "name": "renamed"}],
        [{"agent_name": "source", "model": "different-model"}],
        [{"agent_name": "consumer", "input_requirements": []}],
        [
            {
                "agent_name": "consumer",
                "input_requirements": [{"source_key": "locations"}],
            }
        ],
    ],
)
async def test_invalid_repair_leaves_original_config_intact(updates):
    ctx = context()
    before = ctx.state["config"]
    result = await fix_agent_configs(ctx, json.dumps(updates), "Fix")
    assert result.startswith("Error applying repair")
    assert ctx.state["config"] == before


@pytest.mark.asyncio
async def test_missing_registry_cannot_assign_tools():
    ctx = context()
    del ctx.state[AVAILABLE_TOOLS_STATE_KEY]
    assert "registry is missing" in await fix_agent_configs(
        ctx, json.dumps(coherent_updates()), "Fix"
    )


@pytest.mark.asyncio
async def test_instruction_fix_preserves_explicit_terminal_agent():
    ctx = context()
    await fix_instruction(ctx, "source", "Read the exact path", "Correct file access")
    assert (
        MAWConfig.model_validate_json(ctx.state["config"]).final_answer_agent
        == "consumer"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "limited"])
async def test_controller_recognizes_non_exception_failure_and_preserves_plugins(
    status,
):
    cfg = config()
    maw = MagicMock()
    maw._plugins = [SimpleNamespace(name="trace")]
    maw.generate_config = AsyncMock(return_value=cfg)
    state = {
        "_fedotmas_execution": {
            "handoff_issues": [{"agent": "source", "resolved": False}],
            "limited_agents": {"source": {"turns": 20}} if status == "limited" else {},
        }
    }
    result = PipelineResult(state=state, status=status)
    with patch(
        "fedotmas.control._controller.run_pipeline", AsyncMock(return_value=result)
    ) as run:
        controlled = await Controller(maw).run("Task")
    assert controlled.status == "error" and controlled.error.agent_name == "source"
    assert maw._last_result is result
    assert maw._plugins[0] in run.call_args.kwargs["plugins"]


@pytest.mark.asyncio
async def test_real_debugger_can_apply_structural_tool_with_stub_model(monkeypatch):
    import fedotmas.control._controller as module
    from fedotmas._settings import ModelConfig
    from fedotmas.control import RunError
    from fedotmas.mcp import StdioMCPServer
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class Stub(BaseLlm):
        calls: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.calls += 1
            if self.calls == 1:
                instruction = str(llm_request.config.system_instruction)
                assert "AVAILABLE MCP SERVERS" in instruction
                assert "document" in instruction and "sandbox-light" in instruction
                part = types.Part(
                    function_call=types.FunctionCall(
                        name="repair_agent_configs",
                        args={
                            "updates": coherent_updates(),
                            "reasoning": "Read file and align list paths",
                        },
                    )
                )
            else:
                part = types.Part(text="Fixed related agents.")
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    stub = Stub(model="offline")
    monkeypatch.setattr(module, "make_llm", lambda _: stub)
    maw = SimpleNamespace(
        meta_model=ModelConfig(model="offline"),
        _session_service=None,
        mcp_servers={
            name: StdioMCPServer(command="unused", args=(), description=name)
            for name in ["document", "sandbox-light"]
        },
    )
    updated = await Controller(maw)._run_debugger(
        task="Read attachment and map locations",
        error=RunError(
            agent_name="source", message="Wrong tools and incompatible paths"
        ),
        current_config=config(),
        state={},
    )
    assert stub.calls == 2
    assert updated.agents[0].tools == ["document"]
    assert not handoff_contract_errors(updated)


@pytest.mark.asyncio
async def test_execution_failure_preserves_partial_result_and_active_config():
    from fedotmas.core.runner import PipelineExecutionError

    maw = MagicMock()
    maw._plugins = []
    cfg = config()
    result = PipelineResult(status="failed", state={"locations": "partial evidence"})
    failure = PipelineExecutionError(RuntimeError("Agent 'consumer' failed"), result)
    with patch(
        "fedotmas.control._controller.run_pipeline", AsyncMock(side_effect=failure)
    ):
        controlled = await Controller(maw).run("Task", config=cfg)
    assert controlled.status == "error" and controlled.state == result.state
    assert maw._last_result is result and maw._generated_config is cfg


@pytest.mark.asyncio
async def test_duplicate_plugin_names_are_overridden_by_explicit_plugins():
    maw = MagicMock()
    maw._plugins = [SimpleNamespace(name="trace")]
    explicit = SimpleNamespace(name="trace")
    with patch(
        "fedotmas.control._controller.run_pipeline",
        AsyncMock(return_value=PipelineResult()),
    ) as run:
        await Controller(maw).run("Task", config=config(), plugins=[explicit])
    plugins = run.call_args.kwargs["plugins"]
    assert sum(p.name == "trace" for p in plugins) == 1
    assert explicit in plugins
