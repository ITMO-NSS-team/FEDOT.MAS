"""Replay recovery defects using saved GAIA configurations and offline models."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fedotmas._settings import ModelConfig
from fedotmas.control import Controller, RunError, repair_agent_configs
from fedotmas.control._run import ControlledRun
from fedotmas.control.fixes._fix_agent_configs import AVAILABLE_TOOLS_STATE_KEY
from fedotmas.maw.models import MAWConfig
from fedotmas.mcp import StdioMCPServer
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

CASES = json.loads((Path(__file__).parent / "fixtures/recovery_cases.json").read_text())


@pytest.mark.parametrize(
    "status, expected", [("incomplete", True), ("limited", True), ("timed_out", False)]
)
@pytest.mark.asyncio
async def test_retry_decision_uses_execution_status_not_quoted_timeout(
    status, expected, monkeypatch
):
    from unittest.mock import MagicMock

    import fedotmas.control._controller as module
    from fedotmas.core.runner import PipelineResult

    config = MAWConfig.model_validate(CASES[0]["config"])
    state = {
        "_fedotmas_execution": {
            "handoff_issues": [
                {
                    "kind": "incomplete_handoff",
                    "agent": config.agents[0].name,
                    "received": "Suggestion: increase timeout; no connection error occurred",
                }
            ]
        }
    }
    monkeypatch.setattr(
        module,
        "run_pipeline",
        AsyncMock(return_value=PipelineResult(status=status, state=state)),
    )
    maw = MagicMock()
    maw._plugins = []
    result = await Controller(maw).run("task", config=config)
    assert module._is_retryable(result.error) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[c["task_id"][:8] for c in CASES])
async def test_saved_config_reaches_real_debugger_without_template_expansion(
    case, monkeypatch
):
    import fedotmas.control._controller as module

    config = MAWConfig.model_validate(case["config"])
    original = config.model_dump()
    agent = next(a for a in config.agents if a.name == case["failed_agent"])
    updated_instruction = (
        agent.instruction + '\nPreserve quoted "evidence" and path C:\\data.'
    )

    class Stub(BaseLlm):
        calls: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.calls += 1
            prompt = str(llm_request.config.system_instruction)
            for reference in case["references"]:
                assert reference in prompt
            # Even required references in quoted diagnostics must not be
            # interpolated against the debugger's independent session.
            assert "{missing_required_state}" in prompt
            if self.calls == 1:
                tool = llm_request.tools_dict["repair_agent_configs"]
                schema = tool._get_declaration().model_dump_json()
                assert '"updates"' in schema and "updates_json" not in schema
                part = types.Part(
                    function_call=types.FunctionCall(
                        name="repair_agent_configs",
                        args={
                            "updates": [
                                {
                                    "agent_name": agent.name,
                                    "instruction": updated_instruction,
                                }
                            ],
                            "reasoning": "Offline structural API regression",
                        },
                    )
                )
            else:
                part = types.Part(text="Done")
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    stub = Stub(model="offline")
    monkeypatch.setattr(module, "make_llm", lambda _: stub)
    maw = SimpleNamespace(
        meta_model=ModelConfig(model="offline"),
        _session_service=None,
        mcp_servers={
            name: StdioMCPServer(command="unused", args=())
            for name in [
                "document",
                "sandbox-light",
                "web-scraping",
                "websearch-tavily",
            ]
        },
    )
    updated = await Controller(maw)._run_debugger(
        task="Inspect literal {missing_required_state}",
        error=RunError(agent_name=agent.name, message="incomplete artifact"),
        current_config=config,
        state={"quoted": "{missing_required_state}"},
    )
    assert stub.calls == 2
    assert (
        next(a for a in updated.agents if a.name == agent.name).instruction
        == updated_instruction
    )
    assert config.model_dump() == original


@pytest.mark.asyncio
async def test_structured_repair_keeps_omitted_fields_and_rejects_unknown_fields():
    config = MAWConfig.model_validate(CASES[0]["config"])
    ctx = SimpleNamespace(
        state={
            "config": config.model_dump_json(),
            AVAILABLE_TOOLS_STATE_KEY: ["document", "sandbox-light"],
        }
    )
    before = ctx.state["config"]
    name = config.agents[0].name
    result = await repair_agent_configs(
        ctx, [{"agent_name": name, "model": "other"}], "bad"
    )
    assert result.startswith("Error") and ctx.state["config"] == before
    result = await repair_agent_configs(
        ctx, [{"agent_name": name, "tools": ["missing"]}], "bad"
    )
    assert result.startswith("Error") and ctx.state["config"] == before
    result = await repair_agent_configs(
        ctx, [{"agent_name": name, "instruction": "Read {source?}"}], "good"
    )
    assert result.startswith("Updated")
    expected = config.model_dump()
    expected["agents"][0]["instruction"] = "Read {source?}"
    assert MAWConfig.model_validate_json(ctx.state["config"]).model_dump() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["returned_error", "raised_timeout"])
async def test_recovery_history_survives_later_failure_without_promoting_candidate(
    failure,
):
    config = MAWConfig.model_validate(CASES[0]["config"])
    name = config.agents[-1].name
    output = config.agents[-1].output_key
    initial = ControlledRun(
        config=config,
        status="error",
        state={output: "candidate value"},
        error=RunError(agent_name=name, message="incomplete artifact"),
    )
    final = ControlledRun(
        config=config,
        status="error",
        state={},
        error=RunError(agent_name=name, message="Request timeout"),
    )
    ctrl = Controller(SimpleNamespace())
    ctrl.run = AsyncMock(return_value=initial)
    ctrl._run_debugger = AsyncMock(return_value=config)
    ctrl.resume = (
        AsyncMock(return_value=final)
        if failure == "returned_error"
        else AsyncMock(side_effect=TimeoutError())
    )
    if failure == "raised_timeout":
        with pytest.raises(TimeoutError):
            await ctrl.run_with_recovery("task")
    else:
        result = await ctrl.run_with_recovery("task")
        assert result is final and result.status == "error" and not result.state
    initial.state.clear()
    assert ctrl.recovery_history[0].state[output] == "candidate value"
    ctrl.recovery_history[0].state.clear()
    assert ctrl.recovery_history[0].state[output] == "candidate value"
    ctrl.run = AsyncMock(
        return_value=ControlledRun(config=config, status="success", state={})
    )
    await ctrl.run_with_recovery("new task")
    assert (
        len(ctrl.recovery_history) == 1 and ctrl.recovery_history[0].status == "success"
    )
