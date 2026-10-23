"""Tests for SkipCompletedPlugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fedotmas.core.runner import run_pipeline
from fedotmas.plugins._skip_completed import SkipCompletedPlugin
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel


@pytest.fixture
def plugin():
    return SkipCompletedPlugin(completed_agents={"reader", "writer"})


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


@pytest.mark.asyncio
async def test_skips_completed_agent(plugin):
    agent = _make_agent("reader")
    ctx = MagicMock()

    result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)

    assert result is not None
    assert result.role == "model"


@pytest.mark.asyncio
async def test_does_not_skip_uncompleted_agent(plugin):
    agent = _make_agent("reviewer")
    ctx = MagicMock()

    result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)

    assert result is None


@pytest.mark.asyncio
async def test_ignores_workflow_nodes(plugin):
    for prefix in ("seq_1", "par_2", "loop_3"):
        agent = _make_agent(prefix)
        ctx = MagicMock()
        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None


@pytest.mark.parametrize("value", ['{"entity":"A"}', {"entity": "A"}])
@pytest.mark.parametrize("schema", [False, True])
async def test_real_adk_skip_preserves_checkpoint_output(value, schema):
    class Output(BaseModel):
        entity: str

    class NeverCalled(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            raise AssertionError("Completed agent must not call its model")
            yield

    agent = LlmAgent(
        name="reader",
        model=NeverCalled(model="offline"),
        output_key="artifact",
        output_schema=Output if schema else None,
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("Network forbidden")
    ):
        result = await run_pipeline(
            agent,
            "Continue",
            initial_state={"artifact": value},
            plugins=[SkipCompletedPlugin({"reader"})],
        )
    assert result.status == "completed"
    assert result.state["artifact"] == value
