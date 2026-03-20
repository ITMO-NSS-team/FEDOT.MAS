"""Tests for SkipCompletedPlugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fedotmas.plugins._skip_completed import SkipCompletedPlugin


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
