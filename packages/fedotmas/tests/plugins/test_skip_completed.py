"""Tests for SkipCompletedPlugin."""

from __future__ import annotations

import pytest

from fedotmas.plugins._skip_completed import SkipCompletedPlugin


@pytest.fixture
def plugin():
    return SkipCompletedPlugin(completed_agents={"reader", "writer"})


@pytest.mark.asyncio
async def test_skips_completed_agent(plugin):
    result = await plugin.before_agent("reader", {})

    assert result is not None
    assert "skipped" in result["text"]


@pytest.mark.asyncio
async def test_does_not_skip_uncompleted_agent(plugin):
    result = await plugin.before_agent("reviewer", {})

    assert result is None


@pytest.mark.asyncio
async def test_ignores_workflow_nodes(plugin):
    """Workflow nodes should not be skipped even if named similarly."""
    # SkipCompletedPlugin only checks exact membership in completed set
    for name in ("seq_1", "par_2", "loop_3"):
        result = await plugin.before_agent(name, {})
        assert result is None
