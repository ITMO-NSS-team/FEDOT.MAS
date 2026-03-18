"""Tests for CheckpointPlugin — state snapshots at agent boundaries."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fedotmas.plugins import Checkpoint, CheckpointPlugin


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _make_ctx(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state
    return ctx


class TestAfterAgentCallback:
    @pytest.mark.asyncio
    async def test_creates_checkpoint(self):
        plugin = CheckpointPlugin()
        agent = _make_agent("analyst")
        ctx = _make_ctx({"user_query": "hi", "result": "done"})

        result = await plugin.after_agent_callback(agent=agent, callback_context=ctx)

        assert result is None
        assert len(plugin.checkpoints) == 1
        cp = plugin.checkpoints[0]
        assert cp.agent_name == "analyst"
        assert cp.state == {"user_query": "hi", "result": "done"}
        assert cp.index == 0

    @pytest.mark.asyncio
    async def test_skips_workflow_nodes(self):
        plugin = CheckpointPlugin()
        ctx = _make_ctx({})

        for prefix in ("seq_main", "par_branch", "loop_retry"):
            await plugin.after_agent_callback(
                agent=_make_agent(prefix), callback_context=ctx
            )

        assert plugin.checkpoints == []

    @pytest.mark.asyncio
    async def test_sequential_indexing(self):
        plugin = CheckpointPlugin()

        for i, name in enumerate(["reader", "writer", "reviewer"]):
            ctx = _make_ctx({"step": name})
            await plugin.after_agent_callback(
                agent=_make_agent(name), callback_context=ctx
            )

        assert len(plugin.checkpoints) == 3
        assert [cp.index for cp in plugin.checkpoints] == [0, 1, 2]
        assert [cp.agent_name for cp in plugin.checkpoints] == [
            "reader", "writer", "reviewer",
        ]

    @pytest.mark.asyncio
    async def test_state_is_copied(self):
        """Mutation of original state dict must not affect checkpoint."""
        plugin = CheckpointPlugin()
        state = {"key": "original"}
        ctx = _make_ctx(state)

        await plugin.after_agent_callback(
            agent=_make_agent("a"), callback_context=ctx
        )
        state["key"] = "mutated"

        assert plugin.checkpoints[0].state["key"] == "original"


class TestLookup:
    @pytest.mark.asyncio
    async def test_get_returns_last(self):
        plugin = CheckpointPlugin()
        ctx1 = _make_ctx({"v": 1})
        ctx2 = _make_ctx({"v": 2})

        await plugin.after_agent_callback(
            agent=_make_agent("a"), callback_context=ctx1
        )
        await plugin.after_agent_callback(
            agent=_make_agent("a"), callback_context=ctx2
        )

        cp = plugin.get("a")
        assert cp is not None
        assert cp.state["v"] == 2

    def test_get_missing_returns_none(self):
        plugin = CheckpointPlugin()
        assert plugin.get("nope") is None

    @pytest.mark.asyncio
    async def test_state_at(self):
        plugin = CheckpointPlugin()
        ctx = _make_ctx({"x": 42})

        await plugin.after_agent_callback(
            agent=_make_agent("b"), callback_context=ctx
        )

        assert plugin.state_at("b") == {"x": 42}

    def test_state_at_missing_returns_none(self):
        plugin = CheckpointPlugin()
        assert plugin.state_at("nope") is None


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_empties_checkpoints(self):
        plugin = CheckpointPlugin()
        ctx = _make_ctx({"k": "v"})

        await plugin.after_agent_callback(
            agent=_make_agent("a"), callback_context=ctx
        )
        assert len(plugin.checkpoints) == 1

        plugin.clear()
        assert plugin.checkpoints == []


class TestCheckpointImmutability:
    def test_checkpoints_property_returns_copy(self):
        plugin = CheckpointPlugin()
        cps = plugin.checkpoints
        cps.append(Checkpoint(agent_name="fake", state={}, index=99))
        assert len(plugin.checkpoints) == 0
