"""Tests for CheckpointPlugin — state snapshots at agent boundaries."""

from __future__ import annotations

import pytest

from fedotmas.plugins import Checkpoint, CheckpointPlugin


class TestAfterAgent:
    @pytest.mark.asyncio
    async def test_creates_checkpoint(self):
        plugin = CheckpointPlugin()
        state = {"user_query": "hi", "result": "done"}

        await plugin.after_agent("analyst", state)

        assert len(plugin.checkpoints) == 1
        cp = plugin.checkpoints[0]
        assert cp.agent_name == "analyst"
        assert cp.state == {"user_query": "hi", "result": "done"}
        assert cp.index == 0

    @pytest.mark.asyncio
    async def test_skips_workflow_nodes(self):
        plugin = CheckpointPlugin()
        state = {}

        for prefix in ("seq_main", "par_branch", "loop_retry"):
            await plugin.after_agent(prefix, state)

        assert plugin.checkpoints == []

    @pytest.mark.asyncio
    async def test_sequential_indexing(self):
        plugin = CheckpointPlugin()

        for i, name in enumerate(["reader", "writer", "reviewer"]):
            await plugin.after_agent(name, {"step": name})

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

        await plugin.after_agent("a", state)
        state["key"] = "mutated"

        assert plugin.checkpoints[0].state["key"] == "original"


class TestBeforeAgent:
    @pytest.mark.asyncio
    async def test_returns_none(self):
        plugin = CheckpointPlugin()
        result = await plugin.before_agent("a", {})
        assert result is None


class TestLookup:
    @pytest.mark.asyncio
    async def test_get_returns_last(self):
        plugin = CheckpointPlugin()

        await plugin.after_agent("a", {"v": 1})
        await plugin.after_agent("a", {"v": 2})

        cp = plugin.get("a")
        assert cp is not None
        assert cp.state["v"] == 2

    def test_get_missing_returns_none(self):
        plugin = CheckpointPlugin()
        assert plugin.get("nope") is None

    @pytest.mark.asyncio
    async def test_state_at(self):
        plugin = CheckpointPlugin()

        await plugin.after_agent("b", {"x": 42})

        assert plugin.state_at("b") == {"x": 42}

    def test_state_at_missing_returns_none(self):
        plugin = CheckpointPlugin()
        assert plugin.state_at("nope") is None


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_empties_checkpoints(self):
        plugin = CheckpointPlugin()

        await plugin.after_agent("a", {"k": "v"})
        assert len(plugin.checkpoints) == 1

        plugin.clear()
        assert plugin.checkpoints == []


class TestCheckpointImmutability:
    def test_checkpoints_property_returns_copy(self):
        plugin = CheckpointPlugin()
        cps = plugin.checkpoints
        cps.append(Checkpoint(agent_name="fake", state={}, index=99))
        assert len(plugin.checkpoints) == 0
