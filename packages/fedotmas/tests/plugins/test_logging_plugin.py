"""Tests for LoggingPlugin middleware."""

from __future__ import annotations

import pytest

from fedotmas.maw.maw import MAW
from fedotmas.plugins import LoggingPlugin


# ---------------------------------------------------------------------------
# MAS auto-injection tests
# ---------------------------------------------------------------------------


class TestAutoAddLoggingPlugin:
    """MAW() auto-injects LoggingPlugin when plugins=None (default)."""

    def test_default_has_logging_plugin(self):
        mas = MAW()
        assert any(isinstance(p, LoggingPlugin) for p in mas._middlewares)

    def test_explicit_middlewares_respected(self):
        lp = LoggingPlugin()
        mas = MAW(middlewares=[lp])
        assert len(mas._middlewares) == 1
        assert mas._middlewares[0] is lp

    def test_empty_list_disables_defaults(self):
        mas = MAW(middlewares=[])
        assert mas._middlewares == []


# ---------------------------------------------------------------------------
# Middleware callback tests
# ---------------------------------------------------------------------------


class TestBeforeAgent:
    """before_agent logs agent starts, records start time."""

    @pytest.mark.asyncio
    async def test_records_start_time(self):
        plugin = LoggingPlugin()

        result = await plugin.before_agent("researcher", {})
        assert result is None
        assert "researcher" in plugin._agent_start

    @pytest.mark.asyncio
    async def test_records_workflow_node(self):
        plugin = LoggingPlugin()

        result = await plugin.before_agent("seq_1", {})
        assert result is None
        assert "seq_1" in plugin._agent_start


class TestAfterAgent:
    """after_agent computes elapsed, returns None."""

    @pytest.mark.asyncio
    async def test_returns_none(self):
        plugin = LoggingPlugin()

        await plugin.before_agent("writer", {})
        result = await plugin.after_agent("writer", {})
        assert result is None
        assert "writer" not in plugin._agent_start
