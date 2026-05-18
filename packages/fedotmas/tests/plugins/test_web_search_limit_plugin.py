from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fedotmas.maw.maw import MAW
from fedotmas.plugins import WebSearchLimitExceeded, WebSearchLimitPlugin


def _tool(name: str, description: str = "") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


def _tool_context(*, session_id: str = "s1", agent_name: str = "researcher"):
    ctx = MagicMock()
    ctx._invocation_context.session.id = session_id
    ctx._invocation_context.agent.name = agent_name
    return ctx


def _invocation_context(session_id: str = "s1"):
    ctx = MagicMock()
    ctx.session.id = session_id
    return ctx


class TestWebSearchLimitPlugin:
    @pytest.mark.asyncio
    async def test_allows_until_limit_then_blocks(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=2)
        tool = _tool("search", "Search the web")
        ctx = _tool_context()

        first = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "a"}, tool_context=ctx
        )
        second = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "b"}, tool_context=ctx
        )
        third = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "c"}, tool_context=ctx
        )

        assert first is None
        assert second is None
        assert third is not None
        assert third["isError"] is True
        assert "max 2 calls" in third["error"]

    @pytest.mark.asyncio
    async def test_hard_fail_raises_when_limit_exceeded(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1, hard_fail=True)
        tool = _tool("search", "Search the web")
        ctx = _tool_context()

        await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "a"}, tool_context=ctx
        )

        with pytest.raises(WebSearchLimitExceeded, match="max 1 calls"):
            await plugin.before_tool_callback(
                tool=tool, tool_args={"query": "b"}, tool_context=ctx
            )

    @pytest.mark.asyncio
    async def test_counts_are_per_agent(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("web_search")

        await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=_tool_context(agent_name="a")
        )
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=_tool_context(agent_name="b")
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_plain_non_web_search_tool(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("search", "Search local memory")
        ctx = _tool_context()

        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_before_run_resets_current_session_counts(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("search", "Search the internet")
        ctx = _tool_context(session_id="s1")

        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
        await plugin.before_run_callback(invocation_context=_invocation_context("s1"))
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )

        assert result is None


class TestAutoAddWebSearchLimitPlugin:
    def test_default_has_web_search_limit_plugin(self):
        maw = MAW()
        plugin = next(p for p in maw._plugins if isinstance(p, WebSearchLimitPlugin))
        assert plugin.max_calls_per_agent == 4

    def test_can_disable_default_web_search_limit_plugin(self):
        maw = MAW(web_search_limit=None)
        assert not any(isinstance(p, WebSearchLimitPlugin) for p in maw._plugins)
