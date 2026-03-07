"""Tests for LoggingPlugin and MAS auto-injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from google.adk.plugins import BasePlugin

from fedotmas.maw.maw import MAW
from fedotmas.plugins import LoggingPlugin


# ---------------------------------------------------------------------------
# Fake event helpers (subset of pipeline/conftest)
# ---------------------------------------------------------------------------


@dataclass
class FakeActions:
    state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None


@dataclass
class FakeFunctionCall:
    name: str = "tool_x"
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeFunctionResponse:
    name: str = "tool_x"
    response: Any = None


@dataclass
class FakeEvent:
    partial: bool = False
    author: str = "agent"
    content: Any = None
    error_code: str | None = None
    error_message: str | None = None
    usage_metadata: FakeUsageMetadata | None = None
    actions: FakeActions = field(default_factory=FakeActions)
    _function_calls: list[FakeFunctionCall] = field(default_factory=list)
    _function_responses: list[FakeFunctionResponse] = field(default_factory=list)

    def get_function_calls(self) -> list:
        return self._function_calls

    def get_function_responses(self) -> list:
        return self._function_responses


# ---------------------------------------------------------------------------
# MAS auto-injection tests
# ---------------------------------------------------------------------------


class TestAutoAddLoggingPlugin:
    """MAW() auto-injects LoggingPlugin."""

    def test_default_has_logging_plugin(self):
        mas = MAW()
        assert any(isinstance(p, LoggingPlugin) for p in mas._plugins)

    def test_no_duplicate(self):
        lp = LoggingPlugin()
        mas = MAW(plugins=[lp])
        logging_plugins = [p for p in mas._plugins if isinstance(p, LoggingPlugin)]
        assert len(logging_plugins) == 1
        assert logging_plugins[0] is lp

    def test_ordering_logging_first(self):
        class MockPlugin(BasePlugin):
            def __init__(self):
                super().__init__(name="mock")

        mp = MockPlugin()
        mas = MAW(plugins=[mp])
        assert isinstance(mas._plugins[0], LoggingPlugin)
        assert mas._plugins[1] is mp


# ---------------------------------------------------------------------------
# Plugin callback tests
# ---------------------------------------------------------------------------


class TestBeforeAgentCallback:
    """before_agent_callback logs agent starts, skips workflow nodes."""

    @pytest.mark.asyncio
    async def test_records_start_time(self):
        plugin = LoggingPlugin()
        agent = MagicMock()
        agent.name = "researcher"
        ctx = MagicMock()

        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        assert "researcher" in plugin._agent_start

    @pytest.mark.asyncio
    async def test_skips_workflow_node(self):
        plugin = LoggingPlugin()
        agent = MagicMock()
        agent.name = "seq_1"
        ctx = MagicMock()

        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        # Still records time even for workflow nodes
        assert "seq_1" in plugin._agent_start


class TestAfterAgentCallback:
    """after_agent_callback computes elapsed, returns None."""

    @pytest.mark.asyncio
    async def test_returns_none(self):
        plugin = LoggingPlugin()
        agent = MagicMock()
        agent.name = "writer"
        ctx = MagicMock()

        # Simulate before → after
        await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        result = await plugin.after_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        assert "writer" not in plugin._agent_start


class TestOnEventCallback:
    """on_event_callback logs event details, returns None."""

    @pytest.mark.asyncio
    async def test_partial_skipped(self):
        plugin = LoggingPlugin()
        event = FakeEvent(partial=True)
        inv_ctx = MagicMock()

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_call_logged(self):
        plugin = LoggingPlugin()
        event = FakeEvent(
            _function_calls=[FakeFunctionCall(name="search", args={"q": "test"})]
        )
        inv_ctx = MagicMock()

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)
        assert result is None

    @pytest.mark.asyncio
    async def test_token_usage_logged(self):
        plugin = LoggingPlugin()
        event = FakeEvent(
            usage_metadata=FakeUsageMetadata(
                prompt_token_count=100, candidates_token_count=50
            )
        )
        inv_ctx = MagicMock()

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)
        assert result is None

    @pytest.mark.asyncio
    async def test_state_delta_logged(self):
        plugin = LoggingPlugin()
        event = FakeEvent(actions=FakeActions(state_delta={"result": "done"}))
        inv_ctx = MagicMock()

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_state_value_no_crash(self):
        plugin = LoggingPlugin()
        event = FakeEvent(actions=FakeActions(state_delta={"key": None}))
        inv_ctx = MagicMock()

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_error_logged(self):
        plugin = LoggingPlugin()
        event = FakeEvent(
            _function_responses=[
                FakeFunctionResponse(name="tool_x", response={"error": "not found"})
            ]
        )
        inv_ctx = MagicMock()

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)
        assert result is None
