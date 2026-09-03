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


def _ctx(session_id: str = "session-1"):
    """A context that keys the same as a real one: same session, same bucket."""
    ctx = MagicMock()
    ctx.session = MagicMock(id=session_id)
    return ctx


# ---------------------------------------------------------------------------
# MAS auto-injection tests
# ---------------------------------------------------------------------------


class TestAutoAddLoggingPlugin:
    """MAW() auto-injects LoggingPlugin when plugins=None (default)."""

    def test_default_has_logging_plugin(self):
        mas = MAW()
        assert any(isinstance(p, LoggingPlugin) for p in mas._plugins)

    def test_explicit_plugins_respected(self):
        lp = LoggingPlugin()
        mas = MAW(plugins=[lp])
        assert len(mas._plugins) == 1
        assert mas._plugins[0] is lp

    def test_empty_list_disables_defaults(self):
        mas = MAW(plugins=[])
        assert mas._plugins == []


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

        result = await plugin.before_agent_callback(
            agent=agent, callback_context=_ctx("s1")
        )
        assert result is None
        assert ("s1", "researcher") in plugin._agent_start

    @pytest.mark.asyncio
    async def test_skips_workflow_node(self):
        plugin = LoggingPlugin()
        agent = MagicMock()
        agent.name = "seq_1"

        result = await plugin.before_agent_callback(
            agent=agent, callback_context=_ctx("s1")
        )
        assert result is None
        # Still records time even for workflow nodes
        assert ("s1", "seq_1") in plugin._agent_start

    @pytest.mark.asyncio
    async def test_a_sibling_session_keeps_its_own_start_time(self):
        """Keyed by name alone, one run's elapsed was measured from another's."""
        plugin = LoggingPlugin()
        agent = MagicMock()
        agent.name = "researcher"

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx("s1"))
        await plugin.before_agent_callback(agent=agent, callback_context=_ctx("s2"))

        assert ("s1", "researcher") in plugin._agent_start
        assert ("s2", "researcher") in plugin._agent_start


class TestAfterAgentCallback:
    """after_agent_callback computes elapsed, returns None."""

    @pytest.mark.asyncio
    async def test_returns_none(self):
        plugin = LoggingPlugin()
        agent = MagicMock()
        agent.name = "writer"
        ctx = MagicMock()

        # Simulate before → after
        ctx = _ctx("s1")
        await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        result = await plugin.after_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        assert ("s1", "writer") not in plugin._agent_start


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
    async def test_successful_tool_response_with_is_error_false_is_not_warning(
        self, monkeypatch
    ):
        plugin = LoggingPlugin()
        event = FakeEvent(
            _function_responses=[
                FakeFunctionResponse(
                    name="run_command",
                    response={"isError": False, "content": "no error"},
                )
            ]
        )
        inv_ctx = MagicMock()
        warning = MagicMock()
        info = MagicMock()
        monkeypatch.setattr("fedotmas.plugins._logging._log.warning", warning)
        monkeypatch.setattr("fedotmas.plugins._logging._log.info", info)

        result = await plugin.on_event_callback(invocation_context=inv_ctx, event=event)

        assert result is None
        warning.assert_not_called()
        info.assert_any_call("Tool result | agent={} tool={}", "agent", "run_command")

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


@pytest.fixture()
def warnings_logged():
    """Collect loguru WARNING records; ``caplog`` does not see them."""
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(records.append, level="WARNING", format="{message}")
    yield records
    logger.remove(sink_id)


class TestSilentAgentIsReported:
    """An agent that writes nothing must not pass unnoticed."""

    def _agent(self, name="researcher", output_key="research_data"):
        agent = MagicMock()
        agent.name = name
        agent.output_key = output_key
        return agent

    @pytest.mark.asyncio
    async def test_missing_output_warns(self, warnings_logged):
        plugin = LoggingPlugin()
        agent = self._agent()

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx())
        await plugin.after_agent_callback(agent=agent, callback_context=_ctx())

        logged = "".join(warnings_logged)
        assert "No output" in logged
        assert "research_data" in logged

    @pytest.mark.asyncio
    async def test_no_warning_once_state_was_written(self, warnings_logged):
        plugin = LoggingPlugin()
        agent = self._agent()

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx())
        await plugin.on_event_callback(
            invocation_context=_ctx(),
            event=FakeEvent(
                author="researcher",
                actions=FakeActions(state_delta={"research_data": "found it"}),
            ),
        )
        await plugin.after_agent_callback(agent=agent, callback_context=_ctx())

        assert "No output" not in "".join(warnings_logged)

    @pytest.mark.asyncio
    async def test_agent_without_output_key_is_not_reported(self, warnings_logged):
        """A tool-only agent legitimately writes nothing."""
        plugin = LoggingPlugin()
        agent = self._agent(output_key=None)

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx())
        await plugin.after_agent_callback(agent=agent, callback_context=_ctx())

        assert "No output" not in "".join(warnings_logged)

    @pytest.mark.asyncio
    async def test_another_sessions_write_does_not_silence_the_warning(
        self, warnings_logged
    ):
        """Benchmarks drive concurrent sessions through one plugin instance."""
        plugin = LoggingPlugin()
        agent = self._agent()

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx("s1"))
        await plugin.on_event_callback(
            invocation_context=_ctx("s2"),
            event=FakeEvent(
                author="researcher",
                actions=FakeActions(state_delta={"research_data": "found it"}),
            ),
        )
        await plugin.after_agent_callback(agent=agent, callback_context=_ctx("s1"))

        assert "No output" in "".join(warnings_logged)

    @pytest.mark.asyncio
    async def test_an_aborted_run_does_not_leak_its_bookkeeping(self):
        """A timeout or an open circuit never reaches after_agent_callback."""
        plugin = LoggingPlugin()
        agent = self._agent()

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx("s1"))
        # ... run aborts here, no after_agent_callback ...
        await plugin.before_run_callback(invocation_context=_ctx("s1"))

        assert plugin._agent_start == {}
        assert plugin._written_keys == {}

    @pytest.mark.asyncio
    async def test_pruning_leaves_a_concurrent_run_alone(self):
        plugin = LoggingPlugin()
        agent = self._agent()

        await plugin.before_agent_callback(agent=agent, callback_context=_ctx("s1"))
        await plugin.before_run_callback(invocation_context=_ctx("s2"))

        assert ("s1", "researcher") in plugin._agent_start
