from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fedotmas.plugins import ToolErrorCircuitBreakerPlugin, ToolErrorCircuitOpen
from fedotmas.plugins._tool_error_circuit_breaker import WEB_BUDGET_EXHAUSTED


def _tool(name: str = "markdown") -> MagicMock:
    tool = MagicMock()
    tool.name = name
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


class TestToolErrorCircuitBreakerPlugin:
    @pytest.mark.asyncio
    async def test_ignores_successful_tool_result(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        result = await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=_tool_context(),
            result={"content": "ok"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_opens_on_repeated_same_tool_error_type(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_same_tool_error_type=2)
        tool = _tool("goto")
        ctx = _tool_context()

        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=ctx,
            result={"isError": True, "error": "OperationTimedout: timeout"},
        )

        with pytest.raises(ToolErrorCircuitOpen, match="OperationTimedout"):
            await plugin.after_tool_callback(
                tool=tool,
                tool_args={},
                tool_context=ctx,
                result={"isError": True, "error": "OperationTimedout: timeout"},
            )

    @pytest.mark.asyncio
    async def test_opens_on_total_agent_tool_errors(self):
        plugin = ToolErrorCircuitBreakerPlugin(
            max_errors_per_agent=2,
            max_same_tool_error_type=99,
        )
        ctx = _tool_context()

        await plugin.after_tool_callback(
            tool=_tool("goto"),
            tool_args={},
            tool_context=ctx,
            result={"isError": True, "error": "OperationTimedout: timeout"},
        )

        with pytest.raises(ToolErrorCircuitOpen, match="2 tool errors"):
            await plugin.on_tool_error_callback(
                tool=_tool("markdown"),
                tool_args={},
                tool_context=ctx,
                error=RuntimeError("CouldntResolveHost"),
            )

    @pytest.mark.asyncio
    async def test_counts_are_per_agent(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=2)

        await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=_tool_context(agent_name="a"),
            result={"isError": True, "error": "OperationTimedout"},
        )
        result = await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=_tool_context(agent_name="b"),
            result={"isError": True, "error": "OperationTimedout"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_before_run_resets_current_session_counts(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=2)
        ctx = _tool_context(session_id="s1")

        await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=ctx,
            result={"isError": True, "error": "OperationTimedout"},
        )
        await plugin.before_run_callback(invocation_context=_invocation_context("s1"))

        result = await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=ctx,
            result={"isError": True, "error": "OperationTimedout"},
        )

        assert result is None


class TestRescuedAndControlFlowResults:
    """A server that answered a failed call with a usable substitute."""

    @pytest.mark.asyncio
    async def test_a_rescued_error_does_not_count(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        for _ in range(5):
            result = await plugin.after_tool_callback(
                tool=_tool("extract"),
                tool_args={},
                tool_context=_tool_context(),
                result={
                    "isError": True,
                    "meta": {"fedotmas/rescued": True},
                    "content": [{"type": "text", "text": "# Page"}],
                },
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_repeated_budget_blocks_do_not_count_as_tool_failures(self):
        plugin = ToolErrorCircuitBreakerPlugin(
            max_errors_per_agent=1,
            max_same_tool_error_type=1,
        )

        for _ in range(5):
            result = await plugin.after_tool_callback(
                tool=_tool("search"),
                tool_args={},
                tool_context=_tool_context(),
                result={
                    "isError": True,
                    "error": "Stop searching and finish from current evidence.",
                    "error_code": WEB_BUDGET_EXHAUSTED,
                },
            )

            assert result is None

        assert plugin._total_errors == {}
        assert plugin._pattern_errors == {}

    @pytest.mark.asyncio
    async def test_message_text_alone_does_not_trigger_control_flow_exemption(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        with pytest.raises(ToolErrorCircuitOpen):
            await plugin.after_tool_callback(
                tool=_tool("search"),
                tool_args={},
                tool_context=_tool_context(),
                result={
                    "isError": True,
                    "error": f"{WEB_BUDGET_EXHAUSTED}: unrelated tool failure",
                },
            )

    @pytest.mark.asyncio
    async def test_an_unrescued_error_still_counts(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        with pytest.raises(ToolErrorCircuitOpen):
            await plugin.after_tool_callback(
                tool=_tool("extract"),
                tool_args={},
                tool_context=_tool_context(),
                result={"isError": True, "meta": {}, "content": []},
            )

    @pytest.mark.asyncio
    async def test_a_falsy_marker_does_not_excuse_the_error(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        with pytest.raises(ToolErrorCircuitOpen):
            await plugin.after_tool_callback(
                tool=_tool("extract"),
                tool_args={},
                tool_context=_tool_context(),
                result={"isError": True, "meta": {"fedotmas/rescued": False}},
            )
