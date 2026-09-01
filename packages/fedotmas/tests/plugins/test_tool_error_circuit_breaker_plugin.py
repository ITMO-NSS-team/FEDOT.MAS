from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from google.adk.tools.base_tool import BaseTool

from fedotmas.plugins import ToolErrorCircuitBreakerPlugin, ToolErrorCircuitOpen


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


def _unknown_tool_stub(name: str = "exec") -> BaseTool:
    """What ADK hands the callback when it cannot resolve a called name."""
    return BaseTool(name=name, description="Tool not found")


class _RealTool(BaseTool):
    """Stands in for an actual tool: every genuine one subclasses BaseTool."""


class TestHallucinatedToolName:
    """An invented tool name must not discard the whole pipeline's work."""

    @pytest.mark.asyncio
    async def test_unknown_tool_is_reported_back_to_the_model(self):
        plugin = ToolErrorCircuitBreakerPlugin()

        result = await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={"cmd": ["bash", "-lc", "true"]},
            tool_context=_tool_context(),
            error=ValueError("Tool 'exec' not found.\nAvailable tools: "),
        )

        assert result is not None
        assert "exec" in result["error"]

    @pytest.mark.asyncio
    async def test_a_real_tool_failure_still_propagates(self):
        """Only the unresolved-name case is recovered; other errors re-raise.

        The tool here is a genuine ``BaseTool`` subclass, so this fails if the
        type guard is ever loosened to ``isinstance``.
        """
        plugin = ToolErrorCircuitBreakerPlugin()

        result = await plugin.on_tool_error_callback(
            tool=_RealTool(name="run_code", description="Run code"),
            tool_args={},
            tool_context=_tool_context(),
            error=ValueError("sandbox refused the connection"),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_a_real_tool_reporting_a_missing_file_is_not_recovered(self):
        """The message guard is anchored, so 'not found' alone is not enough."""
        plugin = ToolErrorCircuitBreakerPlugin()

        result = await plugin.on_tool_error_callback(
            tool=_RealTool(name="read_document", description="Read a document"),
            tool_args={},
            tool_context=_tool_context(),
            error=ValueError("File '/tmp/report.pdf' not found."),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_recognised_by_message_when_the_stub_type_changes(self):
        """If ADK stops passing a bare BaseTool, the message still identifies it."""
        plugin = ToolErrorCircuitBreakerPlugin()

        result = await plugin.on_tool_error_callback(
            tool=_RealTool(name="exec", description="Tool not found"),
            tool_args={},
            tool_context=_tool_context(),
            error=ValueError("Tool 'exec' not found.\nAvailable tools: "),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_repeats_still_trip_the_circuit(self):
        """Recovering must not turn a loop on one bad name into a free pass."""
        plugin = ToolErrorCircuitBreakerPlugin(max_same_tool_error_type=2)
        ctx = _tool_context()
        error = ValueError("Tool 'exec' not found.")

        await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={},
            tool_context=ctx,
            error=error,
        )

        with pytest.raises(ToolErrorCircuitOpen):
            await plugin.on_tool_error_callback(
                tool=_unknown_tool_stub(),
                tool_args={},
                tool_context=ctx,
                error=error,
            )
