from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fedotmas.plugins import ToolErrorCircuitBreakerPlugin
from fedotmas.plugins._tool_error_circuit_breaker import (
    DUPLICATE_TOOL_CALL,
    TOOL_CIRCUIT_OPEN,
    WEB_BUDGET_EXHAUSTED,
)


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
    async def test_browser_mcp_error_opens_circuit_with_machine_code(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_same_tool_error_type=2)
        tool = _tool("complete_browser_task")
        result = {
            "is_error": True,
            "structured_content": {
                "status": "failed",
                "error_code": "BROWSER_AGENT_FAILED",
            },
        }
        for _ in range(2):
            await plugin.after_tool_callback(
                tool=tool, tool_args={}, tool_context=_tool_context(), result=result
            )

        blocked = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=_tool_context()
        )
        assert blocked["error_code"] == TOOL_CIRCUIT_OPEN
        assert "BROWSER_AGENT_FAILED" in blocked["error"]

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

        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=ctx,
            result={"isError": True, "error": "OperationTimedout: timeout"},
        )

        control = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )
        assert control["error_code"] == TOOL_CIRCUIT_OPEN
        assert "OperationTimedout" in control["error"]
        assert set(plugin._open_circuits) == {("s1", "researcher", "goto")}

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

        await plugin.on_tool_error_callback(
            tool=_tool("markdown"),
            tool_args={},
            tool_context=ctx,
            error=RuntimeError("CouldntResolveHost"),
        )
        control = await plugin.before_tool_callback(
            tool=_tool("markdown"), tool_args={}, tool_context=ctx
        )
        assert control["error_code"] == TOOL_CIRCUIT_OPEN
        assert "2 tool failures" in control["error"]

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
    @pytest.mark.asyncio
    async def test_duplicate_control_result_does_not_count(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)
        for _ in range(3):
            await plugin.after_tool_callback(
                tool=_tool("search"),
                tool_args={},
                tool_context=_tool_context(),
                result={
                    "error_code": DUPLICATE_TOOL_CALL,
                    "isError": True,
                    "error": "duplicate",
                },
            )
        assert plugin._total_errors == {}

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

        await plugin.after_tool_callback(
            tool=_tool("search"),
            tool_args={},
            tool_context=_tool_context(),
            result={
                "isError": True,
                "error": f"{WEB_BUDGET_EXHAUSTED}: unrelated tool failure",
            },
        )
        assert plugin._open_circuits

    @pytest.mark.asyncio
    async def test_an_unrescued_error_still_counts(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        await plugin.after_tool_callback(
            tool=_tool("extract"),
            tool_args={},
            tool_context=_tool_context(),
            result={"isError": True, "meta": {}, "content": []},
        )
        assert plugin._open_circuits

    @pytest.mark.asyncio
    async def test_structured_backend_error_counts(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)
        await plugin.after_tool_callback(
            tool=_tool("search"),
            tool_args={},
            tool_context=_tool_context(),
            result={"error": {"type": "BackendUnavailable", "message": "down"}},
        )
        assert plugin._open_circuits

    @pytest.mark.asyncio
    async def test_a_falsy_marker_does_not_excuse_the_error(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_errors_per_agent=1)

        await plugin.after_tool_callback(
            tool=_tool("extract"),
            tool_args={},
            tool_context=_tool_context(),
            result={"isError": True, "meta": {"fedotmas/rescued": False}},
        )
        assert plugin._open_circuits

    @pytest.mark.asyncio
    async def test_open_circuit_is_scoped_to_agent_and_tool_and_ignores_control(self):
        plugin = ToolErrorCircuitBreakerPlugin(max_same_tool_error_type=1)
        tool = _tool("search")
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=_tool_context(agent_name="researcher"),
            result={"isError": True, "error": "BackendUnavailable: down"},
        )

        blocked = await plugin.before_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=_tool_context(agent_name="researcher"),
        )
        sibling_agent = await plugin.before_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=_tool_context(agent_name="sibling"),
        )
        sibling_tool = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={},
            tool_context=_tool_context(agent_name="researcher"),
        )
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=_tool_context(agent_name="researcher"),
            result=blocked,
        )

        assert blocked["error_code"] == TOOL_CIRCUIT_OPEN
        assert sibling_agent is None
        assert sibling_tool is None
        assert plugin._total_errors[("s1", "researcher")] == 1
        assert (
            plugin._pattern_errors[("s1", "researcher", "search", "BackendUnavailable")]
            == 1
        )
