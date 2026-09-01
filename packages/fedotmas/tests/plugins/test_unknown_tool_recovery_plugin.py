from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.adk.tools.base_tool import BaseTool

from fedotmas.plugins import UnknownToolRecoveryPlugin


def _unknown_tool_stub(name: str = "exec") -> BaseTool:
    """What ADK hands the callback when it cannot resolve a called name."""
    return BaseTool(name=name, description="Tool not found")


class _RealTool(BaseTool):
    """Stands in for an actual tool: every genuine one subclasses BaseTool."""


def _tool_context(*, session_id: str = "s1", agent_name: str = "researcher"):
    ctx = MagicMock()
    ctx._invocation_context.session.id = session_id
    ctx._invocation_context.agent.name = agent_name
    return ctx


def _not_found(name: str = "exec") -> ValueError:
    return ValueError(f"Tool '{name}' not found.\nAvailable tools: search")


class TestUnknownToolRecoveryPlugin:
    @pytest.mark.asyncio
    async def test_unknown_tool_is_reported_back_to_the_model(self):
        plugin = UnknownToolRecoveryPlugin()

        result = await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={"cmd": ["bash", "-lc", "true"]},
            tool_context=_tool_context(),
            error=_not_found(),
        )

        assert result is not None
        assert "exec" in result["error"]

    @pytest.mark.asyncio
    async def test_a_real_tool_failure_still_propagates(self):
        """Only the unresolved-name case is recovered; other errors re-raise.

        The tool here is a genuine ``BaseTool`` subclass, so this fails if the
        type guard is ever loosened to ``isinstance``.
        """
        plugin = UnknownToolRecoveryPlugin()

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
        plugin = UnknownToolRecoveryPlugin()

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
        plugin = UnknownToolRecoveryPlugin()

        result = await plugin.on_tool_error_callback(
            tool=_RealTool(name="exec", description="Tool not found"),
            tool_args={},
            tool_context=_tool_context(),
            error=_not_found(),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_gives_up_after_the_budget(self):
        """A model that will not correct itself must not loop forever."""
        plugin = UnknownToolRecoveryPlugin(max_recoveries_per_agent=2)
        ctx = _tool_context()

        for _ in range(2):
            assert (
                await plugin.on_tool_error_callback(
                    tool=_unknown_tool_stub(),
                    tool_args={},
                    tool_context=ctx,
                    error=_not_found(),
                )
                is not None
            )

        assert (
            await plugin.on_tool_error_callback(
                tool=_unknown_tool_stub(),
                tool_args={},
                tool_context=ctx,
                error=_not_found(),
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_budget_is_per_agent(self):
        plugin = UnknownToolRecoveryPlugin(max_recoveries_per_agent=1)

        await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={},
            tool_context=_tool_context(agent_name="researcher"),
            error=_not_found(),
        )
        result = await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={},
            tool_context=_tool_context(agent_name="writer"),
            error=_not_found(),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_budget_does_not_leak_across_runs(self):
        """Each run mints a new session, so a spent budget must not carry over."""
        plugin = UnknownToolRecoveryPlugin(max_recoveries_per_agent=1)
        ctx = _tool_context()

        await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={},
            tool_context=ctx,
            error=_not_found(),
        )
        await plugin.before_run_callback(invocation_context=MagicMock())

        result = await plugin.on_tool_error_callback(
            tool=_unknown_tool_stub(),
            tool_args={},
            tool_context=ctx,
            error=_not_found(),
        )

        assert result is not None


class TestRegisteredByDefault:
    def test_present_in_the_default_plugin_set(self):
        """Registering only in the GAIA runner is what made the first fix a no-op.

        This covers ``MAS.run``/``MAW.run`` only.  ``Controller`` and
        ``IterableRun`` assemble their own plugin lists and pick up none of the
        defaults, so the recovery is not active on those paths either.
        """
        from fedotmas import MAW

        maw = MAW()

        assert any(
            isinstance(p, UnknownToolRecoveryPlugin)
            for p in maw._plugins  # noqa: SLF001
        )
