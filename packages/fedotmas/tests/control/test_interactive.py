"""Tests for InteractiveRun and run_interactive context manager."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from fedotmas.control._controller import Controller
from fedotmas.control._interactive import InteractiveRun, _PausePlugin
from fedotmas.core.runner import PipelineResult
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig


def _agent(name: str) -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=f"Do {name}", output_key=name)


def _config(*names: str) -> MAWConfig:
    agents = [_agent(n) for n in names]
    pipeline = MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name=n) for n in names],
    )
    return MAWConfig(agents=agents, pipeline=pipeline)


def _mock_maw() -> MagicMock:
    maw = MagicMock()
    maw._session_service = None
    maw._memory_service = None
    maw.build = MagicMock(return_value=MagicMock())
    return maw


class _FakeState(dict):
    def to_dict(self) -> dict:
        return dict(self)


def _fake_run_pipeline(agent_names: list[str], final_state: dict):
    """Return an async function that simulates pipeline execution with plugin callbacks."""

    async def _run(_agent, _task, *, plugins=None, **_kwargs):
        state = _FakeState(_kwargs.get("initial_state") or {})
        for name in agent_names:
            fake_agent = MagicMock()
            fake_agent.name = name
            fake_ctx = MagicMock()
            fake_ctx.state = state

            for plugin in plugins or []:
                if hasattr(plugin, "before_agent_callback"):
                    await plugin.before_agent_callback(
                        agent=fake_agent, callback_context=fake_ctx
                    )

            state[name] = f"done_{name}"

            for plugin in plugins or []:
                if hasattr(plugin, "after_agent_callback"):
                    await plugin.after_agent_callback(
                        agent=fake_agent, callback_context=fake_ctx
                    )

        return PipelineResult(state=dict(final_state))

    return _run


class TestPausePlugin:
    @pytest.mark.asyncio
    async def test_pauses_at_target(self):
        plugin = _PausePlugin()
        plugin._target = "writer"

        agent = MagicMock()
        agent.name = "writer"
        ctx = MagicMock()

        reached = False

        async def run_plugin():
            nonlocal reached
            await plugin.before_agent_callback(agent=agent, callback_context=ctx)
            reached = True

        task = asyncio.create_task(run_plugin())
        await asyncio.sleep(0)
        assert plugin._reached.is_set()
        assert not reached

        plugin._resume.set()
        await task
        assert reached

    @pytest.mark.asyncio
    async def test_passes_non_target(self):
        plugin = _PausePlugin()
        plugin._target = "writer"

        agent = MagicMock()
        agent.name = "reader"
        ctx = MagicMock()

        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        assert not plugin._reached.is_set()

    @pytest.mark.asyncio
    async def test_skips_workflow_nodes(self):
        plugin = _PausePlugin()
        plugin._target = "seq_main"

        agent = MagicMock()
        agent.name = "seq_main"
        ctx = MagicMock()

        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None


class TestInteractiveRun:
    @pytest.mark.asyncio
    async def test_wait_until_and_continue(self):
        maw = _mock_maw()
        config = _config("a", "b")

        fake_run = _fake_run_pipeline(["a", "b"], {"a": "done_a", "b": "done_b"})

        with patch("fedotmas.control._interactive.run_pipeline", side_effect=fake_run):
            async with Controller(maw).run_interactive(config, "task") as run:
                await run.wait_until("b")
                assert "a" in run.state
                result = await run.continue_()

            assert result.status == "success"
            assert result.state == {"a": "done_a", "b": "done_b"}

    @pytest.mark.asyncio
    async def test_multi_pause(self):
        maw = _mock_maw()
        config = _config("a", "b", "c")

        fake_run = _fake_run_pipeline(
            ["a", "b", "c"], {"a": "done_a", "b": "done_b", "c": "done_c"}
        )

        with patch("fedotmas.control._interactive.run_pipeline", side_effect=fake_run):
            async with Controller(maw).run_interactive(config, "task") as run:
                await run.wait_until("b")
                assert "a" in run.state
                assert "b" not in run.state

                await run.wait_until("c")
                assert "b" in run.state
                assert "c" not in run.state

                result = await run.continue_()

            assert result.status == "success"

    @pytest.mark.asyncio
    async def test_state_empty_before_execution(self):
        run = InteractiveRun(_mock_maw(), _config("a"), "task")
        assert run.state == {}

    @pytest.mark.asyncio
    async def test_continue_without_wait_raises(self):
        run = InteractiveRun(_mock_maw(), _config("a"), "task")
        with pytest.raises(RuntimeError, match="No execution"):
            await run.continue_()

    @pytest.mark.asyncio
    async def test_cleanup_releases_paused_pipeline(self):
        maw = _mock_maw()
        config = _config("a", "b")

        fake_run = _fake_run_pipeline(["a", "b"], {"a": "ok", "b": "ok"})

        with patch("fedotmas.control._interactive.run_pipeline", side_effect=fake_run):
            async with Controller(maw).run_interactive(config, "task") as run:
                await run.wait_until("b")

            assert run._exec_task is not None
            assert run._exec_task.done()

    @pytest.mark.asyncio
    async def test_error_during_pipeline(self):
        maw = _mock_maw()
        config = _config("a", "b")

        async def failing_run(_agent, _task, **_kwargs):
            raise RuntimeError("Agent 'b' failed with error 500: boom")

        with patch(
            "fedotmas.control._interactive.run_pipeline", side_effect=failing_run
        ):
            async with Controller(maw).run_interactive(config, "task") as run:
                run._exec_task = asyncio.create_task(run._run())
                result = await run.continue_()

            assert result.status == "error"
            assert result.error is not None
            assert result.error.agent_name == "b"

    @pytest.mark.asyncio
    async def test_checkpoints_created(self):
        maw = _mock_maw()
        config = _config("a", "b")

        fake_run = _fake_run_pipeline(["a", "b"], {"a": "done_a", "b": "done_b"})

        with patch("fedotmas.control._interactive.run_pipeline", side_effect=fake_run):
            async with Controller(maw).run_interactive(config, "task") as run:
                await run.wait_until("b")
                assert len(run.checkpoints) == 1
                assert run.checkpoints[0].agent_name == "a"

                result = await run.continue_()

            assert len(result.checkpoints) == 2


class TestRunWithRecovery:
    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        maw = _mock_maw()
        ctrl = Controller(maw)
        with pytest.raises(NotImplementedError, match="meta-debugger"):
            await ctrl.run_with_recovery("task")
