"""Tests for IterableRun and Controller.iter context manager."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from fedotmas.control._controller import Controller
from fedotmas.control._iterable import IterableRun, _StepPlugin
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


def _mock_maw(*agent_names: str) -> MagicMock:
    maw = MagicMock()
    maw._session_service = None
    maw._memory_service = None
    root = MagicMock()
    children = []
    for name in agent_names:
        child = MagicMock()
        child.name = name
        children.append(child)
    root.sub_agents = children
    maw.build = MagicMock(return_value=root)
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


class TestStepPlugin:
    @pytest.mark.asyncio
    async def test_pauses_at_known_agent(self):
        plugin = _StepPlugin({"writer"})

        agent = MagicMock()
        agent.name = "writer"
        ctx = MagicMock()

        paused = False

        async def run_plugin():
            nonlocal paused
            await plugin.before_agent_callback(agent=agent, callback_context=ctx)
            paused = True

        task = asyncio.create_task(run_plugin())
        await asyncio.sleep(0)
        # Plugin should have put agent name in queue and be waiting for resume
        assert not plugin._step_queue.empty()
        assert not paused

        plugin._resume.set()
        await task
        assert paused

    @pytest.mark.asyncio
    async def test_passes_unknown_agent(self):
        plugin = _StepPlugin({"writer"})

        agent = MagicMock()
        agent.name = "reader"
        ctx = MagicMock()

        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        assert plugin._step_queue.empty()

    @pytest.mark.asyncio
    async def test_skips_when_not_pausing(self):
        plugin = _StepPlugin({"writer"})
        plugin._pausing = False

        agent = MagicMock()
        agent.name = "writer"
        ctx = MagicMock()

        result = await plugin.before_agent_callback(agent=agent, callback_context=ctx)
        assert result is None
        assert plugin._step_queue.empty()


class TestIterableRun:
    @pytest.mark.asyncio
    async def test_iterate_all_steps(self):
        maw = _mock_maw("a", "b")
        config = _config("a", "b")

        fake_run = _fake_run_pipeline(["a", "b"], {"a": "done_a", "b": "done_b"})

        with patch("fedotmas.control._iterable.run_pipeline", side_effect=fake_run):
            async with Controller(maw).iter("task", config) as run:
                steps = []
                async for step in run:
                    steps.append(step.name)

            assert steps == ["a", "b"]
            assert run.result.status == "success"
            assert run.result.state == {"a": "done_a", "b": "done_b"}

    @pytest.mark.asyncio
    async def test_break_and_finish(self):
        maw = _mock_maw("a", "b", "c")
        config = _config("a", "b", "c")

        fake_run = _fake_run_pipeline(
            ["a", "b", "c"], {"a": "done_a", "b": "done_b", "c": "done_c"}
        )

        with patch("fedotmas.control._iterable.run_pipeline", side_effect=fake_run):
            async with Controller(maw).iter("task", config) as run:
                async for step in run:
                    if step.name == "b":
                        break
                result = await run.finish()

            assert result.status == "success"
            assert result.state == {"a": "done_a", "b": "done_b", "c": "done_c"}

    @pytest.mark.asyncio
    async def test_step_index_increments(self):
        maw = _mock_maw("a", "b", "c")
        config = _config("a", "b", "c")

        fake_run = _fake_run_pipeline(
            ["a", "b", "c"], {"a": "done_a", "b": "done_b", "c": "done_c"}
        )

        with patch("fedotmas.control._iterable.run_pipeline", side_effect=fake_run):
            async with Controller(maw).iter("task", config) as run:
                indices = []
                async for step in run:
                    indices.append(step.index)

            assert indices == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_state_empty_before_execution(self):
        run = IterableRun(_mock_maw("a"), _config("a"), "task")
        assert run.state == {}

    @pytest.mark.asyncio
    async def test_result_raises_before_completion(self):
        run = IterableRun(_mock_maw("a"), _config("a"), "task")
        with pytest.raises(RuntimeError, match="not completed"):
            _ = run.result

    @pytest.mark.asyncio
    async def test_cleanup_releases_paused_pipeline(self):
        maw = _mock_maw("a", "b")
        config = _config("a", "b")

        fake_run = _fake_run_pipeline(["a", "b"], {"a": "ok", "b": "ok"})

        with patch("fedotmas.control._iterable.run_pipeline", side_effect=fake_run):
            async with Controller(maw).iter("task", config) as run:
                async for step in run:
                    if step.name == "b":
                        break
                # Exit without calling finish() — __aexit__ should cleanup

            assert run._exec_task is not None
            assert run._exec_task.done()

    @pytest.mark.asyncio
    async def test_error_during_pipeline(self):
        maw = _mock_maw("a", "b")
        config = _config("a", "b")

        async def failing_run(_agent, _task, **_kwargs):
            raise RuntimeError("Agent 'b' failed with error 500: boom")

        with patch(
            "fedotmas.control._iterable.run_pipeline", side_effect=failing_run
        ):
            async with Controller(maw).iter("task", config) as run:
                async for step in run:
                    pass

            assert run.result.status == "error"
            assert run.result.error is not None
            assert run.result.error.agent_name == "b"

    @pytest.mark.asyncio
    async def test_checkpoints_created(self):
        maw = _mock_maw("a", "b")
        config = _config("a", "b")

        fake_run = _fake_run_pipeline(["a", "b"], {"a": "done_a", "b": "done_b"})

        with patch("fedotmas.control._iterable.run_pipeline", side_effect=fake_run):
            async with Controller(maw).iter("task", config) as run:
                async for step in run:
                    if step.name == "b":
                        assert len(run.checkpoints) == 1
                        assert run.checkpoints[0].agent_name == "a"
                        break
                result = await run.finish()

            assert len(result.checkpoints) == 2


    @pytest.mark.asyncio
    async def test_single_agent_pipeline(self):
        """Single-agent pipeline (no sub_agents) should yield one step."""
        maw = MagicMock()
        maw._session_service = None
        maw._memory_service = None
        root = MagicMock()
        root.name = "solo"
        root.sub_agents = []
        maw.build = MagicMock(return_value=root)

        config = _config("solo")
        fake_run = _fake_run_pipeline(["solo"], {"solo": "done_solo"})

        with patch("fedotmas.control._iterable.run_pipeline", side_effect=fake_run):
            async with Controller(maw).iter("task", config) as run:
                steps = []
                async for step in run:
                    steps.append(step.name)

            assert steps == ["solo"]
            assert run.result.status == "success"


    @pytest.mark.asyncio
    async def test_build_error_does_not_hang(self):
        """If build() raises, iteration should not deadlock."""
        maw = MagicMock()
        maw._session_service = None
        maw._memory_service = None
        maw.build = MagicMock(side_effect=ValueError("bad config"))

        config = _config("a")

        async with Controller(maw).iter("task", config) as run:
            steps = []
            async for step in run:
                steps.append(step.name)

        assert steps == []
        assert run.result.status == "error"
        assert "bad config" in run.result.error.message


class TestRunWithRecovery:
    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        maw = _mock_maw()
        ctrl = Controller(maw)
        with pytest.raises(NotImplementedError, match="meta-debugger"):
            await ctrl.run_with_recovery("task")
