"""Tests for IterableRun and Controller.iter context manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._controller import Controller
from fedotmas.control._iterable import IterableRun, _StepMiddleware
from fedotmas.interfaces.agent import AgentDescriptor, SequentialDescriptor
from fedotmas.interfaces.runner import PipelineResult
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
    maw._backend_name = "adk"
    maw._backend_plugins = []
    # Build returns descriptor tree
    children = [
        AgentDescriptor(name=name, instruction=f"Do {name}", output_key=name)
        for name in agent_names
    ]
    if len(children) == 1:
        tree = children[0]
    else:
        tree = SequentialDescriptor(name="seq_1", children=children)
    maw.build = MagicMock(return_value=tree)
    return maw


def _fake_runner(agent_names: list[str], final_state: dict):
    """Return a mock runner that simulates pipeline execution with middleware callbacks."""
    runner = AsyncMock()

    async def _run_pipeline(agent_tree, task, *, middlewares=None, backend_plugins=None, **_kwargs):
        state = dict(_kwargs.get("initial_state") or {})
        for name in agent_names:
            for mw in middlewares or []:
                result = await mw.before_agent(name, state)
                if result is not None:
                    continue

            state[name] = f"done_{name}"

            for mw in middlewares or []:
                await mw.after_agent(name, state)

        return PipelineResult(state=dict(final_state))

    runner.run_pipeline = _run_pipeline
    return runner


class TestStepMiddleware:
    @pytest.mark.asyncio
    async def test_pauses_at_known_agent(self):
        mw = _StepMiddleware({"writer"})

        paused = False

        async def run_mw():
            nonlocal paused
            await mw.before_agent("writer", {})
            paused = True

        task = asyncio.create_task(run_mw())
        await asyncio.sleep(0)
        assert not mw._step_queue.empty()
        assert not paused

        mw._resume.set()
        await task
        assert paused

    @pytest.mark.asyncio
    async def test_passes_unknown_agent(self):
        mw = _StepMiddleware({"writer"})
        result = await mw.before_agent("reader", {})
        assert result is None
        assert mw._step_queue.empty()

    @pytest.mark.asyncio
    async def test_skips_when_not_pausing(self):
        mw = _StepMiddleware({"writer"})
        mw._pausing = False
        result = await mw.before_agent("writer", {})
        assert result is None
        assert mw._step_queue.empty()


class TestIterableRun:
    @pytest.mark.asyncio
    async def test_iterate_all_steps(self):
        maw = _mock_maw("a", "b")
        config = _config("a", "b")
        runner = _fake_runner(["a", "b"], {"a": "done_a", "b": "done_b"})

        with patch.object(maw, "_get_runner", return_value=runner):
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
        runner = _fake_runner(
            ["a", "b", "c"], {"a": "done_a", "b": "done_b", "c": "done_c"}
        )

        with patch.object(maw, "_get_runner", return_value=runner):
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
        runner = _fake_runner(
            ["a", "b", "c"], {"a": "done_a", "b": "done_b", "c": "done_c"}
        )

        with patch.object(maw, "_get_runner", return_value=runner):
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
        runner = _fake_runner(["a", "b"], {"a": "ok", "b": "ok"})

        with patch.object(maw, "_get_runner", return_value=runner):
            async with Controller(maw).iter("task", config) as run:
                async for step in run:
                    if step.name == "b":
                        break

            assert run._exec_task is not None
            assert run._exec_task.done()

    @pytest.mark.asyncio
    async def test_error_during_pipeline(self):
        maw = _mock_maw("a", "b")
        config = _config("a", "b")

        runner = AsyncMock()

        async def failing_run(agent_tree, task, **_kwargs):
            raise RuntimeError("Agent 'b' failed with error 500: boom")

        runner.run_pipeline = failing_run

        with patch.object(maw, "_get_runner", return_value=runner):
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
        runner = _fake_runner(["a", "b"], {"a": "done_a", "b": "done_b"})

        with patch.object(maw, "_get_runner", return_value=runner):
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
        maw = _mock_maw("solo")
        config = _config("solo")
        runner = _fake_runner(["solo"], {"solo": "done_solo"})

        with patch.object(maw, "_get_runner", return_value=runner):
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
        maw._backend_name = "adk"
        maw._backend_plugins = []
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
    async def test_run_with_recovery_is_implemented(self):
        """run_with_recovery is no longer a stub — see test_recovery.py."""
        maw = _mock_maw()
        maw.meta_model = None
        maw.worker_models = None
        maw.temperature = None
        maw.mcp_registry = None
        config = MAWConfig(
            agents=[MAWAgentConfig(name="a", instruction="do a", output_key="a")],
            pipeline=MAWStepConfig(agent_name="a"),
        )
        maw.generate_config = AsyncMock(return_value=config)

        runner = AsyncMock()
        runner.run_pipeline = AsyncMock(return_value=PipelineResult(state={"a": "ok"}))

        with patch.object(maw, "_get_runner", return_value=runner):
            ctrl = Controller(maw)
            run = await ctrl.run_with_recovery("task")
        assert run.status == "success"
