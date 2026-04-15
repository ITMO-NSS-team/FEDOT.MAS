"""Tests for Controller class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._controller import Controller
from fedotmas.control._run import ControlledRun
from fedotmas.control._strategy import Strategy
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


def _mock_maw() -> MagicMock:
    maw = MagicMock()
    maw._session_service = None
    maw._memory_service = None
    maw._backend_name = "adk"
    maw._backend_plugins = []
    maw.generate_config = AsyncMock()
    tree = SequentialDescriptor(
        name="seq_1",
        children=[AgentDescriptor(name="a", instruction="Do a", output_key="a")],
    )
    maw.build = MagicMock(return_value=tree)
    return maw


def _mock_runner(return_value=None, side_effect=None):
    runner = AsyncMock()
    if side_effect:
        runner.run_pipeline = AsyncMock(side_effect=side_effect)
    else:
        runner.run_pipeline = AsyncMock(return_value=return_value)
    return runner


@pytest.mark.asyncio
async def test_run_success():
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    runner = _mock_runner(PipelineResult(state={"a": "result_a", "b": "result_b"}))
    with patch.object(maw, "_get_runner", return_value=runner):
        ctrl = Controller(maw)
        run = await ctrl.run("test task")

    assert run.status == "success"
    assert run.state == {"a": "result_a", "b": "result_b"}
    assert run.config is config
    assert run.error is None


@pytest.mark.asyncio
async def test_run_error():
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    runner = _mock_runner(
        side_effect=RuntimeError("Agent 'b' failed with error 500: Internal error")
    )
    with patch.object(maw, "_get_runner", return_value=runner):
        ctrl = Controller(maw)
        run = await ctrl.run("test task")

    assert run.status == "error"
    assert run.error is not None
    assert run.error.agent_name == "b"
    assert "500" in run.error.message


@pytest.mark.asyncio
async def test_run_with_provided_config():
    maw = _mock_maw()
    config = _config("a", "b")

    runner = _mock_runner(PipelineResult(state={"a": "ok"}))
    with patch.object(maw, "_get_runner", return_value=runner):
        ctrl = Controller(maw)
        run = await ctrl.run("test task", config=config)

    maw.generate_config.assert_not_called()
    assert run.config is config


@pytest.mark.asyncio
async def test_resume_calls_run_pipeline_with_initial_state():
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    runner = _mock_runner(PipelineResult(state={"a": "result_a", "b": "result_b"}))
    with patch.object(maw, "_get_runner", return_value=runner):
        ctrl = Controller(maw)
        await ctrl.run("task")

        runner.run_pipeline = AsyncMock(
            return_value=PipelineResult(state={"a": "result_a", "b_v2": "new_result"})
        )
        new_config = _config("a", "b_v2")
        run = await ctrl.resume(new_config)

    assert run.status == "success"
    assert run.config is new_config


@pytest.mark.asyncio
async def test_resume_without_run_raises():
    maw = _mock_maw()
    ctrl = Controller(maw)
    with pytest.raises(RuntimeError, match="No previous run"):
        await ctrl.resume(_config("a"))


@pytest.mark.asyncio
async def test_resume_with_restart_all():
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    runner = _mock_runner(PipelineResult(state={"a": "ok", "b": "ok"}))
    with patch.object(maw, "_get_runner", return_value=runner):
        ctrl = Controller(maw)
        await ctrl.run("task")

        runner.run_pipeline = AsyncMock(
            return_value=PipelineResult(state={"a": "new", "b": "new"})
        )
        new_config = _config("a", "b")
        run = await ctrl.resume(new_config, strategy=Strategy.RESTART_ALL)

    assert run.status == "success"
    call_kwargs = runner.run_pipeline.call_args.kwargs
    assert call_kwargs.get("initial_state") is None


@pytest.mark.asyncio
async def test_result_property():
    run = ControlledRun(
        config=_config("a"),
        status="success",
        state={"a": "value"},
    )
    assert run.result == {"a": "value"}
