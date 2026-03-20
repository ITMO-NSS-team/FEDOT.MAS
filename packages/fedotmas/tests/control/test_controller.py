"""Tests for Controller class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._controller import Controller
from fedotmas.control._run import ControlledRun
from fedotmas.control._strategy import Strategy
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
    maw.generate_config = AsyncMock()
    maw.build = MagicMock(return_value=MagicMock())
    return maw


@pytest.mark.asyncio
async def test_run_success():
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    with patch("fedotmas.control._controller.run_pipeline") as mock_run:
        mock_run.return_value = PipelineResult(
            state={"a": "result_a", "b": "result_b"}
        )
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

    with patch("fedotmas.control._controller.run_pipeline") as mock_run:
        mock_run.side_effect = RuntimeError(
            "Agent 'b' failed with error 500: Internal error"
        )
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

    with patch("fedotmas.control._controller.run_pipeline") as mock_run:
        mock_run.return_value = PipelineResult(state={"a": "ok"})
        ctrl = Controller(maw)
        run = await ctrl.run("test task", config=config)

    maw.generate_config.assert_not_called()
    assert run.config is config


@pytest.mark.asyncio
async def test_resume_calls_run_pipeline_with_initial_state():
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    with patch("fedotmas.control._controller.run_pipeline") as mock_run:
        mock_run.return_value = PipelineResult(
            state={"a": "result_a", "b": "result_b"}
        )
        ctrl = Controller(maw)
        await ctrl.run("task")

        new_config = _config("a", "b_v2")
        mock_run.return_value = PipelineResult(
            state={"a": "result_a", "b_v2": "new_result"}
        )
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

    with patch("fedotmas.control._controller.run_pipeline") as mock_run:
        mock_run.return_value = PipelineResult(state={"a": "ok", "b": "ok"})
        ctrl = Controller(maw)
        await ctrl.run("task")

        new_config = _config("a", "b")
        mock_run.return_value = PipelineResult(state={"a": "new", "b": "new"})
        run = await ctrl.resume(new_config, strategy=Strategy.RESTART_ALL)

    assert run.status == "success"
    call_kwargs = mock_run.call_args_list[-1].kwargs
    assert call_kwargs.get("initial_state") is None


@pytest.mark.asyncio
async def test_result_property():
    run = ControlledRun(
        config=_config("a"),
        status="success",
        state={"a": "value"},
    )
    assert run.result == {"a": "value"}
