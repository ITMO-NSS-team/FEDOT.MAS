"""Tests for the Optimizer public API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._run import ControlledRun
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import Optimizer, OptimizationResult
from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._scoring import LLMJudge, ScoringResult


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
    maw.generate_config = AsyncMock(return_value=_config("a", "b"))
    maw.build = MagicMock(return_value=MagicMock())
    return maw


@pytest.mark.asyncio
async def test_optimizer_empty_trainset():
    maw = _mock_maw()
    opt = Optimizer(maw, criteria="quality")
    with pytest.raises(ValueError, match="trainset must not be empty"):
        await opt.optimize([])


@pytest.mark.asyncio
async def test_optimizer_generates_seed_config():
    maw = _mock_maw()
    scorer = MagicMock()
    scorer.evaluate = AsyncMock(
        return_value=ScoringResult(score=0.5, feedback="ok", reasoning="fine")
    )

    opt = Optimizer(
        maw,
        scorer=scorer,
        config=OptimizationConfig(max_iterations=1),
    )

    with patch("fedotmas.optimize._engine.Controller") as MockCtrl:
        ctrl = MagicMock()
        ctrl.run = AsyncMock(
            return_value=ControlledRun(
                config=_config("a", "b"),
                status="success",
                state={"a": "out", "b": "out"},
            )
        )
        MockCtrl.return_value = ctrl

        with patch("fedotmas.optimize.run_optimization") as mock_engine:
            mock_engine.return_value = OptimizationResult(
                best_config=_config("a", "b"),
                best_score=0.5,
                iterations=1,
            )
            result = await opt.optimize(["task1"])

    maw.generate_config.assert_called_once_with("task1")
    assert result.best_score == 0.5


@pytest.mark.asyncio
async def test_optimizer_uses_provided_seed():
    maw = _mock_maw()
    scorer = MagicMock()
    scorer.evaluate = AsyncMock(
        return_value=ScoringResult(score=0.8, feedback="good", reasoning="solid")
    )
    seed = _config("x", "y")

    opt = Optimizer(
        maw,
        scorer=scorer,
        config=OptimizationConfig(max_iterations=1),
    )

    with patch("fedotmas.optimize.run_optimization") as mock_engine:
        mock_engine.return_value = OptimizationResult(
            best_config=seed,
            best_score=0.8,
            iterations=1,
        )
        result = await opt.optimize(["task1"], seed_config=seed)

    maw.generate_config.assert_not_called()
    assert mock_engine.call_args.kwargs["seed_config"] is seed


@pytest.mark.asyncio
async def test_optimizer_last_result():
    maw = _mock_maw()
    opt = Optimizer(
        maw,
        criteria="quality",
        config=OptimizationConfig(max_iterations=1),
    )
    assert opt.last_result is None

    with patch("fedotmas.optimize.run_optimization") as mock_engine:
        mock_engine.return_value = OptimizationResult(
            best_config=_config("a"),
            best_score=0.9,
            iterations=5,
        )
        await opt.optimize(["task1"])

    assert opt.last_result is not None
    assert opt.last_result.best_score == 0.9


def test_optimizer_defaults():
    maw = _mock_maw()
    opt = Optimizer(maw, criteria="test")
    assert isinstance(opt._scorer, LLMJudge)
    assert opt._config.use_merge is True
    assert opt._config.max_merge_attempts == 5
    assert opt._config.minibatch_size == 3


@pytest.mark.asyncio
async def test_optimizer_token_usage_via_public_api():
    """Token usage should be collected via token_usage property, not private attrs."""
    maw = _mock_maw()

    scorer = MagicMock()
    scorer.evaluate = AsyncMock(
        return_value=ScoringResult(score=0.5, feedback="ok", reasoning="fine")
    )
    scorer.token_usage = (200, 100)

    opt = Optimizer(
        maw,
        scorer=scorer,
        config=OptimizationConfig(max_iterations=1),
    )

    with patch("fedotmas.optimize.run_optimization") as mock_engine:
        mock_engine.return_value = OptimizationResult(
            best_config=_config("a"),
            best_score=0.5,
            iterations=1,
            total_prompt_tokens=50,
            total_completion_tokens=25,
        )
        result = await opt.optimize(["task1"], seed_config=_config("a"))

    assert result.total_prompt_tokens >= 200
    assert result.total_completion_tokens >= 100
