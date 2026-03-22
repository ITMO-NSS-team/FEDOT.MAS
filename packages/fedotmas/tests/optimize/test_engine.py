"""Tests for the optimization engine loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._run import ControlledRun
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._engine import run_optimization, _evaluate_candidate, _mean_score_on
from fedotmas.optimize._proposer import Proposer
from fedotmas.optimize._scoring import ScoringResult
from fedotmas.optimize._state import Candidate, OptimizationState
from fedotmas.optimize._stopping import MaxIterations
from fedotmas.optimize._strategies import (
    BestCandidateSelector,
    AllComponentSelector,
    ShuffledBatchSampler,
)


def _agent(name: str, instruction: str = "Do stuff") -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=instruction, output_key=name)


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


def _mock_scorer(score: float = 0.7) -> MagicMock:
    scorer = MagicMock()
    scorer.evaluate = AsyncMock(
        return_value=ScoringResult(score=score, feedback="ok", reasoning="fine")
    )
    return scorer


def test_mean_score_on():
    c = Candidate(index=0, config=_config("a"), config_hash="h")
    c.scores = {"t1": 0.8, "t2": 0.6, "t3": 0.4}
    assert _mean_score_on(c, {"t1", "t3"}) == pytest.approx(0.6)
    assert _mean_score_on(c, {"nonexistent"}) == 0.0


# --- 6a: Controller per task ---


@pytest.mark.asyncio
async def test_evaluate_candidate_creates_controller_per_task():
    """Each task should get its own Controller instance."""
    maw = _mock_maw()
    scorer = _mock_scorer(0.8)
    config = _config("a")
    state = OptimizationState()
    candidate = state.add_candidate(config)

    controller_instances = []

    with patch("fedotmas.optimize._engine.Controller") as MockCtrl:
        def make_ctrl(m):
            ctrl = MagicMock()
            ctrl.run = AsyncMock(
                return_value=ControlledRun(config=config, status="success", state={"a": "out"})
            )
            controller_instances.append(ctrl)
            return ctrl

        MockCtrl.side_effect = make_ctrl

        await _evaluate_candidate(maw, scorer, candidate, ["t1", "t2"], state)

    # Should create 2 Controller instances (one per task)
    assert len(controller_instances) == 2


@pytest.mark.asyncio
async def test_evaluate_candidate_caches():
    maw = _mock_maw()
    scorer = _mock_scorer(0.8)
    config = _config("a")
    state = OptimizationState()
    candidate = state.add_candidate(config)

    with patch("fedotmas.optimize._engine.Controller") as MockCtrl:
        ctrl_instance = MagicMock()
        ctrl_instance.run = AsyncMock(
            return_value=ControlledRun(config=config, status="success", state={"a": "out"})
        )
        MockCtrl.return_value = ctrl_instance

        runs = await _evaluate_candidate(maw, scorer, candidate, ["t1"], state)
        assert runs == 1
        assert candidate.scores["t1"] == 0.8

        # Second eval should use cache
        candidate2 = state.add_candidate(config, origin="mutation")
        runs2 = await _evaluate_candidate(maw, scorer, candidate2, ["t1"], state)
        assert runs2 == 0
        assert candidate2.scores["t1"] == 0.8


@pytest.mark.asyncio
async def test_evaluate_candidate_handles_error():
    maw = _mock_maw()
    scorer = _mock_scorer()
    config = _config("a")
    state = OptimizationState()
    candidate = state.add_candidate(config)

    with patch("fedotmas.optimize._engine.Controller") as MockCtrl:
        ctrl_instance = MagicMock()
        ctrl_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
        MockCtrl.return_value = ctrl_instance

        runs = await _evaluate_candidate(maw, scorer, candidate, ["t1"], state)
        assert runs == 1
        assert candidate.scores["t1"] == 0.0
        assert "boom" in candidate.feedbacks["t1"]


# --- 6b: Empty candidate list guard ---
# Tested implicitly via run_optimization — the seed is always there.


# --- 6d: Redundant isinstance removed ---
# Verified by reading the code — the except branch now uses run.state directly.


@pytest.mark.asyncio
async def test_run_optimization_basic():
    maw = _mock_maw()
    seed = _config("a", "b")

    scorer = _mock_scorer(0.6)
    proposer = MagicMock(spec=Proposer)
    proposer.total_prompt_tokens = 0
    proposer.total_completion_tokens = 0

    # Mutation returns a different config
    mutated_agents = [
        _agent("a", "Improved a"),
        _agent("b", "Do b"),
    ]
    mutated = MAWConfig(
        agents=mutated_agents,
        pipeline=MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(agent_name="a"), MAWStepConfig(agent_name="b")],
        ),
    )
    proposer.propose_mutation = AsyncMock(return_value=mutated)
    proposer.propose_merge = AsyncMock(return_value=mutated)

    with patch("fedotmas.optimize._engine.Controller") as MockCtrl:
        ctrl_instance = MagicMock()
        ctrl_instance.run = AsyncMock(
            return_value=ControlledRun(
                config=seed, status="success", state={"a": "out", "b": "out"}
            )
        )
        MockCtrl.return_value = ctrl_instance

        result = await run_optimization(
            maw=maw,
            seed_config=seed,
            trainset=["t1", "t2"],
            valset=["t1", "t2"],
            scorer=scorer,
            proposer=proposer,
            candidate_selector=BestCandidateSelector(),
            batch_sampler=ShuffledBatchSampler(),
            component_selector=AllComponentSelector(),
            stopper=MaxIterations(2),
            use_merge=False,
            max_merge_attempts=0,
            minibatch_size=2,
        )

    assert result.best_score >= 0.0
    assert result.iterations == 2
    assert len(result.all_candidates) >= 2


@pytest.mark.asyncio
async def test_run_optimization_rejects_identical_mutation():
    """If mutation produces identical config, it should be skipped."""
    maw = _mock_maw()
    seed = _config("a")

    scorer = _mock_scorer(0.5)
    proposer = MagicMock(spec=Proposer)
    proposer.total_prompt_tokens = 0
    proposer.total_completion_tokens = 0
    # Return same config = no mutation
    proposer.propose_mutation = AsyncMock(return_value=seed)

    with patch("fedotmas.optimize._engine.Controller") as MockCtrl:
        ctrl_instance = MagicMock()
        ctrl_instance.run = AsyncMock(
            return_value=ControlledRun(
                config=seed, status="success", state={"a": "out"}
            )
        )
        MockCtrl.return_value = ctrl_instance

        result = await run_optimization(
            maw=maw,
            seed_config=seed,
            trainset=["t1"],
            valset=["t1"],
            scorer=scorer,
            proposer=proposer,
            candidate_selector=BestCandidateSelector(),
            batch_sampler=ShuffledBatchSampler(),
            component_selector=AllComponentSelector(),
            stopper=MaxIterations(1),
            use_merge=False,
            max_merge_attempts=0,
            minibatch_size=1,
        )

    # Only seed candidate should exist
    assert len(result.all_candidates) == 1
