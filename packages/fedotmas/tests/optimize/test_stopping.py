"""Tests for stopping conditions."""

from __future__ import annotations

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._state import OptimizationState, TaskResult
from fedotmas.optimize._stopping import (
    CompositeStopper,
    MaxEvaluations,
    MaxIterations,
    NoImprovement,
    ScoreThreshold,
)


def _config() -> MAWConfig:
    agents = [MAWAgentConfig(name="a", instruction="Do a", output_key="a")]
    return MAWConfig(agents=agents, pipeline=MAWStepConfig(agent_name="a"))


def _state_with_score(score: float) -> OptimizationState:
    state = OptimizationState()
    c = state.add_candidate(_config())
    c.scores = {"t1": score}
    return state


class TestMaxIterations:
    def test_stops_at_max(self):
        s = MaxIterations(5)
        state = OptimizationState()
        assert s.should_stop(state, 4) is False
        assert s.should_stop(state, 5) is True
        assert s.should_stop(state, 6) is True


class TestMaxEvaluations:
    def test_stops_at_max_evaluations(self):
        s = MaxEvaluations(2)
        state = OptimizationState()
        c = state.add_candidate(_config())

        state.record_task_result(
            c, TaskResult(task="t1", state={}, score=0.5, feedback="")
        )
        assert s.should_stop(state, 0) is False

        state.record_task_result(
            c, TaskResult(task="t2", state={}, score=0.5, feedback="")
        )
        assert s.should_stop(state, 0) is True

    def test_not_affected_by_cache_eviction(self):
        """total_evaluations keeps counting even if cache evicts entries."""
        from fedotmas.optimize._state import EvaluationCache

        s = MaxEvaluations(5)
        state = OptimizationState()
        state.cache = EvaluationCache(max_size=2)
        c = state.add_candidate(_config())

        for i in range(4):
            state.record_task_result(
                c, TaskResult(task=f"t{i}", state={}, score=0.5, feedback="")
            )
        # Cache only has 2 entries, but total_evaluations is 4
        assert len(state.cache) == 2
        assert state.total_evaluations == 4
        assert s.should_stop(state, 0) is False

        state.record_task_result(
            c, TaskResult(task="t4", state={}, score=0.5, feedback="")
        )
        assert s.should_stop(state, 0) is True


class TestNoImprovement:
    def test_patience(self):
        s = NoImprovement(patience=3)
        state = _state_with_score(0.5)
        # First check — sets baseline
        assert s.should_stop(state, 0) is False
        # No improvement for 3 iterations
        assert s.should_stop(state, 1) is False
        assert s.should_stop(state, 2) is False
        assert s.should_stop(state, 3) is True

    def test_resets_on_improvement(self):
        s = NoImprovement(patience=2)
        state = _state_with_score(0.5)
        assert s.should_stop(state, 0) is False  # baseline
        assert s.should_stop(state, 1) is False  # stale=1

        # Improve
        state.candidates[0].scores["t1"] = 0.8
        assert s.should_stop(state, 2) is False  # reset
        assert s.should_stop(state, 3) is False  # stale=1
        assert s.should_stop(state, 4) is True   # stale=2

    def test_reset_method(self):
        """reset() should clear internal state."""
        s = NoImprovement(patience=2)
        state = _state_with_score(0.5)
        s.should_stop(state, 0)  # sets baseline
        s.should_stop(state, 1)  # stale=1
        s.should_stop(state, 2)  # stale=2 → would be True

        s.reset()
        assert s._best_score == -1.0
        assert s._stale_count == 0
        # After reset, should behave as fresh
        assert s.should_stop(state, 0) is False


class TestScoreThreshold:
    def test_stops_when_reached(self):
        s = ScoreThreshold(0.9)
        state = _state_with_score(0.5)
        assert s.should_stop(state, 0) is False
        state.candidates[0].scores["t1"] = 0.95
        assert s.should_stop(state, 1) is True


class TestCompositeStopper:
    def test_any_stops(self):
        s = CompositeStopper([MaxIterations(5), ScoreThreshold(0.9)])
        state = _state_with_score(0.5)
        assert s.should_stop(state, 3) is False
        assert s.should_stop(state, 5) is True  # MaxIterations fires

    def test_score_threshold_fires(self):
        s = CompositeStopper([MaxIterations(100), ScoreThreshold(0.9)])
        state = _state_with_score(0.95)
        assert s.should_stop(state, 0) is True  # ScoreThreshold fires

    def test_reset_propagates(self):
        """CompositeStopper.reset() should reset sub-stoppers that support it."""
        ni = NoImprovement(patience=2)
        state = _state_with_score(0.5)
        ni.should_stop(state, 0)
        ni.should_stop(state, 1)

        s = CompositeStopper([MaxIterations(100), ni])
        s.reset()
        assert ni._best_score == -1.0
        assert ni._stale_count == 0
