"""Tests for the callback and metrics system."""

from __future__ import annotations

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._callbacks import (
    CallbackDispatcher,
    MetricsCallback,
    OptimizationCallback,
    OptimizationMetrics,
)
from fedotmas.optimize._result import OptimizationResult
from fedotmas.optimize._state import Candidate, OptimizationState


def _config() -> MAWConfig:
    agents = [MAWAgentConfig(name="a", instruction="Do a", output_key="a")]
    return MAWConfig(agents=agents, pipeline=MAWStepConfig(agent_name="a"))


def _candidate(index: int, score: float) -> Candidate:
    c = Candidate(index=index, config=_config(), config_hash=f"h{index}")
    c.scores = {"t": score}
    return c


class TestOptimizationCallback:
    def test_all_methods_callable(self):
        cb = OptimizationCallback()
        state = OptimizationState()
        result = OptimizationResult(best_config=_config(), best_score=0.5)
        parent = _candidate(0, 0.5)
        child = _candidate(1, 0.6)

        cb.on_iteration_start(1, state)
        cb.on_candidate_evaluated(child, ["t"])
        cb.on_candidate_accepted(child, parent)
        cb.on_candidate_rejected(child, parent)
        cb.on_merge_attempted((parent, child))
        cb.on_iteration_end(1, state)
        cb.on_optimization_end(result)


class TestCallbackDispatcher:
    def test_dispatches_to_all(self):
        calls: list[str] = []

        class TrackerCb(OptimizationCallback):
            def on_iteration_start(self, iteration, state):
                calls.append(f"start_{iteration}")

            def on_iteration_end(self, iteration, state):
                calls.append(f"end_{iteration}")

        dispatcher = CallbackDispatcher([TrackerCb(), TrackerCb()])
        state = OptimizationState()
        dispatcher.on_iteration_start(1, state)
        dispatcher.on_iteration_end(1, state)

        assert calls == ["start_1", "start_1", "end_1", "end_1"]

    def test_add(self):
        calls: list[str] = []

        class TrackerCb(OptimizationCallback):
            def on_iteration_start(self, iteration, state):
                calls.append("called")

        dispatcher = CallbackDispatcher()
        dispatcher.add(TrackerCb())
        dispatcher.on_iteration_start(1, OptimizationState())
        assert calls == ["called"]

    def test_error_isolation(self):
        """A failing callback should not prevent others from running."""
        calls: list[str] = []

        class FailingCb(OptimizationCallback):
            def on_iteration_start(self, iteration, state):
                raise RuntimeError("boom")

        class GoodCb(OptimizationCallback):
            def on_iteration_start(self, iteration, state):
                calls.append("ok")

        dispatcher = CallbackDispatcher([FailingCb(), GoodCb()])
        dispatcher.on_iteration_start(1, OptimizationState())
        assert calls == ["ok"]

class TestOptimizationMetrics:
    def test_acceptance_rate(self):
        m = OptimizationMetrics(accepted=3, rejected=7)
        assert m.acceptance_rate == 0.3

    def test_acceptance_rate_zero(self):
        m = OptimizationMetrics()
        assert m.acceptance_rate == 0.0

    def test_cache_hit_rate(self):
        m = OptimizationMetrics(cache_hits=6, cache_misses=4)
        assert m.cache_hit_rate == 0.6

    def test_cache_hit_rate_zero(self):
        m = OptimizationMetrics()
        assert m.cache_hit_rate == 0.0


class TestMetricsCallback:
    def test_tracks_acceptance_rejection(self):
        cb = MetricsCallback()
        parent = _candidate(0, 0.5)
        child = _candidate(1, 0.6)

        cb.on_candidate_accepted(child, parent)
        cb.on_candidate_accepted(child, parent)
        cb.on_candidate_rejected(child, parent)

        assert cb.metrics.accepted == 2
        assert cb.metrics.rejected == 1
        assert cb.metrics.acceptance_rate == 2.0 / 3.0

    def test_tracks_merge_attempts(self):
        cb = MetricsCallback()
        parent = _candidate(0, 0.5)
        child = _candidate(1, 0.6)
        cb.on_merge_attempted((parent, child))
        assert cb.metrics.merge_attempts == 1

    def test_tracks_best_score_history(self):
        cb = MetricsCallback()
        state = OptimizationState()
        c = state.add_candidate(_config())
        c.scores = {"t": 0.5}

        cb.on_iteration_end(1, state)
        c.scores["t"] = 0.8
        cb.on_iteration_end(2, state)

        assert cb.metrics.best_score_history == [0.5, 0.8]
