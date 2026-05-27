"""Tests for fedotmas.routing.algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedotmas.routing.algorithm import (
    EXPLORE_UTILITY,
    aggregate,
    pareto_filter,
    score_utility,
    thompson_sample,
)
from fedotmas.routing.models import (
    ExperienceRecord,
    LlmPool,
    LlmPoolEntry,
    LlmStats,
    Weights,
)


# ─── Fixtures ─────────────────────────────────────────────────────────


def _pool(*names: str) -> LlmPool:
    # Non-zero default prices avoid the zero-price warning in test logs.
    return LlmPool(
        entries=tuple(
            LlmPoolEntry(model=n, input_price_per_1m=1.0, output_price_per_1m=1.0)
            for n in names
        )
    )


def _rec(
    *,
    llm: str,
    success_step: float = 1.0,
    success_task: float | None = 1.0,
    cost: float = 0.01,
    duration: float = 1.0,
) -> ExperienceRecord:
    return ExperienceRecord(
        trace_id="t",
        step_idx=0,
        agent_role="r",
        llm_used=llm,
        query="q",
        embedding=np.zeros(4, dtype=np.float32),
        tools=("search",),
        cost=cost,
        duration=duration,
        success_step=success_step,
        success_task=success_task,
    )


# ─── aggregate ────────────────────────────────────────────────────────


class TestAggregate:
    def test_empty_records_returns_empty_stats_per_pool_model(self) -> None:
        pool = _pool("a", "b")
        out = aggregate([], pool)
        assert set(out.keys()) == {"a", "b"}
        assert all(s.n == 0 for s in out.values())

    def test_groups_by_llm_used(self) -> None:
        pool = _pool("a", "b")
        recs = [_rec(llm="a"), _rec(llm="a"), _rec(llm="b")]
        out = aggregate(recs, pool)
        assert out["a"].n == 2
        assert out["b"].n == 1

    def test_skips_records_with_unscored_task(self) -> None:
        pool = _pool("a")
        recs = [_rec(llm="a"), _rec(llm="a", success_task=None)]
        out = aggregate(recs, pool)
        assert out["a"].n == 1

    def test_skips_records_for_models_outside_pool(self) -> None:
        pool = _pool("a")
        recs = [_rec(llm="a"), _rec(llm="stranger")]
        out = aggregate(recs, pool)
        assert set(out.keys()) == {"a"}
        assert out["a"].n == 1

    def test_perf_is_product_of_step_and_task_success(self) -> None:
        pool = _pool("a")
        recs = [_rec(llm="a", success_step=0.5, success_task=0.4)]
        out = aggregate(recs, pool)
        assert out["a"].perf == [pytest.approx(0.2)]

    def test_cost_min_max_normalised_to_unit_interval(self) -> None:
        # cheap=0.001, mid=0.01, dear=0.1  →  after min-max  0, 0.1, 1
        pool = _pool("cheap", "mid", "dear")
        recs = [
            _rec(llm="cheap", cost=0.001),
            _rec(llm="mid", cost=0.01),
            _rec(llm="dear", cost=0.1),
        ]
        out = aggregate(recs, pool)
        assert out["cheap"].cost == [pytest.approx(0.0)]
        assert out["mid"].cost == [pytest.approx(0.09090909)]
        assert out["dear"].cost == [pytest.approx(1.0)]

    def test_delay_min_max_normalised_to_unit_interval(self) -> None:
        pool = _pool("fast", "slow")
        recs = [
            _rec(llm="fast", duration=0.5),
            _rec(llm="slow", duration=10.0),
        ]
        out = aggregate(recs, pool)
        assert out["fast"].delay == [pytest.approx(0.0)]
        assert out["slow"].delay == [pytest.approx(1.0)]

    def test_zero_range_yields_zero_not_division_error(self) -> None:
        # All records have identical cost/duration → range = 0; we emit
        # 0.0 so the metric is inert this round (no NaN, no ZeroDiv).
        pool = _pool("a", "b")
        recs = [
            _rec(llm="a", cost=0.5, duration=2.0),
            _rec(llm="b", cost=0.5, duration=2.0),
        ]
        out = aggregate(recs, pool)
        assert out["a"].cost == [0.0]
        assert out["b"].cost == [0.0]
        assert out["a"].delay == [0.0]
        assert out["b"].delay == [0.0]

    def test_normalisation_uses_global_range_not_per_model(self) -> None:
        # If we accidentally normalised per-model, every entry would be
        # 0.0 since each model has only one observation. Cross-model
        # ranking would be lost. Guard against that regression.
        pool = _pool("a", "b")
        recs = [_rec(llm="a", cost=1.0), _rec(llm="b", cost=10.0)]
        out = aggregate(recs, pool)
        assert out["a"].cost == [0.0]
        assert out["b"].cost == [1.0]


# ─── pareto_filter ────────────────────────────────────────────────────


def _stats(model: str, *, n: int, perf: float, cost: float, delay: float) -> LlmStats:
    return LlmStats(
        model=model,
        perf=[perf] * n,
        cost=[cost] * n,
        delay=[delay] * n,
    )


class TestParetoFilter:
    def test_empty(self) -> None:
        assert pareto_filter({}) == []

    def test_undersampled_models_always_pass(self) -> None:
        out = pareto_filter({"a": LlmStats(model="a")})
        assert out == ["a"]

    def test_single_well_sampled_model_passes(self) -> None:
        out = pareto_filter(
            {"a": _stats("a", n=5, perf=0.8, cost=0.1, delay=1.0)}
        )
        assert out == ["a"]

    def test_dominated_model_removed(self) -> None:
        sb = {
            "good": _stats("good", n=5, perf=0.9, cost=0.1, delay=1.0),
            "bad": _stats("bad", n=5, perf=0.5, cost=0.5, delay=2.0),
        }
        assert pareto_filter(sb) == ["good"]

    def test_pareto_front_with_three_models(self) -> None:
        # cheap+slow vs expensive+fast vs middle — all non-dominated
        sb = {
            "cheap": _stats("cheap", n=5, perf=0.7, cost=0.01, delay=5.0),
            "fast": _stats("fast", n=5, perf=0.7, cost=0.5, delay=0.5),
            "mid": _stats("mid", n=5, perf=0.7, cost=0.1, delay=1.5),
        }
        assert set(pareto_filter(sb)) == {"cheap", "fast", "mid"}

    def test_undersampled_pass_alongside_dominated(self) -> None:
        sb = {
            "good": _stats("good", n=5, perf=0.9, cost=0.1, delay=1.0),
            "bad": _stats("bad", n=5, perf=0.5, cost=0.5, delay=2.0),
            "new": LlmStats(model="new"),  # n=0
        }
        out = pareto_filter(sb)
        assert "good" in out
        assert "new" in out
        assert "bad" not in out


# ─── thompson_sample ──────────────────────────────────────────────────


class TestThompsonSample:
    def test_n_zero_returns_explore_sentinel(self) -> None:
        rng = np.random.default_rng(0)
        p, c, d = thompson_sample(LlmStats(model="m"), rng=rng)
        assert p == EXPLORE_UTILITY
        assert c == EXPLORE_UTILITY
        assert d == EXPLORE_UTILITY

    def test_n_one_returns_explore_sentinel(self) -> None:
        rng = np.random.default_rng(0)
        s = LlmStats(model="m", perf=[0.8], cost=[0.1], delay=[1.0])
        p, c, d = thompson_sample(s, rng=rng)
        assert p == EXPLORE_UTILITY
        assert c == EXPLORE_UTILITY
        assert d == EXPLORE_UTILITY

    def test_zero_variance_returns_sample_mean(self) -> None:
        rng = np.random.default_rng(0)
        s = LlmStats(
            model="m",
            perf=[0.7, 0.7, 0.7],
            cost=[0.1, 0.1, 0.1],
            delay=[1.0, 1.0, 1.0],
        )
        p, c, d = thompson_sample(s, rng=rng)
        assert p == pytest.approx(0.7)
        assert c == pytest.approx(0.1)
        assert d == pytest.approx(1.0)

    def test_samples_concentrate_near_mean_for_large_n(self) -> None:
        rng = np.random.default_rng(42)
        true_mean = 0.75
        n_obs = 100
        s = LlmStats(
            model="m",
            perf=list(rng.normal(true_mean, 0.05, n_obs).tolist()),
            cost=[0.1] * n_obs,
            delay=[1.0] * n_obs,
        )
        draws = [thompson_sample(s, rng=rng)[0] for _ in range(500)]
        empirical = float(np.mean(draws))
        # Concentration: 500 draws → std of posterior-mean ≈ σ/√n ≈ 0.005
        # so empirical mean stays within ±0.02 of true_mean comfortably.
        assert abs(empirical - true_mean) < 0.02

    def test_deterministic_with_fixed_seed(self) -> None:
        s = LlmStats(model="m", perf=[0.6, 0.8, 0.7], cost=[0.1, 0.2, 0.15], delay=[1.0, 1.5, 1.2])
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        assert thompson_sample(s, rng=rng_a) == thompson_sample(s, rng=rng_b)


# ─── score_utility ────────────────────────────────────────────────────


class TestScoreUtility:
    def test_basic_arithmetic(self) -> None:
        w = Weights(perf=1.0, cost=0.1, delay=0.05)
        # 1.0·0.8 − 0.1·0.5 − 0.05·2.0 = 0.8 − 0.05 − 0.1 = 0.65
        assert score_utility(0.8, 0.5, 2.0, weights=w) == pytest.approx(0.65)

    def test_perf_drives_up(self) -> None:
        w = Weights()
        lo = score_utility(0.1, 0.0, 0.0, weights=w)
        hi = score_utility(0.9, 0.0, 0.0, weights=w)
        assert hi > lo

    def test_cost_drives_down(self) -> None:
        w = Weights()
        cheap = score_utility(0.5, 0.01, 0.0, weights=w)
        pricey = score_utility(0.5, 1.0, 0.0, weights=w)
        assert cheap > pricey

    def test_delay_drives_down(self) -> None:
        w = Weights()
        fast = score_utility(0.5, 0.0, 0.1, weights=w)
        slow = score_utility(0.5, 0.0, 10.0, weights=w)
        assert fast > slow

    @pytest.mark.parametrize(
        "p,c,d",
        [
            (EXPLORE_UTILITY, 0.1, 1.0),
            (0.5, EXPLORE_UTILITY, 1.0),
            (0.5, 0.1, EXPLORE_UTILITY),
        ],
    )
    def test_any_explore_sentinel_propagates(self, p: float, c: float, d: float) -> None:
        assert math.isinf(score_utility(p, c, d, weights=Weights()))
