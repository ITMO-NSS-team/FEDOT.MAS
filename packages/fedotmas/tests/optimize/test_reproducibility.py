"""Tests for reproducibility with seeded RNG."""

from __future__ import annotations

from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._strategies import (
    EpsilonGreedySelector,
    ParetoCandidateSelector,
    ShuffledBatchSampler,
    make_candidate_selector,
)
from fedotmas.optimize._state import Candidate
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig


def _config() -> MAWConfig:
    agents = [MAWAgentConfig(name="a", instruction="Do a", output_key="a")]
    return MAWConfig(agents=agents, pipeline=MAWStepConfig(agent_name="a"))


def _candidate(index: int, score: float, on_pareto: bool = False) -> Candidate:
    c = Candidate(index=index, config=_config(), config_hash=f"h{index}")
    c.scores = {"t": score}
    c.on_pareto_front = on_pareto
    return c


class TestSeedReproducibility:
    def test_batch_sampler_deterministic(self):
        cfg1 = OptimizationConfig(seed=42)
        cfg2 = OptimizationConfig(seed=42)
        tasks = [f"t{i}" for i in range(20)]

        s1 = ShuffledBatchSampler(rng=cfg1.rng)
        s2 = ShuffledBatchSampler(rng=cfg2.rng)

        batches1 = [s1.sample(tasks, 5) for _ in range(10)]
        batches2 = [s2.sample(tasks, 5) for _ in range(10)]
        assert batches1 == batches2

    def test_pareto_selector_deterministic(self):
        cfg1 = OptimizationConfig(seed=42)
        cfg2 = OptimizationConfig(seed=42)

        candidates = [_candidate(i, 0.5 + i * 0.1, on_pareto=True) for i in range(5)]

        s1 = ParetoCandidateSelector(rng=cfg1.rng)
        s2 = ParetoCandidateSelector(rng=cfg2.rng)

        picks1 = [s1.select(candidates).index for _ in range(20)]
        picks2 = [s2.select(candidates).index for _ in range(20)]
        assert picks1 == picks2

    def test_epsilon_greedy_deterministic(self):
        cfg1 = OptimizationConfig(seed=42)
        cfg2 = OptimizationConfig(seed=42)

        candidates = [_candidate(i, 0.3 + i * 0.1) for i in range(5)]

        s1 = EpsilonGreedySelector(epsilon=0.5, rng=cfg1.rng)
        s2 = EpsilonGreedySelector(epsilon=0.5, rng=cfg2.rng)

        picks1 = [s1.select(candidates).index for _ in range(50)]
        picks2 = [s2.select(candidates).index for _ in range(50)]
        assert picks1 == picks2

    def test_make_candidate_selector_with_rng(self):
        cfg = OptimizationConfig(seed=42)
        sel = make_candidate_selector("pareto", rng=cfg.rng)
        assert isinstance(sel, ParetoCandidateSelector)

    def test_different_seeds_differ(self):
        cfg1 = OptimizationConfig(seed=42)
        cfg2 = OptimizationConfig(seed=99)
        tasks = [f"t{i}" for i in range(20)]

        s1 = ShuffledBatchSampler(rng=cfg1.rng)
        s2 = ShuffledBatchSampler(rng=cfg2.rng)

        batches1 = [s1.sample(tasks, 5) for _ in range(10)]
        batches2 = [s2.sample(tasks, 5) for _ in range(10)]
        assert batches1 != batches2
