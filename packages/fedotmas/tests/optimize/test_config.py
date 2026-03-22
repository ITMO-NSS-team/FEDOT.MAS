"""Tests for OptimizationConfig."""

from __future__ import annotations

from fedotmas.optimize._config import OptimizationConfig


class TestOptimizationConfig:
    def test_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.temperature_reflect == 0.7
        assert cfg.temperature_merge == 0.5
        assert cfg.temperature_judge == 0.1
        assert cfg.epsilon == 1e-6
        assert cfg.max_merge_context_tasks == 5
        assert cfg.max_state_chars == 2000
        assert cfg.max_output_chars == 3000
        assert cfg.max_consecutive_failures == 3
        assert cfg.seed is None
        # Consolidated fields
        assert cfg.use_merge is True
        assert cfg.max_merge_attempts == 5
        assert cfg.minibatch_size == 3
        assert cfg.candidate_selection == "pareto"
        assert cfg.checkpoint_path is None
        assert cfg.graceful_shutdown is False

    def test_rng_created(self):
        cfg = OptimizationConfig()
        assert cfg.rng is not None

    def test_seed_determinism(self):
        cfg1 = OptimizationConfig(seed=42)
        cfg2 = OptimizationConfig(seed=42)
        vals1 = [cfg1.rng.random() for _ in range(10)]
        vals2 = [cfg2.rng.random() for _ in range(10)]
        assert vals1 == vals2

    def test_different_seeds(self):
        cfg1 = OptimizationConfig(seed=42)
        cfg2 = OptimizationConfig(seed=99)
        vals1 = [cfg1.rng.random() for _ in range(10)]
        vals2 = [cfg2.rng.random() for _ in range(10)]
        assert vals1 != vals2

    def test_custom_values(self):
        cfg = OptimizationConfig(
            temperature_reflect=0.9,
            epsilon=1e-3,
            max_consecutive_failures=5,
            use_merge=False,
            minibatch_size=10,
            candidate_selection="best",
        )
        assert cfg.temperature_reflect == 0.9
        assert cfg.epsilon == 1e-3
        assert cfg.max_consecutive_failures == 5
        assert cfg.use_merge is False
        assert cfg.minibatch_size == 10
        assert cfg.candidate_selection == "best"
