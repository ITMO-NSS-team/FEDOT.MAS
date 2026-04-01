"""Tests for OptimizationConfig."""

from __future__ import annotations

from fedotmas.optimize._config import OptimizationConfig


class TestOptimizationConfig:
    def test_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.temperature_reflect == 0.7
        assert cfg.temperature_merge == 0.5
        assert cfg.temperature_judge == 0.1
        assert cfg.improvement_epsilon == 1e-6
        assert cfg.max_merge_context_tasks == 5
        assert cfg.max_state_chars is None
        assert cfg.max_output_chars is None
        assert cfg.max_consecutive_failures == 3
        assert cfg.seed is None
        assert cfg.use_merge is True
        assert cfg.max_merge_attempts == 5
        assert cfg.minibatch_size == 3
        assert cfg.candidate_selection == "pareto"
        assert cfg.checkpoint_path is None
        assert cfg.graceful_shutdown is False
        # Stopping criteria defaults
        assert cfg.max_iterations == 20
        assert cfg.patience == 5
        assert cfg.score_threshold is None
        assert cfg.max_evaluations is None
        # LLM safety
        assert cfg.llm_timeout == 120.0

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
            improvement_epsilon=1e-3,
            max_consecutive_failures=5,
            use_merge=False,
            minibatch_size=10,
            candidate_selection="best",
            max_iterations=50,
            patience=10,
            score_threshold=0.95,
            llm_timeout=60.0,
        )
        assert cfg.temperature_reflect == 0.9
        assert cfg.improvement_epsilon == 1e-3
        assert cfg.max_consecutive_failures == 5
        assert cfg.use_merge is False
        assert cfg.minibatch_size == 10
        assert cfg.candidate_selection == "best"
        assert cfg.max_iterations == 50
        assert cfg.patience == 10
        assert cfg.score_threshold == 0.95
        assert cfg.llm_timeout == 60.0
