from __future__ import annotations

import random as _random_module
from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """Centralised configuration for all optimization hyperparameters."""

    # LLM temperatures
    temperature_reflect: float = 0.7
    temperature_merge: float = 0.5
    temperature_judge: float = 0.1

    # Numeric thresholds
    epsilon: float = 1e-6
    max_merge_context_tasks: int = 5
    max_state_chars: int = 2000
    max_output_chars: int = 3000

    # Error recovery
    max_consecutive_failures: int = 3

    # Optimizer params (previously scattered across Optimizer.__init__ / run_optimization)
    use_merge: bool = True
    max_merge_attempts: int = 5
    minibatch_size: int = 3
    candidate_selection: str = "pareto"
    checkpoint_path: str | None = None
    graceful_shutdown: bool = False

    # Reproducibility
    seed: int | None = None
    _rng: _random_module.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = _random_module.Random(self.seed)

    @property
    def rng(self) -> _random_module.Random:
        return self._rng
