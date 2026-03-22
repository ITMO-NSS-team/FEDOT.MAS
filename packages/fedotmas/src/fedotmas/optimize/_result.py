"""OptimizationResult — outcome of an optimization run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedotmas.maw.models import MAWConfig
    from fedotmas.optimize._callbacks import OptimizationMetrics
    from fedotmas.optimize._state import Candidate


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_config: MAWConfig
    best_score: float
    all_candidates: list[Candidate] = field(default_factory=list)
    iterations: int = 0
    total_evaluation_runs: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    metrics: OptimizationMetrics | None = None

    def pareto_front(self) -> list[Candidate]:
        """Return candidates on the Pareto front."""
        return [c for c in self.all_candidates if c.on_pareto_front]
