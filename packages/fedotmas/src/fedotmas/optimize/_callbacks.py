from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from fedotmas.optimize._result import OptimizationResult
    from fedotmas.optimize._state import Candidate, OptimizationState, Task

_log = get_logger("fedotmas.optimize._callbacks")


class OptimizationCallback:
    """Default callback that does nothing. Subclass and override specific methods."""

    def on_iteration_start(self, iteration: int, state: OptimizationState) -> None:
        pass

    def on_candidate_evaluated(self, candidate: Candidate, tasks: list[Task]) -> None:
        pass

    def on_candidate_accepted(self, child: Candidate, parent: Candidate) -> None:
        pass

    def on_candidate_rejected(self, child: Candidate, parent: Candidate) -> None:
        pass

    def on_merge_attempted(self, pair: tuple[Candidate, Candidate]) -> None:
        pass

    def on_iteration_end(self, iteration: int, state: OptimizationState) -> None:
        pass

    def on_optimization_end(self, result: OptimizationResult) -> None:
        pass


class CallbackDispatcher:
    """Dispatches events to multiple callbacks with error isolation."""

    def __init__(self, callbacks: list[OptimizationCallback] | None = None) -> None:
        self._callbacks: list[OptimizationCallback] = callbacks or []

    def add(self, cb: OptimizationCallback) -> None:
        self._callbacks.append(cb)

    def _dispatch(self, name: str, *args: Any, **kwargs: Any) -> None:
        for cb in self._callbacks:
            method = getattr(cb, name, None)
            if method is not None:
                try:
                    method(*args, **kwargs)
                except Exception:
                    _log.warning(
                        "Callback error in {}.{}",
                        type(cb).__name__,
                        name,
                        exc_info=True,
                    )

    def on_iteration_start(self, iteration: int, state: OptimizationState) -> None:
        self._dispatch("on_iteration_start", iteration, state)

    def on_candidate_evaluated(self, candidate: Candidate, tasks: list[Task]) -> None:
        self._dispatch("on_candidate_evaluated", candidate, tasks)

    def on_candidate_accepted(self, child: Candidate, parent: Candidate) -> None:
        self._dispatch("on_candidate_accepted", child, parent)

    def on_candidate_rejected(self, child: Candidate, parent: Candidate) -> None:
        self._dispatch("on_candidate_rejected", child, parent)

    def on_merge_attempted(self, pair: tuple[Candidate, Candidate]) -> None:
        self._dispatch("on_merge_attempted", pair)

    def on_iteration_end(self, iteration: int, state: OptimizationState) -> None:
        self._dispatch("on_iteration_end", iteration, state)

    def on_optimization_end(self, result: OptimizationResult) -> None:
        self._dispatch("on_optimization_end", result)


@dataclass
class OptimizationMetrics:
    """Accumulated metrics from an optimization run."""

    accepted: int = 0
    rejected: int = 0
    merge_attempts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    best_score_history: list[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted + self.rejected
        return self.accepted / total if total > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class MetricsCallback(OptimizationCallback):
    """Built-in callback that tracks optimization metrics."""

    def __init__(self) -> None:
        self.metrics = OptimizationMetrics()

    def on_candidate_accepted(self, child: Candidate, parent: Candidate) -> None:
        self.metrics.accepted += 1

    def on_candidate_rejected(self, child: Candidate, parent: Candidate) -> None:
        self.metrics.rejected += 1

    def on_merge_attempted(self, pair: tuple[Candidate, Candidate]) -> None:
        self.metrics.merge_attempts += 1

    def on_iteration_end(self, iteration: int, state: OptimizationState) -> None:
        best = state.best_candidate()
        if best is not None:
            self.metrics.best_score_history.append(best.mean_score or 0.0)
