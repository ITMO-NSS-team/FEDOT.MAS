from __future__ import annotations

from typing import Protocol, runtime_checkable

from fedotmas.optimize._state import OptimizationState


@runtime_checkable
class Stopper(Protocol):
    def should_stop(self, state: OptimizationState, iteration: int) -> bool: ...


class MaxIterations:
    def __init__(self, max_iter: int) -> None:
        self._max = max_iter

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        return iteration >= self._max


class MaxEvaluations:
    def __init__(self, max_evals: int) -> None:
        self._max = max_evals

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        return len(state.cache) >= self._max


class NoImprovement:
    def __init__(self, patience: int) -> None:
        self._patience = patience
        self._best_score = -1.0
        self._stale_count = 0

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        best = state.best_candidate()
        if best is None:
            return False
        current = best.mean_score or 0.0
        if current > self._best_score + 1e-6:
            self._best_score = current
            self._stale_count = 0
        else:
            self._stale_count += 1
        return self._stale_count >= self._patience

    def reset(self) -> None:
        self._best_score = -1.0
        self._stale_count = 0


class ScoreThreshold:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        best = state.best_candidate()
        if best is None:
            return False
        return (best.mean_score or 0.0) >= self._threshold


class CompositeStopper:
    def __init__(self, stoppers: list[Stopper]) -> None:
        self._stoppers = stoppers

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        return any(s.should_stop(state, iteration) for s in self._stoppers)

    def reset(self) -> None:
        for s in self._stoppers:
            reset_fn = getattr(s, "reset", None)
            if reset_fn is not None:
                reset_fn()
