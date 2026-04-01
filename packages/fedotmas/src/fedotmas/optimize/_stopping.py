from __future__ import annotations

import signal
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
        return state.total_evaluations >= self._max


class NoImprovement:
    def __init__(self, patience: int, epsilon: float = 1e-6) -> None:
        self._patience = patience
        self._epsilon = epsilon
        self._best_score = -1.0
        self._stale_count = 0

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        best = state.best_candidate()
        if best is None:
            return False
        current = best.mean_score or 0.0
        if current > self._best_score + self._epsilon:
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


class SignalStopper:
    """Catches SIGINT/SIGTERM and finishes the current iteration gracefully."""

    def __init__(self) -> None:
        self._triggered = False
        self._prev_sigint = None
        self._prev_sigterm = None

    def __enter__(self) -> SignalStopper:
        self.install()
        return self

    def __exit__(self, *exc: object) -> None:
        self.uninstall()

    def install(self) -> None:
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def uninstall(self) -> None:
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)

    def _handle(self, signum: int, frame: object) -> None:
        self._triggered = True

    def should_stop(self, state: OptimizationState, iteration: int) -> bool:
        return self._triggered

    def reset(self) -> None:
        self._triggered = False
