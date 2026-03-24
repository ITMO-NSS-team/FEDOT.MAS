from __future__ import annotations

import random
from enum import Enum
from typing import Protocol, runtime_checkable

from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._state import Candidate


class MutationType(Enum):
    INSTRUCTION = "instruction"
    TOOL = "tool"
    MODEL = "model"
    STRUCTURE = "structure"


@runtime_checkable
class CandidateSelector(Protocol):
    def select(self, candidates: list[Candidate]) -> Candidate: ...


class BestCandidateSelector:
    def select(self, candidates: list[Candidate]) -> Candidate:
        return max(candidates, key=lambda c: c.mean_score or 0.0)


class ParetoCandidateSelector:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def select(self, candidates: list[Candidate]) -> Candidate:
        pareto = [c for c in candidates if c.on_pareto_front]
        if not pareto:
            pareto = candidates
        return self._rng.choice(pareto)


class EpsilonGreedySelector:
    def __init__(self, epsilon: float = 0.1, rng: random.Random | None = None) -> None:
        self._epsilon = epsilon
        self._rng = rng or random.Random()

    def select(self, candidates: list[Candidate]) -> Candidate:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(candidates)
        return max(candidates, key=lambda c: c.mean_score or 0.0)


@runtime_checkable
class BatchSampler(Protocol):
    def sample(self, tasks: list[str], batch_size: int) -> list[str]: ...


class ShuffledBatchSampler:
    """Random sample each iteration (no coverage guarantee)."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def sample(self, tasks: list[str], batch_size: int) -> list[str]:
        k = min(batch_size, len(tasks))
        return self._rng.sample(tasks, k)


class EpochShuffledBatchSampler:
    """Sequential minibatches from a shuffled trainset.

    Guarantees full coverage of all tasks per epoch (GEPA default).
    When the epoch is exhausted, reshuffles for the next epoch.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._shuffled: list[str] = []
        self._position: int = 0
        self._last_size: int = 0

    def sample(self, tasks: list[str], batch_size: int) -> list[str]:
        if not tasks:
            raise ValueError("Cannot sample from empty task list")

        needs_refresh = len(tasks) != self._last_size or self._position >= len(
            self._shuffled
        )
        if needs_refresh:
            self._shuffled = list(tasks)
            self._rng.shuffle(self._shuffled)
            self._last_size = len(tasks)
            self._position = 0
            # Pad to a multiple of batch_size with least-seen items
            mod = len(self._shuffled) % batch_size
            if mod:
                pad = batch_size - mod
                for i in range(pad):
                    self._shuffled.append(self._shuffled[i % len(tasks)])

        result = self._shuffled[self._position : self._position + batch_size]
        self._position += batch_size
        return result


@runtime_checkable
class ComponentSelector(Protocol):
    def select(self, config: MAWConfig, iteration: int) -> list[str]: ...


class AllComponentSelector:
    def select(self, config: MAWConfig, iteration: int) -> list[str]:
        return [a.name for a in config.agents]


class RoundRobinComponentSelector:
    def select(self, config: MAWConfig, iteration: int) -> list[str]:
        agent_names = [a.name for a in config.agents]
        if not agent_names:
            return []
        idx = iteration % len(agent_names)
        return [agent_names[idx]]


def make_candidate_selector(
    name: str, rng: random.Random | None = None
) -> CandidateSelector:
    if name == "pareto":
        return ParetoCandidateSelector(rng=rng)
    if name == "best":
        return BestCandidateSelector()
    if name == "epsilon_greedy":
        return EpsilonGreedySelector(rng=rng)
    raise ValueError(
        f"Unknown candidate selector '{name}'. "
        f"Available: pareto, best, epsilon_greedy"
    )


def make_batch_sampler(name: str, rng: random.Random | None = None) -> BatchSampler:
    if name == "epoch_shuffled":
        return EpochShuffledBatchSampler(rng=rng)
    if name == "random":
        return ShuffledBatchSampler(rng=rng)
    raise ValueError(
        f"Unknown batch strategy '{name}'. " f"Available: epoch_shuffled, random"
    )


def make_component_selector(num_agents: int) -> ComponentSelector:
    if num_agents <= 3:
        return AllComponentSelector()
    return RoundRobinComponentSelector()
