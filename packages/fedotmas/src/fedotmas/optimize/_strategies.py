from __future__ import annotations

import random as _random_module
from typing import Protocol, runtime_checkable

from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._state import Candidate


@runtime_checkable
class CandidateSelector(Protocol):
    def select(self, candidates: list[Candidate]) -> Candidate: ...


class BestCandidateSelector:
    def select(self, candidates: list[Candidate]) -> Candidate:
        return max(candidates, key=lambda c: c.mean_score or 0.0)


class ParetoCandidateSelector:
    def __init__(self, rng: _random_module.Random | None = None) -> None:
        self._rng = rng or _random_module.Random()

    def select(self, candidates: list[Candidate]) -> Candidate:
        pareto = [c for c in candidates if c.on_pareto_front]
        if not pareto:
            pareto = candidates
        return self._rng.choice(pareto)


class EpsilonGreedySelector:
    def __init__(
        self, epsilon: float = 0.1, rng: _random_module.Random | None = None
    ) -> None:
        self._epsilon = epsilon
        self._rng = rng or _random_module.Random()

    def select(self, candidates: list[Candidate]) -> Candidate:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(candidates)
        return max(candidates, key=lambda c: c.mean_score or 0.0)


@runtime_checkable
class BatchSampler(Protocol):
    def sample(self, tasks: list[str], batch_size: int) -> list[str]: ...


class ShuffledBatchSampler:
    def __init__(self, rng: _random_module.Random | None = None) -> None:
        self._rng = rng or _random_module.Random()

    def sample(self, tasks: list[str], batch_size: int) -> list[str]:
        k = min(batch_size, len(tasks))
        return self._rng.sample(tasks, k)


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
    name: str, rng: _random_module.Random | None = None
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


def make_component_selector(num_agents: int) -> ComponentSelector:
    if num_agents <= 3:
        return AllComponentSelector()
    return RoundRobinComponentSelector()
