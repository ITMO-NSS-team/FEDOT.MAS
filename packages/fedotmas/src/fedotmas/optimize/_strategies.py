from __future__ import annotations

import random
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
    def select(self, candidates: list[Candidate]) -> Candidate:
        pareto = [c for c in candidates if c.on_pareto_front]
        if not pareto:
            pareto = candidates
        return random.choice(pareto)


class EpsilonGreedySelector:
    def __init__(self, epsilon: float = 0.1) -> None:
        self._epsilon = epsilon

    def select(self, candidates: list[Candidate]) -> Candidate:
        if random.random() < self._epsilon:
            return random.choice(candidates)
        return max(candidates, key=lambda c: c.mean_score or 0.0)


@runtime_checkable
class BatchSampler(Protocol):
    def sample(self, tasks: list[str], batch_size: int) -> list[str]: ...


class ShuffledBatchSampler:
    def sample(self, tasks: list[str], batch_size: int) -> list[str]:
        k = min(batch_size, len(tasks))
        return random.sample(tasks, k)


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


def make_candidate_selector(name: str) -> CandidateSelector:
    selectors: dict[str, type] = {
        "pareto": ParetoCandidateSelector,
        "best": BestCandidateSelector,
        "epsilon_greedy": EpsilonGreedySelector,
    }
    cls = selectors.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown candidate selector '{name}'. "
            f"Available: {', '.join(selectors)}"
        )
    return cls()


def make_component_selector(num_agents: int) -> ComponentSelector:
    if num_agents <= 3:
        return AllComponentSelector()
    return RoundRobinComponentSelector()
