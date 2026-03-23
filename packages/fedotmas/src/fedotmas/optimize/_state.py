from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fedotmas.maw.models import MAWConfig


@dataclass
class TaskResult:
    task: str
    state: dict[str, Any]
    score: float
    feedback: str
    error: bool = False


@dataclass
class Candidate:
    index: int
    config: MAWConfig
    config_hash: str
    scores: dict[str, float] = field(default_factory=dict)
    feedbacks: dict[str, str] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    parent_index: int | None = None
    origin: str = "seed"
    on_pareto_front: bool = False
    merge_parent_indices: tuple[int, int] | None = None

    @property
    def mean_score(self) -> float | None:
        if not self.scores:
            return None
        return sum(self.scores.values()) / len(self.scores)

    @property
    def min_score(self) -> float | None:
        if not self.scores:
            return None
        return min(self.scores.values())


def config_hash(config: MAWConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:32]


class EvaluationCache:
    def __init__(self, *, max_size: int | None = None) -> None:
        self._cache: OrderedDict[tuple[str, str], TaskResult] = OrderedDict()
        self._max_size = max_size

    def get(self, cfg_hash: str, task: str) -> TaskResult | None:
        return self._cache.get((cfg_hash, task))

    def put(self, cfg_hash: str, task: str, result: TaskResult) -> None:
        key = (cfg_hash, task)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result
        if self._max_size is not None and len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


class OptimizationState:
    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self.cache = EvaluationCache()
        self._next_index = 0
        self.total_evaluations: int = 0
        self.iteration: int = 0

    def add_candidate(
        self,
        config: MAWConfig,
        *,
        parent_index: int | None = None,
        merge_parent_indices: tuple[int, int] | None = None,
        origin: str = "seed",
    ) -> Candidate:
        c = Candidate(
            index=self._next_index,
            config=config,
            config_hash=config_hash(config),
            parent_index=parent_index,
            origin=origin,
            merge_parent_indices=merge_parent_indices,
        )
        self.candidates.append(c)
        self._next_index += 1
        return c

    def record_task_result(self, candidate: Candidate, result: TaskResult) -> None:
        candidate.scores[result.task] = result.score
        candidate.feedbacks[result.task] = result.feedback
        candidate.states[result.task] = result.state
        self.cache.put(candidate.config_hash, result.task, result)
        self.total_evaluations += 1

    def update_pareto_front(self) -> None:
        evaluated = [c for c in self.candidates if c.scores]
        if not evaluated:
            return

        for c in evaluated:
            c.on_pareto_front = True

        for c in evaluated:
            for other in evaluated:
                if other is c:
                    continue
                if _dominates(other, c):
                    c.on_pareto_front = False
                    break

    def get_pareto_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.on_pareto_front]

    def best_candidate(self) -> Candidate | None:
        evaluated = [c for c in self.candidates if c.scores]
        if not evaluated:
            return None
        return max(evaluated, key=lambda c: c.mean_score or 0.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "next_index": self._next_index,
            "total_evaluations": self.total_evaluations,
            "iteration": self.iteration,
            "candidates": [
                {
                    "index": c.index,
                    "config": json.loads(c.config.model_dump_json()),
                    "config_hash": c.config_hash,
                    "scores": c.scores,
                    "feedbacks": c.feedbacks,
                    "states": c.states,
                    "parent_index": c.parent_index,
                    "origin": c.origin,
                    "on_pareto_front": c.on_pareto_front,
                    "merge_parent_indices": list(c.merge_parent_indices)
                    if c.merge_parent_indices
                    else None,
                }
                for c in self.candidates
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> OptimizationState:
        path = Path(path)
        data = json.loads(path.read_text())
        state = cls()
        state._next_index = data["next_index"]
        state.total_evaluations = data.get("total_evaluations", 0)
        state.iteration = data.get("iteration", 0)
        for cd in data["candidates"]:
            config = MAWConfig.model_validate(cd["config"])
            mpi = cd.get("merge_parent_indices")
            c = Candidate(
                index=cd["index"],
                config=config,
                config_hash=cd["config_hash"],
                scores=cd["scores"],
                feedbacks=cd["feedbacks"],
                states=cd["states"],
                parent_index=cd["parent_index"],
                origin=cd["origin"],
                on_pareto_front=cd["on_pareto_front"],
                merge_parent_indices=tuple(mpi) if mpi else None,
            )
            state.candidates.append(c)
            for task, score_val in c.scores.items():
                result = TaskResult(
                    task=task,
                    state=c.states.get(task, {}),
                    score=score_val,
                    feedback=c.feedbacks.get(task, ""),
                )
                state.cache.put(c.config_hash, task, result)
        return state


def find_common_ancestor(
    candidate_a: Candidate,
    candidate_b: Candidate,
    candidates: list[Candidate],
) -> Candidate | None:
    """Find the lowest common ancestor of two candidates via parent_index chains."""
    index_map = {c.index: c for c in candidates}

    def _ancestors(c: Candidate) -> list[int]:
        chain: list[int] = [c.index]
        cur = c
        while cur.parent_index is not None:
            chain.append(cur.parent_index)
            cur = index_map.get(cur.parent_index)
            if cur is None:
                break
        return chain

    ancestors_a = _ancestors(candidate_a)
    ancestors_b_set = set(_ancestors(candidate_b))

    for idx in ancestors_a:
        if idx in ancestors_b_set:
            return index_map.get(idx)
    return None


def _dominates(a: Candidate, b: Candidate) -> bool:
    """Compares only on intersection of tasks. Empty intersection → no domination."""
    common_tasks = a.scores.keys() & b.scores.keys()
    if not common_tasks:
        return False
    at_least_one_better = False
    for task in common_tasks:
        sa = a.scores[task]
        sb = b.scores[task]
        if sa < sb:
            return False
        if sa > sb:
            at_least_one_better = True
    return at_least_one_better
