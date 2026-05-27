"""Routing orchestration. The :class:`Router` ties together the
experience store, embedder, and algorithm primitives to make one
selection per LLM call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .algorithm import aggregate, pareto_filter, score_utility, thompson_sample
from .embeddings import Embedder
from .models import LlmPool, RoutingDecision, Weights
from .store import ExperienceStore


@dataclass(slots=True)
class RoutingOutcome:
    """Bundle returned to the plugin. ``query_embedding`` is reused when
    persisting the resulting record, so we don't re-embed the same query
    after the call completes."""

    decision: RoutingDecision
    query_embedding: np.ndarray


class Router:
    """Pick a model from ``pool`` for one ``(agent_role, tools, query)`` tuple."""

    def __init__(
        self,
        *,
        pool: LlmPool,
        store: ExperienceStore,
        embedder: Embedder,
        weights: Weights = Weights(),
        cold_start_threshold: int = 50,
        sim_threshold: float = 0.85,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._pool = pool
        self._store = store
        self._embedder = embedder
        self._weights = weights
        self._cold_start_threshold = cold_start_threshold
        self._sim_threshold = sim_threshold
        self._rng = rng if rng is not None else np.random.default_rng()

    async def select(
        self,
        *,
        agent_role: str,
        tools: Iterable[str],
        query: str,
    ) -> RoutingOutcome:
        embedding = await self._embedder.embed(query)

        if self._store.count() < self._cold_start_threshold:
            return self._random_pick(
                embedding, retrieved=0, reason="cold_start"
            )

        records = self._store.retrieve(
            agent_role=agent_role,
            tools=tools,
            query_emb=embedding,
            sim_threshold=self._sim_threshold,
        )
        if not records:
            return self._random_pick(
                embedding, retrieved=0, reason="no_match"
            )

        stats_by_model = aggregate(records, self._pool)
        considered = pareto_filter(stats_by_model)

        utilities = {
            m: score_utility(
                *thompson_sample(stats_by_model[m], rng=self._rng),
                weights=self._weights,
            )
            for m in considered
        }
        # Multiple under-sampled models all score +inf; pick uniformly
        # among the maximum to avoid starving any one of them.
        max_u = max(utilities.values())
        top = [m for m, u in utilities.items() if u == max_u]
        chosen = top[0] if len(top) == 1 else str(self._rng.choice(top))

        return RoutingOutcome(
            decision=RoutingDecision(
                chosen_model=chosen,
                considered_models=tuple(considered),
                was_cold_start=False,
                no_relevant_history=False,
                retrieved_records=len(records),
            ),
            query_embedding=embedding,
        )

    def _random_pick(
        self,
        embedding: np.ndarray,
        *,
        retrieved: int,
        reason: str,
    ) -> RoutingOutcome:
        chosen = str(self._rng.choice(self._pool.models))
        return RoutingOutcome(
            decision=RoutingDecision(
                chosen_model=chosen,
                considered_models=self._pool.models,
                was_cold_start=reason == "cold_start",
                no_relevant_history=reason == "no_match",
                retrieved_records=retrieved,
            ),
            query_embedding=embedding,
        )
