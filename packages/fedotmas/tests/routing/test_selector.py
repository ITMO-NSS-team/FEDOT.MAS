"""Tests for fedotmas.routing.selector.Router."""

from __future__ import annotations

import numpy as np
import pytest

from fedotmas.routing.embeddings import Embedder
from fedotmas.routing.models import (
    ExperienceRecord,
    LlmPool,
    LlmPoolEntry,
    Weights,
)
from fedotmas.routing.selector import Router
from fedotmas.routing.store import SQLiteExperienceStore


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def pool() -> LlmPool:
    return LlmPool(
        entries=(
            LlmPoolEntry(model="a", input_price_per_1m=1.0, output_price_per_1m=1.0),
            LlmPoolEntry(model="b", input_price_per_1m=1.0, output_price_per_1m=1.0),
            LlmPoolEntry(model="c", input_price_per_1m=1.0, output_price_per_1m=1.0),
        )
    )


@pytest.fixture()
def store() -> SQLiteExperienceStore:
    return SQLiteExperienceStore(":memory:")


_FIXED_EMB = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture()
def embedder() -> Embedder:
    async def fake(_model: str, _text: str) -> np.ndarray:
        return _FIXED_EMB

    return Embedder(model="fake", embed_fn=fake)


def _seed(
    store: SQLiteExperienceStore,
    *,
    llm: str,
    n: int,
    success_step: float = 1.0,
    success_task: float = 1.0,
    cost: float = 0.01,
    duration: float = 1.0,
    agent_role: str = "researcher",
    tools: tuple[str, ...] = ("search",),
) -> None:
    for i in range(n):
        store.append(
            ExperienceRecord(
                trace_id=f"t-{llm}-{i}",
                step_idx=0,
                agent_role=agent_role,
                llm_used=llm,
                query="seed query",
                embedding=_FIXED_EMB,
                tools=tools,
                cost=cost,
                duration=duration,
                success_step=success_step,
                success_task=success_task,
            )
        )


# ─── Cold start ───────────────────────────────────────────────────────


class TestColdStart:
    @pytest.mark.asyncio
    async def test_under_threshold_returns_random_from_pool(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        router = Router(
            pool=pool,
            store=store,
            embedder=embedder,
            cold_start_threshold=50,
            rng=np.random.default_rng(0),
        )
        outcome = await router.select(
            agent_role="researcher", tools=("search",), query="hello"
        )
        assert outcome.decision.was_cold_start is True
        assert outcome.decision.no_relevant_history is False
        assert outcome.decision.chosen_model in pool.models
        assert outcome.decision.retrieved_records == 0

    @pytest.mark.asyncio
    async def test_uniform_over_pool_at_zero_threshold(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        router = Router(
            pool=pool,
            store=store,
            embedder=embedder,
            cold_start_threshold=50,
            rng=np.random.default_rng(123),
        )
        counts = {m: 0 for m in pool.models}
        for _ in range(900):
            o = await router.select(
                agent_role="r", tools=("t",), query="q"
            )
            counts[o.decision.chosen_model] += 1
        # ~300 per model for uniform draws — allow loose bounds
        for n in counts.values():
            assert 200 < n < 400

    @pytest.mark.asyncio
    async def test_local_cold_start_when_no_records_retrieved(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        # Above threshold globally, but no record matches role/tools/sim.
        _seed(
            store, llm="a", n=60,
            agent_role="other_role",
            tools=("other_tool",),
        )
        # Make the seeded embeddings orthogonal so sim falls below threshold.
        # (We can't easily override embedding in _seed for this — the fixed
        # _FIXED_EMB matches the query exactly. So instead use mismatched
        # role+tools AND a non-matching query embedding fixture.)
        orthogonal_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        async def ortho_fake(_m: str, _t: str) -> np.ndarray:
            return orthogonal_emb

        router = Router(
            pool=pool,
            store=store,
            embedder=Embedder(model="fake", embed_fn=ortho_fake),
            cold_start_threshold=10,
            rng=np.random.default_rng(0),
        )
        outcome = await router.select(
            agent_role="researcher",
            tools=("search",),
            query="anything",
        )
        # Retrieve would return nothing → no_relevant_history path
        # (distinct from global cold start which is about total store size).
        assert outcome.decision.was_cold_start is False
        assert outcome.decision.no_relevant_history is True
        assert outcome.decision.retrieved_records == 0


# ─── Happy path ───────────────────────────────────────────────────────


class TestSelection:
    @pytest.mark.asyncio
    async def test_dominant_model_wins_after_enough_evidence(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        # a: high perf, low cost   → dominant
        # b: low  perf, high cost  → dominated
        # c: middling              → dominated by a
        _seed(store, llm="a", n=20, success_task=0.95, cost=0.005, duration=0.5)
        _seed(store, llm="b", n=20, success_task=0.40, cost=0.50, duration=5.0)
        _seed(store, llm="c", n=20, success_task=0.60, cost=0.10, duration=1.0)

        router = Router(
            pool=pool,
            store=store,
            embedder=embedder,
            cold_start_threshold=10,
            rng=np.random.default_rng(0),
        )
        # Across many draws "a" should win the vast majority.
        wins: dict[str, int] = {m: 0 for m in pool.models}
        for _ in range(100):
            o = await router.select(
                agent_role="researcher",
                tools=("search",),
                query="seed query",
            )
            wins[o.decision.chosen_model] += 1
        assert wins["a"] >= 90

    @pytest.mark.asyncio
    async def test_under_explored_model_is_forced_into_rotation(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        # "a" has tons of evidence; "b" has just one record; "c" has none.
        _seed(store, llm="a", n=50, success_task=0.95, cost=0.005)
        _seed(store, llm="b", n=1, success_task=0.40, cost=0.50)

        router = Router(
            pool=pool,
            store=store,
            embedder=embedder,
            cold_start_threshold=10,
            rng=np.random.default_rng(0),
        )
        wins: dict[str, int] = {m: 0 for m in pool.models}
        for _ in range(60):
            o = await router.select(
                agent_role="researcher",
                tools=("search",),
                query="seed query",
            )
            wins[o.decision.chosen_model] += 1

        # Both b (n=1, under MIN_N_FOR_INFERENCE) and c (n=0) should each
        # get explored at least sometimes — the explore-sentinel forces
        # them into the argmax tie-break alongside a.
        assert wins["b"] > 0
        assert wins["c"] > 0
        # And cumulative under-explored wins should be a meaningful share.
        assert wins["b"] + wins["c"] >= 10

    @pytest.mark.asyncio
    async def test_decision_carries_embedding(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        router = Router(
            pool=pool,
            store=store,
            embedder=embedder,
            cold_start_threshold=10,
            rng=np.random.default_rng(0),
        )
        outcome = await router.select(
            agent_role="r", tools=("t",), query="q"
        )
        np.testing.assert_array_equal(outcome.query_embedding, _FIXED_EMB)


# ─── Weights influence ────────────────────────────────────────────────


class TestWeights:
    @pytest.mark.asyncio
    async def test_cost_heavy_weights_prefer_cheap_model(
        self,
        pool: LlmPool,
        store: SQLiteExperienceStore,
        embedder: Embedder,
    ) -> None:
        # "a" is best on perf but very expensive; "b" is OK and cheap.
        # With huge cost weight, "b" should win.
        _seed(store, llm="a", n=20, success_task=0.95, cost=10.0)
        _seed(store, llm="b", n=20, success_task=0.80, cost=0.01)
        _seed(store, llm="c", n=20, success_task=0.50, cost=5.0)

        # Cost and delay are now min-max normalised to [0,1] in
        # aggregate(), so a heavy cost weight on the same order as perf
        # is enough to flip the choice — no more orders-of-magnitude
        # weight hacking required.
        router = Router(
            pool=pool,
            store=store,
            embedder=embedder,
            weights=Weights(perf=1.0, cost=5.0, delay=0.0),
            cold_start_threshold=10,
            rng=np.random.default_rng(0),
        )
        wins: dict[str, int] = {m: 0 for m in pool.models}
        for _ in range(60):
            o = await router.select(
                agent_role="researcher",
                tools=("search",),
                query="seed query",
            )
            wins[o.decision.chosen_model] += 1
        assert wins["b"] >= 50
