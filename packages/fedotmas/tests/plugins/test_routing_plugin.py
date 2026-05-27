"""Tests for LLMRoutingPlugin — wires Router → ADK callbacks → store."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from fedotmas.plugins import LLMRoutingPlugin
from fedotmas.routing import (
    Embedder,
    LlmPool,
    LlmPoolEntry,
    SQLiteExperienceStore,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


_FIXED_EMB = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture()
def pool() -> LlmPool:
    return LlmPool(
        entries=(
            LlmPoolEntry(
                model="openai/cheap", input_price_per_1m=1.0, output_price_per_1m=2.0
            ),
            LlmPoolEntry(
                model="openai/strong", input_price_per_1m=10.0, output_price_per_1m=20.0
            ),
        )
    )


@pytest.fixture()
def store() -> SQLiteExperienceStore:
    return SQLiteExperienceStore(":memory:")


@pytest.fixture()
def embedder() -> Embedder:
    async def fake(_model: str, _text: str) -> np.ndarray:
        return _FIXED_EMB

    return Embedder(model="fake", embed_fn=fake)


def _make_plugin(
    pool: LlmPool,
    store: SQLiteExperienceStore,
    embedder: Embedder,
    *,
    rng_seed: int = 0,
) -> LLMRoutingPlugin:
    # Force cold-start so the deterministic rng picks the first pool entry.
    return LLMRoutingPlugin(
        pool=pool,
        store=store,
        embedder=embedder,
        cold_start_threshold=10_000,
        rng=np.random.default_rng(rng_seed),
    )


def _make_callback_ctx(*, trace_id: str, agent_name: str) -> MagicMock:
    ctx = MagicMock()
    ctx._invocation_context = SimpleNamespace(
        invocation_id=trace_id,
        agent=SimpleNamespace(name=agent_name),
    )
    return ctx


def _make_invocation_ctx(trace_id: str, agent_name: str = "root") -> MagicMock:
    return SimpleNamespace(
        invocation_id=trace_id,
        agent=SimpleNamespace(name=agent_name),
    )


def _make_request(model: str, *, query: str = "what is 2+2?") -> LlmRequest:
    return LlmRequest(
        model=model,
        contents=[
            types.Content(role="user", parts=[types.Part(text=query)])
        ],
    )


def _make_response(*, prompt_tokens: int = 100, completion_tokens: int = 50) -> LlmResponse:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=prompt_tokens + completion_tokens,
    )
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="4")]),
        usage_metadata=usage,
    )


# ─── before_model_callback ────────────────────────────────────────────


class TestBeforeModelCallback:
    @pytest.mark.asyncio
    async def test_mutates_model_to_router_choice(self, pool, store, embedder):
        # rng_seed=0 with cold-start over a 2-model pool: choice is deterministic.
        plugin = _make_plugin(pool, store, embedder, rng_seed=0)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)

        assert req.model in {"openai/cheap", "openai/strong"}
        assert req.model != "openai/original"

    @pytest.mark.asyncio
    async def test_skips_workflow_nodes(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        req = _make_request("openai/original")

        for prefix in ("seq_x", "par_y", "loop_z"):
            ctx = _make_callback_ctx(trace_id="t1", agent_name=prefix)
            await plugin.before_model_callback(callback_context=ctx, llm_request=req)
            assert req.model == "openai/original"

    @pytest.mark.asyncio
    async def test_router_failure_falls_through(self, pool, store):
        async def failing_embed(_model: str, _text: str) -> np.ndarray:
            raise RuntimeError("network down")

        plugin = _make_plugin(
            pool, store, Embedder(model="fake", embed_fn=failing_embed)
        )
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)

        # On failure, original model is preserved and nothing gets stashed.
        assert req.model == "openai/original"
        assert plugin._pending == {}


# ─── after_model_callback ─────────────────────────────────────────────


class TestAfterModelCallback:
    @pytest.mark.asyncio
    async def test_appends_record_with_correct_fields(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t-abc", agent_name="researcher")
        req = _make_request("openai/original", query="explain entropy")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        chosen = req.model
        await plugin.after_model_callback(
            callback_context=ctx, llm_response=_make_response()
        )

        assert store.count() == 1
        records = store.retrieve(
            agent_role="researcher", tools=(), query_emb=_FIXED_EMB
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.trace_id == "t-abc"
        assert rec.agent_role == "researcher"
        assert rec.llm_used == chosen
        assert rec.query == "explain entropy"
        assert rec.success_step == 1.0
        assert rec.success_task is None  # backfilled later
        assert rec.cost > 0
        assert rec.duration >= 0

    @pytest.mark.asyncio
    async def test_cost_uses_pool_pricing(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        chosen = req.model
        await plugin.after_model_callback(
            callback_context=ctx,
            llm_response=_make_response(prompt_tokens=1_000_000, completion_tokens=0),
        )

        records = store.retrieve(
            agent_role="researcher", tools=(), query_emb=_FIXED_EMB
        )
        entry = pool.by_model(chosen)
        assert records[0].cost == pytest.approx(entry.input_price_per_1m)

    @pytest.mark.asyncio
    async def test_step_idx_increments_per_trace(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        await plugin.before_run_callback(
            invocation_context=_make_invocation_ctx("t-multi")
        )

        for _ in range(3):
            ctx = _make_callback_ctx(trace_id="t-multi", agent_name="researcher")
            req = _make_request("openai/original")
            await plugin.before_model_callback(callback_context=ctx, llm_request=req)
            await plugin.after_model_callback(
                callback_context=ctx, llm_response=_make_response()
            )

        records = sorted(
            store.retrieve(agent_role="researcher", tools=(), query_emb=_FIXED_EMB),
            key=lambda r: r.id,
        )
        assert [r.step_idx for r in records] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_no_record_when_before_was_skipped(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="seq_main")
        await plugin.after_model_callback(
            callback_context=ctx, llm_response=_make_response()
        )
        assert store.count() == 0


# ─── on_model_error_callback ──────────────────────────────────────────


class TestOnModelErrorCallback:
    @pytest.mark.asyncio
    async def test_appends_record_with_success_step_zero(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t-err", agent_name="researcher")
        req = _make_request("openai/original")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        await plugin.on_model_error_callback(
            callback_context=ctx, llm_request=req, error=RuntimeError("boom")
        )

        records = store.retrieve(
            agent_role="researcher", tools=(), query_emb=_FIXED_EMB
        )
        assert len(records) == 1
        assert records[0].success_step == 0.0
        assert records[0].cost == 0.0  # no usage on error


# ─── Concurrency ──────────────────────────────────────────────────────


class TestParallelAgents:
    @pytest.mark.asyncio
    async def test_per_trace_per_agent_keying(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        a_ctx = _make_callback_ctx(trace_id="t1", agent_name="alpha")
        b_ctx = _make_callback_ctx(trace_id="t1", agent_name="beta")
        a_req = _make_request("openai/original", query="alpha-query")
        b_req = _make_request("openai/original", query="beta-query")

        # Two before-hooks interleave before either after-hook fires.
        await plugin.before_model_callback(callback_context=a_ctx, llm_request=a_req)
        await plugin.before_model_callback(callback_context=b_ctx, llm_request=b_req)

        # Both pending slots populated, keyed independently.
        assert ("t1", "alpha") in plugin._pending
        assert ("t1", "beta") in plugin._pending

        # After-hooks complete in reverse order.
        await plugin.after_model_callback(
            callback_context=b_ctx, llm_response=_make_response()
        )
        await plugin.after_model_callback(
            callback_context=a_ctx, llm_response=_make_response()
        )

        records = store.retrieve(
            agent_role="alpha", tools=(), query_emb=_FIXED_EMB
        ) + store.retrieve(
            agent_role="beta", tools=(), query_emb=_FIXED_EMB
        )
        by_agent = {r.agent_role: r for r in records}
        assert by_agent["alpha"].query == "alpha-query"
        assert by_agent["beta"].query == "beta-query"


# ─── commit_task_score ────────────────────────────────────────────────


class TestCommitTaskScore:
    @pytest.mark.asyncio
    async def test_backfills_all_records_for_trace(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        for _ in range(2):
            ctx = _make_callback_ctx(trace_id="t-score", agent_name="researcher")
            req = _make_request("openai/original")
            await plugin.before_model_callback(callback_context=ctx, llm_request=req)
            await plugin.after_model_callback(
                callback_context=ctx, llm_response=_make_response()
            )

        n = plugin.commit_task_score("t-score", 0.75)
        assert n == 2
        records = store.retrieve(
            agent_role="researcher", tools=(), query_emb=_FIXED_EMB
        )
        assert all(r.success_task == 0.75 for r in records)

    @pytest.mark.asyncio
    async def test_unknown_trace_id_returns_zero(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        assert plugin.commit_task_score("never-seen", 1.0) == 0


# ─── Tools extraction ─────────────────────────────────────────────────


class TestToolsExtraction:
    @pytest.mark.asyncio
    async def test_tools_dict_keys_are_persisted(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")
        tool_a = MagicMock()
        tool_b = MagicMock()
        req.tools_dict = {"search": tool_a, "fetch": tool_b}

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        await plugin.after_model_callback(
            callback_context=ctx, llm_response=_make_response()
        )

        records = store.retrieve(
            agent_role="researcher", tools=("search",), query_emb=_FIXED_EMB
        )
        assert records[0].tools == ("fetch", "search")  # sorted
