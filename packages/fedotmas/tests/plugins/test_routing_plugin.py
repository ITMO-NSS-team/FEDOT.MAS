"""Tests for LLMRoutingPlugin — wires Router → ADK callbacks → store."""

from __future__ import annotations

import asyncio
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
    async def test_concurrent_agents_keyed_independently(self, pool, store):
        # Embedder with a real await suspension point so the two
        # before-hooks actually interleave under asyncio.gather rather
        # than running to completion sequentially.
        async def slow_embed(_model: str, _text: str) -> np.ndarray:
            await asyncio.sleep(0)
            return _FIXED_EMB

        plugin = _make_plugin(
            pool, store, Embedder(model="fake", embed_fn=slow_embed)
        )
        a_ctx = _make_callback_ctx(trace_id="t1", agent_name="alpha")
        b_ctx = _make_callback_ctx(trace_id="t1", agent_name="beta")
        a_req = _make_request("openai/original", query="alpha-query")
        b_req = _make_request("openai/original", query="beta-query")

        await asyncio.gather(
            plugin.before_model_callback(callback_context=a_ctx, llm_request=a_req),
            plugin.before_model_callback(callback_context=b_ctx, llm_request=b_req),
        )

        assert ("t1", "alpha") in plugin._pending
        assert ("t1", "beta") in plugin._pending

        await asyncio.gather(
            plugin.after_model_callback(callback_context=b_ctx, llm_response=_make_response()),
            plugin.after_model_callback(callback_context=a_ctx, llm_response=_make_response()),
        )

        records = store.retrieve(
            agent_role="alpha", tools=(), query_emb=_FIXED_EMB
        ) + store.retrieve(
            agent_role="beta", tools=(), query_emb=_FIXED_EMB
        )
        by_agent = {r.agent_role: r for r in records}
        assert by_agent["alpha"].query == "alpha-query"
        assert by_agent["beta"].query == "beta-query"


# ─── Re-entrancy ──────────────────────────────────────────────────────


class TestReentrancy:
    @pytest.mark.asyncio
    async def test_overwriting_pending_logs_warning(
        self, pool, store, embedder, caplog
    ):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        await plugin.before_model_callback(callback_context=ctx, llm_request=req)

        assert any(
            "Overwriting unconsumed pending routing record" in r.getMessage()
            for r in caplog.records
        )
        # Only the second pending remains; the first is lost (the
        # warning is the user-facing surface for that data loss).
        assert len(plugin._pending) == 1


# ─── after_run_callback cleanup ───────────────────────────────────────


class TestAfterRunCleanup:
    @pytest.mark.asyncio
    async def test_clears_step_counter_and_orphans(
        self, pool, store, embedder, caplog
    ):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")

        # before fires, after never does → orphan pending entry.
        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        plugin._step_counter["t1"] = 5
        assert ("t1", "researcher") in plugin._pending

        await plugin.after_run_callback(
            invocation_context=_make_invocation_ctx("t1")
        )

        assert plugin._pending == {}
        assert "t1" not in plugin._step_counter
        assert any(
            "orphan pending routing record" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_does_not_touch_other_traces(self, pool, store, embedder):
        plugin = _make_plugin(pool, store, embedder)
        ctx1 = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        ctx2 = _make_callback_ctx(trace_id="t2", agent_name="researcher")
        await plugin.before_model_callback(
            callback_context=ctx1, llm_request=_make_request("openai/original")
        )
        await plugin.before_model_callback(
            callback_context=ctx2, llm_request=_make_request("openai/original")
        )

        await plugin.after_run_callback(
            invocation_context=_make_invocation_ctx("t1")
        )

        assert ("t1", "researcher") not in plugin._pending
        assert ("t2", "researcher") in plugin._pending


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


# ─── Buffer cap ───────────────────────────────────────────────────────


class TestBufferCap:
    @pytest.mark.asyncio
    async def test_pending_evicts_oldest_at_cap(
        self, pool, store, embedder, caplog, monkeypatch
    ):
        import fedotmas.plugins._routing as routing_mod

        monkeypatch.setattr(routing_mod, "_BUFFER_CAP", 3)
        plugin = _make_plugin(pool, store, embedder)

        # 4 distinct (trace, agent) keys → cap of 3 → oldest evicted.
        for i in range(4):
            ctx = _make_callback_ctx(trace_id=f"t{i}", agent_name="r")
            await plugin.before_model_callback(
                callback_context=ctx, llm_request=_make_request("openai/original")
            )

        assert len(plugin._pending) == 3
        assert ("t0", "r") not in plugin._pending  # oldest evicted
        assert ("t3", "r") in plugin._pending
        assert any(
            "Evicted" in r.getMessage() and "pending routing entries" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_step_counter_caps_in_fifo_order(
        self, pool, store, embedder, monkeypatch
    ):
        import fedotmas.plugins._routing as routing_mod

        monkeypatch.setattr(routing_mod, "_BUFFER_CAP", 2)
        plugin = _make_plugin(pool, store, embedder)

        # Seed three step-counter entries manually, then trip the cap
        # by running before_model_callback (which calls _enforce_buffer_cap).
        plugin._step_counter["old"] = 1
        plugin._step_counter["mid"] = 1
        plugin._step_counter["new"] = 1

        ctx = _make_callback_ctx(trace_id="trip", agent_name="r")
        await plugin.before_model_callback(
            callback_context=ctx, llm_request=_make_request("openai/original")
        )

        # _BUFFER_CAP=2 means after eviction _step_counter has 2 entries.
        # FIFO via insertion order: "old" goes first.
        assert "old" not in plugin._step_counter
        assert "new" in plugin._step_counter


# ─── Reasoning tokens in cost ─────────────────────────────────────────


class TestCostIncludesReasoningTokens:
    @pytest.mark.asyncio
    async def test_thoughts_tokens_charged_at_output_rate(
        self, pool, store, embedder
    ):
        plugin = _make_plugin(pool, store, embedder)
        ctx = _make_callback_ctx(trace_id="t1", agent_name="researcher")
        req = _make_request("openai/original")

        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        chosen = req.model

        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=1_000_000,
            total_token_count=1_000_000,
        )
        await plugin.after_model_callback(
            callback_context=ctx,
            llm_response=LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="x")]),
                usage_metadata=usage,
            ),
        )

        records = store.retrieve(
            agent_role="researcher", tools=(), query_emb=_FIXED_EMB
        )
        # Reasoning tokens are billed at output_price_per_1m. With
        # 1M reasoning tokens the cost equals output_price_per_1m
        # exactly — and importantly is nonzero, where the old impl
        # would have reported 0.
        entry = pool.by_model(chosen)
        assert records[0].cost == pytest.approx(entry.output_price_per_1m)


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
