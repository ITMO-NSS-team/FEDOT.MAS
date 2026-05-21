"""Tests for fedotmas.routing.store.SQLiteExperienceStore."""

from __future__ import annotations

import numpy as np
import pytest

from fedotmas.routing.models import ExperienceRecord
from fedotmas.routing.store import SQLiteExperienceStore


@pytest.fixture()
def store() -> SQLiteExperienceStore:
    return SQLiteExperienceStore(":memory:")


def _make_rec(
    *,
    trace_id: str = "t1",
    step_idx: int = 0,
    agent_role: str = "researcher",
    llm: str = "openai/gpt-4o",
    tools: tuple[str, ...] = ("search",),
    embedding: np.ndarray | None = None,
    success_task: float | None = None,
) -> ExperienceRecord:
    if embedding is None:
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return ExperienceRecord(
        trace_id=trace_id,
        step_idx=step_idx,
        agent_role=agent_role,
        llm_used=llm,
        query="what is the capital of France?",
        embedding=embedding,
        tools=tools,
        cost=0.01,
        duration=1.2,
        success_step=1.0,
        success_task=success_task,
    )


class TestAppendAndCount:
    def test_empty_store_count_is_zero(self, store: SQLiteExperienceStore) -> None:
        assert store.count() == 0

    def test_append_increments_count_and_sets_id(self, store: SQLiteExperienceStore) -> None:
        rec = _make_rec()
        rec_id = store.append(rec)
        assert rec_id > 0
        assert rec.id == rec_id
        assert store.count() == 1

    def test_append_sets_created_at_when_zero(self, store: SQLiteExperienceStore) -> None:
        rec = _make_rec()
        assert rec.created_at == 0
        store.append(rec)
        assert rec.created_at > 0

    def test_append_preserves_explicit_created_at(self, store: SQLiteExperienceStore) -> None:
        rec = _make_rec()
        rec.created_at = 1234567890.0
        store.append(rec)
        retrieved = store.retrieve(
            agent_role=rec.agent_role,
            tools=rec.tools,
            query_emb=rec.embedding,
        )
        assert retrieved[0].created_at == pytest.approx(1234567890.0)


class TestRoundTrip:
    def test_field_preservation(self, store: SQLiteExperienceStore) -> None:
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        rec = _make_rec(
            tools=("search", "browser"),
            embedding=emb,
            success_task=0.7,
        )
        store.append(rec)
        out = store.retrieve(
            agent_role=rec.agent_role,
            tools=rec.tools,
            query_emb=rec.embedding,
        )
        assert len(out) == 1
        got = out[0]
        assert got.trace_id == rec.trace_id
        assert got.step_idx == rec.step_idx
        assert got.agent_role == rec.agent_role
        assert got.llm_used == rec.llm_used
        assert got.query == rec.query
        assert got.tools == ("search", "browser")
        assert got.cost == pytest.approx(rec.cost)
        assert got.duration == pytest.approx(rec.duration)
        assert got.success_step == pytest.approx(rec.success_step)
        assert got.success_task == pytest.approx(0.7)
        np.testing.assert_allclose(got.embedding, emb)


class TestBackfill:
    def test_backfill_sets_only_unscored_records(self, store: SQLiteExperienceStore) -> None:
        store.append(_make_rec(trace_id="A", success_task=None))
        store.append(_make_rec(trace_id="A", step_idx=1, success_task=None))
        store.append(_make_rec(trace_id="A", step_idx=2, success_task=0.5))  # pre-scored
        store.append(_make_rec(trace_id="B"))  # different trace

        updated = store.backfill_task_success("A", 1.0)
        assert updated == 2  # the two with None

        recs = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        by_trace_step = {(r.trace_id, r.step_idx): r.success_task for r in recs}
        assert by_trace_step[("A", 0)] == pytest.approx(1.0)
        assert by_trace_step[("A", 1)] == pytest.approx(1.0)
        assert by_trace_step[("A", 2)] == pytest.approx(0.5)  # unchanged
        assert by_trace_step[("B", 0)] is None  # untouched

    def test_backfill_unknown_trace_is_noop(self, store: SQLiteExperienceStore) -> None:
        store.append(_make_rec(trace_id="A"))
        assert store.backfill_task_success("nonexistent", 1.0) == 0


class TestRetrievalUnion:
    """retrieve() returns the union of role-match ∪ tool-overlap ∪ sim>θ."""

    def test_role_match_brings_in_records_regardless_of_other_signals(
        self, store: SQLiteExperienceStore
    ) -> None:
        store.append(
            _make_rec(
                agent_role="researcher",
                tools=("nothing_in_common",),
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            sim_threshold=0.85,
        )
        assert len(out) == 1

    def test_tool_overlap_brings_in_records(self, store: SQLiteExperienceStore) -> None:
        store.append(
            _make_rec(
                agent_role="someone_else",
                tools=("search", "extra"),
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            sim_threshold=0.85,
        )
        assert len(out) == 1

    def test_semantic_similarity_brings_in_records(self, store: SQLiteExperienceStore) -> None:
        store.append(
            _make_rec(
                agent_role="someone_else",
                tools=("unrelated",),
                embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32),
            sim_threshold=0.85,
        )
        assert len(out) == 1

    def test_no_signals_excludes_record(self, store: SQLiteExperienceStore) -> None:
        store.append(
            _make_rec(
                agent_role="other_role",
                tools=("other_tool",),
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            sim_threshold=0.85,
        )
        assert out == []

    def test_high_threshold_excludes_weakly_similar(self, store: SQLiteExperienceStore) -> None:
        # cosine sim ≈ 0.707 → below 0.85
        store.append(
            _make_rec(
                agent_role="other",
                tools=("nope",),
                embedding=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            sim_threshold=0.85,
        )
        assert out == []

    def test_union_does_not_duplicate(self, store: SQLiteExperienceStore) -> None:
        # All three signals match the same record → should appear once.
        store.append(
            _make_rec(
                agent_role="researcher",
                tools=("search",),
                embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        assert len(out) == 1


class TestPersistence:
    def test_records_survive_reconnection(self, tmp_path) -> None:
        db = tmp_path / "exp.sqlite"
        s1 = SQLiteExperienceStore(db)
        s1.append(_make_rec())
        s1.close()

        s2 = SQLiteExperienceStore(db)
        assert s2.count() == 1
        s2.close()


class TestMaxPool:
    def test_retrieve_respects_max_pool_limit(
        self, store: SQLiteExperienceStore
    ) -> None:
        for i in range(20):
            store.append(_make_rec(trace_id=f"t{i}", step_idx=i))

        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_pool=5,
        )
        assert len(out) == 5

    def test_max_pool_keeps_most_recent_records(
        self, store: SQLiteExperienceStore
    ) -> None:
        for i in range(10):
            store.append(_make_rec(trace_id=f"t{i}", step_idx=i))

        out = store.retrieve(
            agent_role="researcher",
            tools=("search",),
            query_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_pool=3,
        )
        step_ids = sorted(r.step_idx for r in out)
        # ORDER BY id DESC LIMIT 3 → newest insertions
        assert step_ids == [7, 8, 9]
