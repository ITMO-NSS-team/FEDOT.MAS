"""Experience store: append-only persistence of step-level routing
records and retrieval by (agent role ∪ semantic similarity ∪ tool overlap).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .models import ExperienceRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id     TEXT    NOT NULL,
    step_idx     INTEGER NOT NULL,
    agent_role   TEXT    NOT NULL,
    llm_used     TEXT    NOT NULL,
    query        TEXT    NOT NULL,
    embedding    BLOB    NOT NULL,
    tools        TEXT    NOT NULL,
    cost         REAL    NOT NULL,
    duration     REAL    NOT NULL,
    success_step REAL    NOT NULL,
    success_task REAL,
    created_at   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_agent_role ON records(agent_role);
CREATE INDEX IF NOT EXISTS idx_records_trace_id   ON records(trace_id);
"""


DEFAULT_MAX_POOL = 10_000


class ExperienceStore(Protocol):
    def append(self, rec: ExperienceRecord) -> int: ...
    def backfill_task_success(self, trace_id: str, score: float) -> int: ...
    def retrieve(
        self,
        *,
        agent_role: str,
        tools: Iterable[str],
        query_emb: np.ndarray,
        sim_threshold: float = 0.85,
        max_pool: int = DEFAULT_MAX_POOL,
    ) -> list[ExperienceRecord]: ...
    def count(self) -> int: ...


def _row_to_record(row: sqlite3.Row) -> ExperienceRecord:
    return ExperienceRecord(
        id=row["id"],
        trace_id=row["trace_id"],
        step_idx=row["step_idx"],
        agent_role=row["agent_role"],
        llm_used=row["llm_used"],
        query=row["query"],
        embedding=np.frombuffer(row["embedding"], dtype=np.float32),
        tools=tuple(json.loads(row["tools"])),
        cost=row["cost"],
        duration=row["duration"],
        success_step=row["success_step"],
        success_task=row["success_task"],
        created_at=row["created_at"],
    )


class SQLiteExperienceStore:
    """SQLite-backed experience store. Safe for use from multiple threads;
    ``check_same_thread=False`` + a per-instance lock serialises writes.

    Passing ``path=":memory:"`` keeps the store in process memory — useful
    for unit tests.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ─── writes ───────────────────────────────────────────────────

    def append(self, rec: ExperienceRecord) -> int:
        emb_bytes = np.ascontiguousarray(rec.embedding, dtype=np.float32).tobytes()
        tools_json = json.dumps(list(rec.tools))
        created = rec.created_at if rec.created_at > 0 else time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO records (trace_id, step_idx, agent_role, llm_used, "
                "query, embedding, tools, cost, duration, success_step, "
                "success_task, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.trace_id,
                    rec.step_idx,
                    rec.agent_role,
                    rec.llm_used,
                    rec.query,
                    emb_bytes,
                    tools_json,
                    rec.cost,
                    rec.duration,
                    rec.success_step,
                    rec.success_task,
                    created,
                ),
            )
            self._conn.commit()
            new_id = int(cur.lastrowid or 0)
        rec.id = new_id
        rec.created_at = created
        return new_id

    def backfill_task_success(self, trace_id: str, score: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE records SET success_task = ? "
                "WHERE trace_id = ? AND success_task IS NULL",
                (score, trace_id),
            )
            self._conn.commit()
            return cur.rowcount

    # ─── reads ────────────────────────────────────────────────────

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM records"
            ).fetchone()
        return int(row["n"])

    def retrieve(
        self,
        *,
        agent_role: str,
        tools: Iterable[str],
        query_emb: np.ndarray,
        sim_threshold: float = 0.85,
        max_pool: int = DEFAULT_MAX_POOL,
    ) -> list[ExperienceRecord]:
        tools_set = set(tools)
        q = np.asarray(query_emb, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        # Pre-normalize the query once; per-row we still need to normalize
        # the row embedding, but this halves the norm computations per row.
        q_unit = q / q_norm if q_norm > 0.0 else None

        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM records ORDER BY id DESC LIMIT ?",
                (max_pool,),
            )
            rows = cur.fetchall()

        out: list[ExperienceRecord] = []
        for row in rows:
            match_role = row["agent_role"] == agent_role
            row_tools = set(json.loads(row["tools"]))
            match_tools = bool(tools_set & row_tools)
            if match_role or match_tools:
                out.append(_row_to_record(row))
                continue
            if q_unit is None:
                continue
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_norm = float(np.linalg.norm(emb))
            if emb_norm == 0.0:
                continue
            sim = float(np.dot(emb, q_unit) / emb_norm)
            if sim >= sim_threshold:
                out.append(_row_to_record(row))
        return out
