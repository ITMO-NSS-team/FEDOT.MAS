"""Plugin that wires the routing module into FEDOT.MAS pipelines.

For each LLM call the plugin asks the :class:`Router` to pick a model
from the user-supplied pool, mutates ``llm_request.model`` in place
(see ``test_llm_request_mutation.py`` for the regression guard), and
appends a step-level experience record. Task-level success is filled
in later via :meth:`LLMRoutingPlugin.commit_task_score`, which the
benchmark runner calls after scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.genai import types

from fedotmas.common.logging import get_logger
from fedotmas.routing import (
    Embedder,
    ExperienceRecord,
    ExperienceStore,
    LlmPool,
    Router,
    RoutingOutcome,
    SQLiteExperienceStore,
    Weights,
)

_log = get_logger("fedotmas.plugins.routing")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


def _is_workflow_node(name: str) -> bool:
    return name.startswith(_WORKFLOW_PREFIXES)


def _last_user_text(contents: list[types.Content]) -> str:
    """Return the most recent user-role text, joined across parts.
    Falls back to the last message of any role if no user message has
    text. Used as the embedding query — the system instruction is
    excluded because it's a constant for a given ``agent_role``, which
    is already captured separately."""
    for c in reversed(contents):
        if c.role == "user" and c.parts:
            texts = [p.text for p in c.parts if p.text]
            if texts:
                return "\n".join(texts)
    if contents and contents[-1].parts:
        texts = [p.text for p in contents[-1].parts if p.text]
        if texts:
            return "\n".join(texts)
    return ""


def _tools_from_request(llm_request: LlmRequest) -> tuple[str, ...]:
    if not llm_request.tools_dict:
        return ()
    return tuple(sorted(llm_request.tools_dict.keys()))


def _estimate_cost(
    usage: types.GenerateContentResponseUsageMetadata | None,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    if usage is None:
        return 0.0
    prompt_tokens = usage.prompt_token_count or 0
    completion_tokens = usage.candidates_token_count or 0
    return (
        prompt_tokens * input_price_per_1m
        + completion_tokens * output_price_per_1m
    ) / 1_000_000


@dataclass(slots=True)
class _Pending:
    t0: float
    outcome: RoutingOutcome
    query: str
    tools: tuple[str, ...]
    original_model: str | None


class LLMRoutingPlugin(BasePlugin):
    """Routes each LLM call via the experience-driven :class:`Router`.

    The plugin is opt-in. When not registered, ``MAWAgentConfig.model``
    drives static per-agent selection unchanged.

    Usage::

        from fedotmas.plugins import LLMRoutingPlugin
        from fedotmas.routing import LlmPool, LlmPoolEntry

        pool = LlmPool(entries=(
            LlmPoolEntry("openai/gpt-4o-mini", 0.15, 0.6),
            LlmPoolEntry("openai/gpt-4o", 2.5, 10.0),
        ))
        plugin = LLMRoutingPlugin(pool=pool)
        maw = MAW(plugins=[plugin])

    After each task is scored::

        plugin.commit_task_score(run.invocation_id, score)
    """

    DEFAULT_DB_PATH = "outputs/routing/experience.sqlite"

    def __init__(
        self,
        *,
        pool: LlmPool,
        store: ExperienceStore | None = None,
        embedder: Embedder | None = None,
        weights: Weights | None = None,
        cold_start_threshold: int = 50,
        sim_threshold: float = 0.85,
        db_path: str | Path = DEFAULT_DB_PATH,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(name="fedotmas_routing")
        self._pool = pool
        self._store: ExperienceStore = store or SQLiteExperienceStore(db_path)
        self._embedder = embedder or Embedder()
        self._router = Router(
            pool=pool,
            store=self._store,
            embedder=self._embedder,
            weights=weights or Weights(),
            cold_start_threshold=cold_start_threshold,
            sim_threshold=sim_threshold,
            rng=rng,
        )
        # (trace_id, agent_name) → state stashed in before_model_callback,
        # consumed in after_model_callback / on_model_error_callback.
        # Keyed by both because parallel agents in the same trace would
        # otherwise race on a shared agent_name slot.
        self._pending: dict[tuple[str, str], _Pending] = {}
        self._step_counter: dict[str, int] = {}

    @property
    def store(self) -> ExperienceStore:
        return self._store

    @property
    def router(self) -> Router:
        return self._router

    def commit_task_score(self, trace_id: str, score: float) -> int:
        """Backfill ``success_task`` into every record for ``trace_id``
        that doesn't have one yet. Returns the number of rows updated.
        Called by the benchmark runner after the scorer produces a
        final score for the task."""
        n = self._store.backfill_task_success(trace_id, score)
        _log.debug(
            "Backfilled task score | trace_id={} score={} rows={}",
            trace_id,
            score,
            n,
        )
        return n

    # ── Run lifecycle ──────────────────────────────────────────────

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        self._step_counter[invocation_context.invocation_id] = 0
        return None

    # ── Model lifecycle ────────────────────────────────────────────

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        agent_name = callback_context._invocation_context.agent.name
        if _is_workflow_node(agent_name):
            return None

        trace_id = callback_context._invocation_context.invocation_id
        query = _last_user_text(llm_request.contents)
        tools = _tools_from_request(llm_request)

        try:
            outcome = await self._router.select(
                agent_role=agent_name,
                tools=tools,
                query=query,
            )
        except Exception as e:
            _log.warning(
                "Router selection failed, falling back to original model | "
                "agent={} model={} error={}",
                agent_name,
                llm_request.model,
                e,
            )
            return None

        original_model = llm_request.model
        llm_request.model = outcome.decision.chosen_model
        self._pending[(trace_id, agent_name)] = _Pending(
            t0=time.monotonic(),
            outcome=outcome,
            query=query,
            tools=tools,
            original_model=original_model,
        )
        _log.debug(
            "Routed | agent={} chosen={} cold_start={} retrieved={}",
            agent_name,
            outcome.decision.chosen_model,
            outcome.decision.was_cold_start,
            outcome.decision.retrieved_records,
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        agent_name = callback_context._invocation_context.agent.name
        trace_id = callback_context._invocation_context.invocation_id
        pending = self._pending.pop((trace_id, agent_name), None)
        if pending is None:
            return None
        self._record(
            trace_id=trace_id,
            agent_name=agent_name,
            pending=pending,
            usage=llm_response.usage_metadata,
            success_step=1.0,
        )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        agent_name = callback_context._invocation_context.agent.name
        trace_id = callback_context._invocation_context.invocation_id
        pending = self._pending.pop((trace_id, agent_name), None)
        if pending is None:
            return None
        self._record(
            trace_id=trace_id,
            agent_name=agent_name,
            pending=pending,
            usage=None,
            success_step=0.0,
        )
        return None

    # ── Persistence helper ─────────────────────────────────────────

    def _record(
        self,
        *,
        trace_id: str,
        agent_name: str,
        pending: _Pending,
        usage: types.GenerateContentResponseUsageMetadata | None,
        success_step: float,
    ) -> None:
        chosen = pending.outcome.decision.chosen_model
        try:
            entry = self._pool.by_model(chosen)
        except KeyError:
            _log.warning(
                "Chosen model not in pool, skipping record | model={}", chosen
            )
            return

        cost = _estimate_cost(
            usage, entry.input_price_per_1m, entry.output_price_per_1m
        )
        duration = time.monotonic() - pending.t0
        step_idx = self._step_counter.get(trace_id, 0)
        self._step_counter[trace_id] = step_idx + 1

        try:
            self._store.append(
                ExperienceRecord(
                    trace_id=trace_id,
                    step_idx=step_idx,
                    agent_role=agent_name,
                    llm_used=chosen,
                    query=pending.query,
                    embedding=pending.outcome.query_embedding,
                    tools=pending.tools,
                    cost=cost,
                    duration=duration,
                    success_step=success_step,
                )
            )
        except Exception as e:
            _log.error(
                "Failed to append experience record | "
                "trace_id={} agent={} error={}",
                trace_id,
                agent_name,
                e,
            )
