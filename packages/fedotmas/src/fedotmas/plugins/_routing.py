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

# Bound for the per-trace state buffers below. ADK does not call
# after_run_callback when a run raises, so _pending and _step_counter
# can leak across failed runs; we cap by size and FIFO-evict so the
# leak is bounded rather than fatal. 1000 entries ≈ a long benchmark.
_BUFFER_CAP = 1000


def _is_workflow_node(name: str) -> bool:
    return name.startswith(_WORKFLOW_PREFIXES)


def _invocation_id(callback_context: CallbackContext) -> str:
    """Accessor for the ADK private attribute we depend on. Isolated
    here so a future ADK rename breaks one line, not the whole plugin.
    No public alternative exists at the moment ADK 1.x."""
    return callback_context._invocation_context.invocation_id


def _agent_name(callback_context: CallbackContext) -> str:
    return callback_context._invocation_context.agent.name


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
    """Best-effort cost estimate from token usage and pool pricing.

    Reasoning tokens (``thoughts_token_count`` for o1/o4-mini/Claude
    thinking) are billed at the output rate and would otherwise be
    invisible to the router — for reasoning models that's the
    dominant cost component. Cached prompt tokens are still counted
    at the input rate; v1 has no separate cached-input pricing field
    on :class:`LlmPoolEntry`.
    """
    if usage is None:
        return 0.0
    prompt_tokens = usage.prompt_token_count or 0
    completion_tokens = usage.candidates_token_count or 0
    reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    return (
        prompt_tokens * input_price_per_1m
        + (completion_tokens + reasoning_tokens) * output_price_per_1m
    ) / 1_000_000


@dataclass(slots=True)
class _Pending:
    t0: float
    outcome: RoutingOutcome
    query: str
    tools: tuple[str, ...]


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
        self._router = Router(
            pool=pool,
            store=self._store,
            embedder=embedder or Embedder(),
            weights=weights or Weights(),
            cold_start_threshold=cold_start_threshold,
            sim_threshold=sim_threshold,
            rng=rng,
        )
        # (trace_id, agent_name) → state stashed in before_model_callback,
        # consumed in after_model_callback / on_model_error_callback.
        # Keyed by both because parallel agents in the same trace would
        # otherwise race on a shared agent_name slot. Cleared per-trace
        # in after_run_callback so an unpaired before-hook (cancel,
        # crash in another plugin) doesn't leak across runs.
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

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Clear per-trace state on **successful** run end. Warns when
        a ``before_model_callback`` from this trace never received a
        matching ``after`` / ``error`` — the LLM call's record was
        never appended, so it's data loss worth surfacing.

        Note: ADK does not call ``after_run_callback`` when the run
        itself raises. Leak protection on the error path is provided
        by the FIFO cap in :meth:`_enforce_buffer_cap`, which runs on
        every ``before_model_callback``.
        """
        trace_id = invocation_context.invocation_id
        self._step_counter.pop(trace_id, None)
        orphans = [
            (t, a) for (t, a) in self._pending if t == trace_id
        ]
        for key in orphans:
            self._pending.pop(key, None)
        if orphans:
            _log.warning(
                "Discarded {} orphan pending routing record(s) at run end | "
                "trace_id={} agents={}",
                len(orphans),
                trace_id,
                [a for _, a in orphans],
            )

    # ── Buffer cap (defensive against unpaired before-hooks) ──────

    def _enforce_buffer_cap(self) -> None:
        """Bound ``_pending`` and ``_step_counter`` by FIFO eviction.
        Routine cleanup is :meth:`after_run_callback`; this cap is the
        backstop for the case where ADK skips that callback (run-time
        exception) and the buffers would otherwise grow without
        bound. Eviction is logged so a real leak is visible."""
        excess_pending = len(self._pending) - _BUFFER_CAP
        if excess_pending > 0:
            victims = sorted(
                self._pending.items(), key=lambda kv: kv[1].t0
            )[:excess_pending]
            for key, _ in victims:
                del self._pending[key]
            _log.warning(
                "Evicted {} pending routing entries (cap={}) | "
                "suggests unpaired before_model_callbacks",
                len(victims),
                _BUFFER_CAP,
            )
        excess_counter = len(self._step_counter) - _BUFFER_CAP
        if excess_counter > 0:
            # dict iteration order is insertion order in CPython 3.7+,
            # so list(...)[:n] yields the oldest n keys.
            victims_k = list(self._step_counter.keys())[:excess_counter]
            for k in victims_k:
                del self._step_counter[k]

    # ── Model lifecycle ────────────────────────────────────────────

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        agent_name = _agent_name(callback_context)
        if _is_workflow_node(agent_name):
            return None

        trace_id = _invocation_id(callback_context)
        # TODO(phase-5): contents[-1] is often the same user task across
        # all agents in a MAW pipeline — they're differentiated by
        # config.system_instruction, not by user content. Routing
        # retrieval by similarity is uninformative until we mix the
        # system instruction into the query string.
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

        key = (trace_id, agent_name)
        if key in self._pending:
            # ADK normally pairs before/after per agent call, but a retry
            # loop inside one agent.run() could re-enter before the
            # earlier response surfaced. We'd silently lose that record;
            # warn so it shows up in logs.
            _log.warning(
                "Overwriting unconsumed pending routing record | "
                "trace_id={} agent={} prev_model={} new_model={}",
                trace_id,
                agent_name,
                self._pending[key].outcome.decision.chosen_model,
                outcome.decision.chosen_model,
            )

        llm_request.model = outcome.decision.chosen_model
        self._pending[key] = _Pending(
            t0=time.monotonic(),
            outcome=outcome,
            query=query,
            tools=tools,
        )
        # Enforce cap *after* insertion so we don't grow past the
        # cap by a single entry between calls.
        self._enforce_buffer_cap()
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
        agent_name = _agent_name(callback_context)
        trace_id = _invocation_id(callback_context)
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
        agent_name = _agent_name(callback_context)
        trace_id = _invocation_id(callback_context)
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
