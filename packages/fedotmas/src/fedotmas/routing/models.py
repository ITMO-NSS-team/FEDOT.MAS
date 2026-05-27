"""Data types for the routing module."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class LlmPoolEntry:
    """One model in the routing pool, with per-token prices used for cost
    estimation. Prices are user-supplied (USD per 1M tokens) — no automatic
    LiteLLM price lookup in v1.
    """

    model: str
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0


@dataclass(frozen=True, slots=True)
class LlmPool:
    """Curated set of candidate models the router can choose from."""

    entries: tuple[LlmPoolEntry, ...]

    def __post_init__(self) -> None:
        names = [e.model for e in self.entries]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate models in pool: {names}")
        if not self.entries:
            raise ValueError("LlmPool must be non-empty")
        zero_priced = [
            e.model
            for e in self.entries
            if e.input_price_per_1m == 0.0 and e.output_price_per_1m == 0.0
        ]
        if zero_priced:
            warnings.warn(
                f"LlmPool entries with zero prices: {zero_priced}. "
                "Cost will always be 0 for these models, so the cost term "
                "of the routing utility is inert. Set input_price_per_1m "
                "and output_price_per_1m if cost should influence routing.",
                stacklevel=2,
            )

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(e.model for e in self.entries)

    def by_model(self, model: str) -> LlmPoolEntry:
        for e in self.entries:
            if e.model == model:
                return e
        raise KeyError(model)


@dataclass(frozen=True, slots=True)
class Weights:
    """Scalarisation weights for the utility ``w_p·perf − w_c·cost − w_d·delay``."""

    perf: float = 1.0
    cost: float = 0.1
    delay: float = 0.05


@dataclass(slots=True)
class ExperienceRecord:
    """One step-level record persisted in the experience store.

    ``success_step`` is set when the record is appended (1.0 on success,
    0.0 on transport error). ``success_task`` is None until the parent
    task completes and is backfilled by trace_id.
    """

    trace_id: str
    step_idx: int
    agent_role: str
    llm_used: str
    query: str
    embedding: np.ndarray  # float32, 1-D
    tools: tuple[str, ...]
    cost: float
    duration: float
    success_step: float
    success_task: Optional[float] = None
    created_at: float = 0.0
    id: Optional[int] = None  # assigned by store on append


@dataclass(slots=True)
class LlmStats:
    """Per-LLM observations over a retrieved subset of records.

    Lists are flat float arrays so :func:`algorithm.thompson_sample` can
    compute posterior parameters directly.
    """

    model: str
    perf: list[float] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    delay: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.perf)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Outcome of one routing call, returned to the plugin.

    ``considered_models`` is the set the chosen model was picked from —
    the full pool during cold start, the Pareto-surviving subset otherwise.

    ``was_cold_start`` and ``no_relevant_history`` distinguish the two
    fall-back paths that both end in a uniform-random pick from the
    full pool:

    - ``was_cold_start=True``: global store has fewer than
      ``cold_start_threshold`` records — too little data anywhere.
    - ``no_relevant_history=True``: the store has data but none of it
      matches this ``(agent_role, tools, query)`` retrieval — the
      problem is targeted exploration, not bootstrap.

    The two are mutually exclusive.
    """

    chosen_model: str
    considered_models: tuple[str, ...]
    was_cold_start: bool
    no_relevant_history: bool
    retrieved_records: int
