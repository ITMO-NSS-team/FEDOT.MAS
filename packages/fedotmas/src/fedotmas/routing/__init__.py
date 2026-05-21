"""Per-call LLM routing for FEDOT.MAS multi-agent pipelines.

Subpackage layout:
    models     — data types for records, pool, weights, decisions
    algorithm  — pure functions: aggregate, pareto_filter, thompson_sample
    store      — ExperienceStore protocol + SQLite implementation
    embeddings — query-embedding helper (litellm.aembedding + cache)
    selector   — Router orchestrating one routing decision

Nothing here imports ADK; the plugin that wires it into the pipeline
lives in ``fedotmas.plugins``.
"""

from .embeddings import Embedder
from .models import (
    ExperienceRecord,
    LlmPool,
    LlmPoolEntry,
    LlmStats,
    RoutingDecision,
    Weights,
)
from .selector import Router, RoutingOutcome
from .store import ExperienceStore, SQLiteExperienceStore

__all__ = [
    "Embedder",
    "ExperienceRecord",
    "ExperienceStore",
    "LlmPool",
    "LlmPoolEntry",
    "LlmStats",
    "Router",
    "RoutingDecision",
    "RoutingOutcome",
    "SQLiteExperienceStore",
    "Weights",
]
