"""Pure-function core of the router: aggregation, Pareto filtering,
Thompson sampling, utility scoring. No I/O, no ADK, no global state — RNG
is injected so behaviour is reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .models import ExperienceRecord, LlmPool, LlmStats, Weights

# Forces exploration when a model has < MIN_N_FOR_INFERENCE samples in
# the retrieved subset — Thompson sampling needs at least two
# observations to estimate variance.
MIN_N_FOR_INFERENCE = 2

# Pareto filter ignores models with fewer than this many observations:
# their point estimate is too noisy to dominate or be dominated.
MIN_N_FOR_PARETO = 3

EXPLORE_UTILITY = math.inf


def aggregate(
    records: Iterable[ExperienceRecord], pool: LlmPool
) -> dict[str, LlmStats]:
    """Group records by LLM and assemble per-metric observation lists.

    Models from ``pool`` that have no matching records still appear in
    the result with empty lists — they will be picked up by the
    "force-exploration on n=0" branch of :func:`thompson_sample`.

    Records whose ``success_task`` is still ``None`` (parent task not
    yet scored) are skipped: their perf signal isn't known.
    """
    by_model: dict[str, LlmStats] = {m: LlmStats(model=m) for m in pool.models}
    for r in records:
        if r.llm_used not in by_model:
            continue
        if r.success_task is None:
            continue
        s = by_model[r.llm_used]
        s.perf.append(r.success_step * r.success_task)
        s.cost.append(r.cost)
        s.delay.append(r.duration)
    return by_model


def pareto_filter(stats_by_model: dict[str, LlmStats]) -> list[str]:
    """Return models that are non-dominated on (perf↑, cost↓, delay↓).

    Models with fewer than :data:`MIN_N_FOR_PARETO` observations always
    pass through — their position is too uncertain to be confidently
    dominated.
    """
    well_sampled: list[str] = []
    under_sampled: list[str] = []
    means: dict[str, tuple[float, float, float]] = {}

    for model, s in stats_by_model.items():
        if s.n < MIN_N_FOR_PARETO:
            under_sampled.append(model)
            continue
        well_sampled.append(model)
        means[model] = (
            float(np.mean(s.perf)),
            float(np.mean(s.cost)),
            float(np.mean(s.delay)),
        )

    non_dominated: list[str] = []
    for a in well_sampled:
        pa, ca, da = means[a]
        dominated = False
        for b in well_sampled:
            if a == b:
                continue
            pb, cb, db = means[b]
            if pb >= pa and cb <= ca and db <= da and (
                pb > pa or cb < ca or db < da
            ):
                dominated = True
                break
        if not dominated:
            non_dominated.append(a)

    return under_sampled + non_dominated


def _sample_metric(values: list[float], rng: np.random.Generator) -> float:
    """Draw a posterior-mean sample under a non-informative
    (Jeffreys) Normal-Inverse-Gamma prior. Returns ``+inf`` when the
    sample is too small to estimate variance — see
    :data:`MIN_N_FOR_INFERENCE`.
    """
    n = len(values)
    if n < MIN_N_FOR_INFERENCE:
        return EXPLORE_UTILITY
    mean = float(np.mean(values))
    var = float(np.var(values, ddof=1))
    if var <= 0:
        return mean
    # NIG with non-informative prior (λ₀=0, α₀=0, β₀=0):
    # posterior σ² ~ InvGamma((n-1)/2, (n-1)·s²/2); μ ~ N(mean, σ²/n)
    alpha = (n - 1) / 2.0
    beta = (n - 1) * var / 2.0
    sigma2 = beta / rng.gamma(shape=alpha, scale=1.0)
    return float(rng.normal(loc=mean, scale=math.sqrt(sigma2 / n)))


def thompson_sample(
    stats: LlmStats, *, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Draw one ``(perf, cost, delay)`` triple for a candidate model.

    Each metric is sampled independently. When ``stats.n == 0`` all
    three metrics return the explore-sentinel; the caller scores any
    sentinel-bearing triple as :data:`EXPLORE_UTILITY` so under-sampled
    models are forced into rotation.
    """
    return (
        _sample_metric(stats.perf, rng),
        _sample_metric(stats.cost, rng),
        _sample_metric(stats.delay, rng),
    )


def score_utility(
    perf: float, cost: float, delay: float, *, weights: Weights
) -> float:
    """``w_p·perf − w_c·cost − w_d·delay``. Returns ``+inf`` if any
    component is the explore-sentinel, which lets argmax pick
    under-sampled candidates over confidently-scored ones."""
    if perf == EXPLORE_UTILITY or cost == EXPLORE_UTILITY or delay == EXPLORE_UTILITY:
        return EXPLORE_UTILITY
    return weights.perf * perf - weights.cost * cost - weights.delay * delay
