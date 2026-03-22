"""Tests for optimization state, Pareto frontier, and cache."""

from __future__ import annotations

import pytest

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._state import (
    Candidate,
    EvaluationCache,
    OptimizationState,
    TaskResult,
    config_hash,
)


def _agent(name: str, instruction: str = "Do stuff") -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=instruction, output_key=name)


def _config(*names: str, instructions: dict[str, str] | None = None) -> MAWConfig:
    instr = instructions or {}
    agents = [_agent(n, instr.get(n, f"Do {n}")) for n in names]
    pipeline = MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name=n) for n in names],
    )
    return MAWConfig(agents=agents, pipeline=pipeline)


# --- 1a: config_hash uses full model_dump_json ---


def test_config_hash_deterministic():
    c1 = _config("a", "b")
    c2 = _config("a", "b")
    assert config_hash(c1) == config_hash(c2)


def test_config_hash_differs_on_instruction():
    c1 = _config("a", instructions={"a": "Do X"})
    c2 = _config("a", instructions={"a": "Do Y"})
    assert config_hash(c1) != config_hash(c2)


def test_config_hash_differs_on_tools():
    """Hash should change when tools differ, not just instructions."""
    a1 = MAWAgentConfig(name="a", instruction="Do stuff", output_key="a", tools=["search"])
    a2 = MAWAgentConfig(name="a", instruction="Do stuff", output_key="a", tools=[])
    cfg1 = MAWConfig(agents=[a1], pipeline=MAWStepConfig(agent_name="a"))
    cfg2 = MAWConfig(agents=[a2], pipeline=MAWStepConfig(agent_name="a"))
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_differs_on_pipeline_structure():
    """Hash should change when pipeline structure differs."""
    a = _agent("a")
    b = _agent("b")
    cfg1 = MAWConfig(
        agents=[a, b],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(agent_name="a"), MAWStepConfig(agent_name="b")],
        ),
    )
    cfg2 = MAWConfig(
        agents=[a, b],
        pipeline=MAWStepConfig(
            type="parallel",
            children=[MAWStepConfig(agent_name="a"), MAWStepConfig(agent_name="b")],
        ),
    )
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_length():
    """Hash should be 32 hex chars."""
    h = config_hash(_config("a"))
    assert len(h) == 32


# --- 1b: Pareto front — intersection-based ---


def test_pareto_front_single():
    state = OptimizationState()
    c = state.add_candidate(_config("a"))
    c.scores = {"t1": 0.5}
    state.update_pareto_front()
    assert c.on_pareto_front is True


def test_pareto_front_domination():
    state = OptimizationState()
    c1 = state.add_candidate(_config("a", instructions={"a": "v1"}))
    c1.scores = {"t1": 0.5, "t2": 0.5}
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), parent_index=0, origin="mutation"
    )
    c2.scores = {"t1": 0.8, "t2": 0.7}
    state.update_pareto_front()
    assert c2.on_pareto_front is True
    assert c1.on_pareto_front is False


def test_pareto_front_non_domination():
    state = OptimizationState()
    c1 = state.add_candidate(_config("a", instructions={"a": "v1"}))
    c1.scores = {"t1": 0.9, "t2": 0.3}
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), parent_index=0, origin="mutation"
    )
    c2.scores = {"t1": 0.3, "t2": 0.9}
    state.update_pareto_front()
    assert c1.on_pareto_front is True
    assert c2.on_pareto_front is True


def test_pareto_front_disjoint_tasks_no_domination():
    """Candidates with no shared tasks should not dominate each other."""
    state = OptimizationState()
    c1 = state.add_candidate(_config("a", instructions={"a": "v1"}))
    c1.scores = {"t1": 0.9}
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), parent_index=0, origin="mutation"
    )
    c2.scores = {"t2": 0.1}
    state.update_pareto_front()
    # Neither should dominate — disjoint tasks
    assert c1.on_pareto_front is True
    assert c2.on_pareto_front is True


def test_pareto_front_partial_overlap():
    """With partial task overlap, domination uses only common tasks."""
    state = OptimizationState()
    c1 = state.add_candidate(_config("a", instructions={"a": "v1"}))
    c1.scores = {"t1": 0.9, "t2": 0.8}
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), parent_index=0, origin="mutation"
    )
    # c2 is better on the shared task t1, but doesn't have t2
    c2.scores = {"t1": 0.95, "t3": 0.1}
    state.update_pareto_front()
    # c2 dominates c1 on {t1} (only common task)
    assert c2.on_pareto_front is True
    assert c1.on_pareto_front is False


# --- 1c: mean_score / min_score return None for unevaluated ---


def test_candidate_mean_score_none_when_empty():
    c = Candidate(index=0, config=_config("a"), config_hash="h")
    assert c.mean_score is None


def test_candidate_min_score_none_when_empty():
    c = Candidate(index=0, config=_config("a"), config_hash="h")
    assert c.min_score is None


def test_candidate_mean_score():
    c = Candidate(index=0, config=_config("a"), config_hash="h")
    c.scores = {"t1": 0.8, "t2": 0.6}
    assert c.mean_score == pytest.approx(0.7)


def test_candidate_min_score():
    c = Candidate(index=0, config=_config("a"), config_hash="h")
    c.scores = {"t1": 0.8, "t2": 0.3, "t3": 0.5}
    assert c.min_score == pytest.approx(0.3)


# --- 1d: EvaluationCache with max_size ---


def test_evaluation_cache():
    cache = EvaluationCache()
    assert cache.get("h", "t") is None
    result = TaskResult(task="t", state={}, score=0.5, feedback="ok")
    cache.put("h", "t", result)
    assert cache.get("h", "t") is result
    assert len(cache) == 1


def test_evaluation_cache_no_limit():
    """Without max_size, cache grows without bound."""
    cache = EvaluationCache()
    for i in range(100):
        cache.put(f"h{i}", "t", TaskResult(task="t", state={}, score=0.5, feedback="ok"))
    assert len(cache) == 100


def test_evaluation_cache_max_size():
    """Cache evicts oldest entries when max_size is exceeded."""
    cache = EvaluationCache(max_size=3)
    for i in range(5):
        cache.put(f"h{i}", "t", TaskResult(task="t", state={}, score=float(i), feedback="ok"))
    assert len(cache) == 3
    # Oldest (h0, h1) should be evicted
    assert cache.get("h0", "t") is None
    assert cache.get("h1", "t") is None
    assert cache.get("h2", "t") is not None
    assert cache.get("h3", "t") is not None
    assert cache.get("h4", "t") is not None


def test_evaluation_cache_max_size_update_existing():
    """Re-putting an existing key should move it to end, not evict it."""
    cache = EvaluationCache(max_size=2)
    r1 = TaskResult(task="t", state={}, score=0.1, feedback="ok")
    r2 = TaskResult(task="t", state={}, score=0.2, feedback="ok")
    r3 = TaskResult(task="t", state={}, score=0.3, feedback="ok")
    cache.put("h0", "t", r1)
    cache.put("h1", "t", r2)
    # Re-put h0 — should move to end
    cache.put("h0", "t", r1)
    # Now add h2 — h1 should be evicted (oldest)
    cache.put("h2", "t", r3)
    assert len(cache) == 2
    assert cache.get("h1", "t") is None
    assert cache.get("h0", "t") is r1
    assert cache.get("h2", "t") is r3


# --- Existing tests ---


def test_state_add_candidate():
    state = OptimizationState()
    c = state.add_candidate(_config("a"))
    assert c.index == 0
    assert c.origin == "seed"
    c2 = state.add_candidate(_config("a"), parent_index=0, origin="mutation")
    assert c2.index == 1
    assert c2.parent_index == 0


def test_state_record_and_cache():
    state = OptimizationState()
    c = state.add_candidate(_config("a"))
    result = TaskResult(task="t1", state={"a": "val"}, score=0.9, feedback="great")
    state.record_task_result(c, result)
    assert c.scores["t1"] == 0.9
    assert c.feedbacks["t1"] == "great"
    assert c.states["t1"] == {"a": "val"}
    assert state.cache.get(c.config_hash, "t1") is result


def test_best_candidate():
    state = OptimizationState()
    c1 = state.add_candidate(_config("a", instructions={"a": "v1"}))
    c1.scores = {"t1": 0.5}
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), origin="mutation"
    )
    c2.scores = {"t1": 0.9}
    assert state.best_candidate() is c2


def test_best_candidate_empty():
    state = OptimizationState()
    assert state.best_candidate() is None
