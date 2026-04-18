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
    is_ancestor_of,
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
    state.record_task_result(c, result, split="val")
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


def test_is_ancestor_of():
    state = OptimizationState()
    c0 = state.add_candidate(_config("a"), origin="seed")
    c1 = state.add_candidate(
        _config("a", instructions={"a": "v1"}), parent_index=0, origin="mutation"
    )
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), parent_index=1, origin="mutation"
    )
    # c0 → c1 → c2
    assert is_ancestor_of(c0, c2, state.candidates) is True
    assert is_ancestor_of(c0, c1, state.candidates) is True
    assert is_ancestor_of(c1, c2, state.candidates) is True
    # Not ancestors
    assert is_ancestor_of(c2, c0, state.candidates) is False
    assert is_ancestor_of(c1, c0, state.candidates) is False
    # Self is not its own ancestor
    assert is_ancestor_of(c0, c0, state.candidates) is False


def test_is_ancestor_of_unrelated():
    state = OptimizationState()
    c0 = state.add_candidate(_config("a"), origin="seed")
    c1 = state.add_candidate(
        _config("a", instructions={"a": "v1"}), parent_index=0, origin="mutation"
    )
    c2 = state.add_candidate(
        _config("a", instructions={"a": "v2"}), parent_index=0, origin="mutation"
    )
    # c1 and c2 are siblings, not ancestors of each other
    assert is_ancestor_of(c1, c2, state.candidates) is False
    assert is_ancestor_of(c2, c1, state.candidates) is False


# --- Train/val score isolation ---


def test_record_task_result_routes_to_split():
    """split='train' writes to train_*; split='val' writes to scores/feedbacks/states."""
    state = OptimizationState()
    c = state.add_candidate(_config("a"))

    state.record_task_result(
        c, TaskResult(task="v1", state={"a": "v_out"}, score=0.7, feedback="vf"),
        split="val",
    )
    state.record_task_result(
        c, TaskResult(task="t1", state={"a": "t_out"}, score=0.4, feedback="tf"),
        split="train",
    )

    assert c.scores == {"v1": 0.7}
    assert c.feedbacks == {"v1": "vf"}
    assert c.states == {"v1": {"a": "v_out"}}
    assert c.train_scores == {"t1": 0.4}
    assert c.train_feedbacks == {"t1": "tf"}
    assert c.train_states == {"t1": {"a": "t_out"}}


def test_train_scores_dont_affect_mean_score():
    """mean_score reads only val scores, ignoring train minibatch evals."""
    state = OptimizationState()
    c = state.add_candidate(_config("a"))
    for i, score in enumerate([0.5, 0.6, 0.7]):  # val
        state.record_task_result(
            c, TaskResult(task=f"v{i}", state={}, score=score, feedback=""),
            split="val",
        )
    # Add poor train scores — must not drag mean_score down
    for i in range(10):
        state.record_task_result(
            c, TaskResult(task=f"t{i}", state={}, score=0.0, feedback=""),
            split="train",
        )
    assert c.mean_score == pytest.approx(0.6)


def test_train_scores_dont_affect_pareto_dominance():
    """Pareto dominance ignores train scores (uses only c.scores)."""
    from fedotmas.optimize._state import _dominates

    state = OptimizationState()
    a = state.add_candidate(_config("a"))
    b = state.add_candidate(_config("a", instructions={"a": "v2"}))
    # Same val scores
    for cand in (a, b):
        state.record_task_result(
            cand, TaskResult(task="v1", state={}, score=0.5, feedback=""),
            split="val",
        )
    # 'a' has a private train task with perfect score — must not dominate.
    state.record_task_result(
        a, TaskResult(task="t1", state={}, score=1.0, feedback=""),
        split="train",
    )
    assert _dominates(a, b) is False
    assert _dominates(b, a) is False


def test_save_load_preserves_train_scores():
    import json
    import tempfile
    from pathlib import Path

    state = OptimizationState()
    c = state.add_candidate(_config("a"))
    state.record_task_result(
        c, TaskResult(task="v1", state={"a": "vo"}, score=0.7, feedback="vf"),
        split="val",
    )
    state.record_task_result(
        c, TaskResult(task="t1", state={"a": "to"}, score=0.4, feedback="tf"),
        split="train",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        state.save(path)
        loaded = OptimizationState.load(path)
        lc = loaded.candidates[0]
        assert lc.scores == {"v1": 0.7}
        assert lc.train_scores == {"t1": 0.4}
        assert lc.train_feedbacks == {"t1": "tf"}
        assert lc.train_states == {"t1": {"a": "to"}}
        # Cache restored for both splits
        assert loaded.cache.get(lc.config_hash, "v1") is not None
        assert loaded.cache.get(lc.config_hash, "t1") is not None
    finally:
        path.unlink(missing_ok=True)


def test_load_old_checkpoint_without_train_fields():
    """Backward compat: old JSONs without train_* should load with empty defaults."""
    import json
    import tempfile
    from pathlib import Path

    cfg = _config("a")
    legacy = {
        "next_index": 1,
        "total_evaluations": 1,
        "iteration": 0,
        "candidates": [
            {
                "index": 0,
                "config": json.loads(cfg.model_dump_json()),
                "config_hash": config_hash(cfg),
                "scores": {"v1": 0.5},
                "feedbacks": {"v1": "ok"},
                "states": {"v1": {"a": "out"}},
                "parent_index": None,
                "origin": "seed",
                "on_pareto_front": True,
                "merge_parent_indices": None,
            }
        ],
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(legacy, f)
        path = Path(f.name)
    try:
        loaded = OptimizationState.load(path)
        c = loaded.candidates[0]
        assert c.scores == {"v1": 0.5}
        assert c.train_scores == {}
        assert c.train_feedbacks == {}
        assert c.train_states == {}
    finally:
        path.unlink(missing_ok=True)
