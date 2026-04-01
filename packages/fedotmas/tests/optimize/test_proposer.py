"""Tests for InstructionMutator (mutation, merge, genealogy merge)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas.meta._adk_runner import LLMCallResult
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._mutators._instruction import (
    InstructionMutator,
    ReflectionExample,
    _build_reflection_examples,
    _find_agent,
    _format_reflection_examples,
    _mean_score_on_common,
)
from fedotmas.optimize._state import Candidate, Task


def _agent(name: str, instruction: str = "Do stuff") -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=instruction, output_key=name)


def _config(*agents: MAWAgentConfig) -> MAWConfig:
    pipeline = MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name=a.name) for a in agents],
    )
    return MAWConfig(agents=list(agents), pipeline=pipeline)


def _candidate_with_scores(config: MAWConfig) -> Candidate:
    c = Candidate(index=0, config=config, config_hash="h0")
    c.scores = {"t1": 0.5, "t2": 0.8}
    c.feedbacks = {"t1": "needs work", "t2": "good"}
    c.states = {
        "t1": {"a": "output_a_t1", "b": "output_b_t1"},
        "t2": {"a": "output_a_t2", "b": "output_b_t2"},
    }
    return c


def test_find_agent():
    config = _config(_agent("a"), _agent("b"))
    assert _find_agent(config, "a") is not None
    assert _find_agent(config, "a").name == "a"
    assert _find_agent(config, "z") is None


def test_build_reflection_examples():
    config = _config(_agent("a"), _agent("b"))
    candidate = _candidate_with_scores(config)
    agent = _find_agent(config, "a")
    examples = _build_reflection_examples(candidate, agent, [Task("t1"), Task("t2")])
    assert len(examples) == 2
    assert examples[0].task == "t1"
    assert examples[0].score == 0.5
    assert examples[0].agent_output == "output_a_t1"


def test_build_reflection_examples_missing_task():
    config = _config(_agent("a"))
    candidate = _candidate_with_scores(config)
    agent = _find_agent(config, "a")
    examples = _build_reflection_examples(candidate, agent, [Task("t1"), Task("nonexistent")])
    assert len(examples) == 1


def test_format_reflection_examples():
    examples = [
        ReflectionExample(
            task="test task",
            agent_instruction="Do stuff",
            agent_output="result",
            pipeline_output={"a": "result"},
            score=0.7,
            feedback="Pretty good",
        )
    ]
    text = _format_reflection_examples(examples)
    assert "test task" in text
    assert "0.70" in text
    assert "Pretty good" in text


def test_format_reflection_examples_truncation():
    """Custom max_output_chars controls agent output truncation."""
    examples = [
        ReflectionExample(
            task="t",
            agent_instruction="i",
            agent_output="x" * 500,
            pipeline_output={},
            score=0.5,
            feedback="f",
        )
    ]
    text = _format_reflection_examples(examples, max_output_chars=100)
    assert "truncated" in text


@pytest.mark.asyncio
async def test_mutate():
    a1 = _agent("a", "Original instruction")
    config = _config(a1)
    candidate = Candidate(index=0, config=config, config_hash="h0")
    candidate.scores = {"t1": 0.5}
    candidate.feedbacks = {"t1": "improve"}
    candidate.states = {"t1": {"a": "output"}}

    mutator = InstructionMutator()
    with patch("fedotmas.optimize._mutators._instruction.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"improved_instruction": "Better instruction"},
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )
        new_config = await mutator.mutate(candidate, ["a"], [Task("t1")])

    assert new_config.agents[0].name == "a"
    assert "Better instruction" in new_config.agents[0].instruction
    assert new_config.agents[0].output_key == "a"  # preserved


@pytest.mark.asyncio
async def test_mutate_preserves_invariants():
    """Name and output_key must not change during mutation."""
    a1 = MAWAgentConfig(
        name="researcher",
        instruction="Research the topic",
        output_key="research_result",
        tools=["search"],
    )
    config = _config(a1)
    candidate = Candidate(index=0, config=config, config_hash="h0")
    candidate.scores = {"t1": 0.5}
    candidate.feedbacks = {"t1": "improve"}
    candidate.states = {"t1": {"research_result": "output"}}

    mutator = InstructionMutator()
    with patch("fedotmas.optimize._mutators._instruction.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"improved_instruction": "New instruction"},
            prompt_tokens=50,
            completion_tokens=25,
            elapsed=0.5,
        )
        new_config = await mutator.mutate(candidate, ["researcher"], [Task("t1")])

    agent = new_config.agents[0]
    assert agent.name == "researcher"
    assert agent.output_key == "research_result"
    assert agent.tools == ["search"]


@pytest.mark.asyncio
async def test_merge():
    a1 = _agent("a", "Approach A")
    a2 = _agent("a", "Approach B")
    config_a = _config(a1)
    config_b = _config(a2)

    ca = Candidate(index=0, config=config_a, config_hash="h0")
    cb = Candidate(index=1, config=config_b, config_hash="h1")

    mutator = InstructionMutator()
    with patch("fedotmas.optimize._mutators._instruction.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"merged_instruction": "Combined approach"},
            prompt_tokens=80,
            completion_tokens=40,
            elapsed=0.8,
        )
        merged = await mutator.merge(ca, cb, [Task("t1")])

    assert "Combined approach" in merged.agents[0].instruction


@pytest.mark.asyncio
async def test_merge_skips_identical():
    """If instructions are identical, no merge call is made."""
    a = _agent("a", "Same instruction")
    config = _config(a)

    ca = Candidate(index=0, config=config, config_hash="h0")
    cb = Candidate(index=1, config=config, config_hash="h1")

    mutator = InstructionMutator()
    with patch("fedotmas.optimize._mutators._instruction.run_meta_agent_call") as mock_call:
        merged = await mutator.merge(ca, cb, [Task("t1")])

    mock_call.assert_not_called()
    assert merged.agents[0].instruction == config.agents[0].instruction


@pytest.mark.asyncio
async def test_merge_union_agents():
    """Agents present only in config_b should be preserved in the result."""
    a = _agent("a", "Approach A")
    b = _agent("b", "Only in B")
    config_a = _config(a)
    config_b = MAWConfig(
        agents=[_agent("a", "Approach B"), b],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(agent_name="a"), MAWStepConfig(agent_name="b")],
        ),
    )

    ca = Candidate(index=0, config=config_a, config_hash="h0")
    cb = Candidate(index=1, config=config_b, config_hash="h1")

    mutator = InstructionMutator()
    with patch("fedotmas.optimize._mutators._instruction.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"merged_instruction": "Merged A"},
            prompt_tokens=80,
            completion_tokens=40,
            elapsed=0.8,
        )
        merged = await mutator.merge(ca, cb, [Task("t1")])

    agent_names = [ag.name for ag in merged.agents]
    assert "a" in agent_names
    assert "b" in agent_names


# --- _mean_score_on_common ---


def test_mean_score_on_common_with_overlap():
    c1 = Candidate(index=0, config=_config(_agent("a")), config_hash="h0")
    c1.scores = {"t1": 0.8, "t2": 0.6, "t3": 0.4}
    c2 = Candidate(index=1, config=_config(_agent("a")), config_hash="h1")
    c2.scores = {"t2": 0.5, "t3": 0.9}
    # Common tasks: t2, t3 → c1 scores on those: (0.6 + 0.4) / 2 = 0.5
    assert _mean_score_on_common(c1, c2) == pytest.approx(0.5)


def test_mean_score_on_common_no_overlap():
    c1 = Candidate(index=0, config=_config(_agent("a")), config_hash="h0")
    c1.scores = {"t1": 0.8}
    c2 = Candidate(index=1, config=_config(_agent("a")), config_hash="h1")
    c2.scores = {"t2": 0.5}
    # No common tasks → falls back to c1.mean_score
    assert _mean_score_on_common(c1, c2) == pytest.approx(0.8)


# --- Genealogy merge fallback uses common-task scoring ---


@pytest.mark.asyncio
async def test_genealogy_merge_fallback_uses_common_tasks():
    """When LLM merge fails, fallback should compare on common tasks, not mean_score."""
    anc_agent = _agent("a", "Original")
    config_anc = _config(anc_agent)
    config_a = _config(_agent("a", "Changed by A"))
    config_b = _config(_agent("a", "Changed by B"))

    ancestor = Candidate(index=0, config=config_anc, config_hash="h0")
    child_a = Candidate(index=1, config=config_a, config_hash="h1", parent_index=0)
    child_a.scores = {"t1": 0.9, "t2": 0.3}
    child_b = Candidate(index=2, config=config_b, config_hash="h2", parent_index=0)
    child_b.scores = {"t2": 0.8, "t3": 0.95}

    mutator = InstructionMutator()
    with patch("fedotmas.optimize._mutators._instruction.run_meta_agent_call") as mock_call:
        mock_call.side_effect = RuntimeError("LLM unavailable")
        merged = await mutator.genealogy_merge(
            ancestor, child_a, child_b, [Task("t1")]
        )

    # Should pick child_b's instruction (higher on common task t2)
    assert merged.agents[0].instruction == "Changed by B"


# --- token_usage property ---


def test_mutator_token_usage():
    """token_usage property returns (prompt, completion) tuple."""
    mutator = InstructionMutator()
    assert mutator.token_usage == (0, 0)
    mutator._total_prompt_tokens = 100
    mutator._total_completion_tokens = 50
    assert mutator.token_usage == (100, 50)
