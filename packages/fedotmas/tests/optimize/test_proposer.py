"""Tests for mutation and merge proposer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas.meta._adk_runner import LLMCallResult
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._proposer import (
    Proposer,
    ReflectionExample,
    _build_reflection_examples,
    _find_agent,
    _format_reflection_examples,
)
from fedotmas.optimize._state import Candidate


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
    examples = _build_reflection_examples(candidate, agent, ["t1", "t2"])
    assert len(examples) == 2
    assert examples[0].task == "t1"
    assert examples[0].score == 0.5
    assert examples[0].agent_output == "output_a_t1"


def test_build_reflection_examples_missing_task():
    config = _config(_agent("a"))
    candidate = _candidate_with_scores(config)
    agent = _find_agent(config, "a")
    examples = _build_reflection_examples(candidate, agent, ["t1", "nonexistent"])
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
async def test_propose_mutation():
    a1 = _agent("a", "Original instruction")
    config = _config(a1)
    candidate = Candidate(index=0, config=config, config_hash="h0")
    candidate.scores = {"t1": 0.5}
    candidate.feedbacks = {"t1": "improve"}
    candidate.states = {"t1": {"a": "output"}}

    proposer = Proposer()
    with patch("fedotmas.optimize._proposer.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"improved_instruction": "Better instruction"},
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )
        new_config = await proposer.propose_mutation(candidate, ["a"], ["t1"])

    assert new_config.agents[0].name == "a"
    assert "Better instruction" in new_config.agents[0].instruction
    assert new_config.agents[0].output_key == "a"  # preserved


@pytest.mark.asyncio
async def test_propose_mutation_preserves_invariants():
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

    proposer = Proposer()
    with patch("fedotmas.optimize._proposer.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"improved_instruction": "New instruction"},
            prompt_tokens=50,
            completion_tokens=25,
            elapsed=0.5,
        )
        new_config = await proposer.propose_mutation(candidate, ["researcher"], ["t1"])

    agent = new_config.agents[0]
    assert agent.name == "researcher"
    assert agent.output_key == "research_result"
    assert agent.tools == ["search"]


@pytest.mark.asyncio
async def test_propose_merge():
    a1 = _agent("a", "Approach A")
    a2 = _agent("a", "Approach B")
    config_a = _config(a1)
    config_b = _config(a2)

    ca = Candidate(index=0, config=config_a, config_hash="h0")
    cb = Candidate(index=1, config=config_b, config_hash="h1")

    proposer = Proposer()
    with patch("fedotmas.optimize._proposer.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"merged_instruction": "Combined approach"},
            prompt_tokens=80,
            completion_tokens=40,
            elapsed=0.8,
        )
        merged = await proposer.propose_merge(ca, cb, ["t1"])

    assert "Combined approach" in merged.agents[0].instruction


@pytest.mark.asyncio
async def test_propose_merge_skips_identical():
    """If instructions are identical, no merge call is made."""
    a = _agent("a", "Same instruction")
    config = _config(a)

    ca = Candidate(index=0, config=config, config_hash="h0")
    cb = Candidate(index=1, config=config, config_hash="h1")

    proposer = Proposer()
    with patch("fedotmas.optimize._proposer.run_meta_agent_call") as mock_call:
        merged = await proposer.propose_merge(ca, cb, ["t1"])

    mock_call.assert_not_called()
    assert merged.agents[0].instruction == config.agents[0].instruction


@pytest.mark.asyncio
async def test_propose_merge_union_agents():
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

    proposer = Proposer()
    with patch("fedotmas.optimize._proposer.run_meta_agent_call") as mock_call:
        mock_call.return_value = LLMCallResult(
            raw_output={"merged_instruction": "Merged A"},
            prompt_tokens=80,
            completion_tokens=40,
            elapsed=0.8,
        )
        merged = await proposer.propose_merge(ca, cb, ["t1"])

    agent_names = [ag.name for ag in merged.agents]
    assert "a" in agent_names
    assert "b" in agent_names


# --- 5c: token_usage property ---


def test_proposer_token_usage():
    """token_usage property returns (prompt, completion) tuple."""
    proposer = Proposer()
    assert proposer.token_usage == (0, 0)
    proposer.total_prompt_tokens = 100
    proposer.total_completion_tokens = 50
    assert proposer.token_usage == (100, 50)
