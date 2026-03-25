"""Tests for scoring module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas.optimize._scoring import LLMJudge, ScoringResult, _format_state


def test_scoring_result_fields():
    r = ScoringResult(score=0.8, feedback="Good", reasoning="Solid work")
    assert r.score == 0.8
    assert r.feedback == "Good"
    assert r.reasoning == "Solid work"


def test_format_state_empty():
    assert _format_state({}) == "(empty state)"


def test_format_state_no_truncation_by_default():
    state = {"key": "x" * 5000}
    result = _format_state(state)
    assert "truncated" not in result
    assert "x" * 5000 in result


def test_format_state_truncation_with_limit():
    state = {"key": "x" * 3000}
    result = _format_state(state, max_chars=2000)
    assert "truncated" in result
    assert len(result) < 3000


def test_format_state_custom_max_chars():
    """Custom max_chars parameter controls truncation."""
    state = {"key": "x" * 500}
    result_default = _format_state(state, max_chars=2000)
    assert "truncated" not in result_default

    result_small = _format_state(state, max_chars=100)
    assert "truncated" in result_small


@pytest.mark.asyncio
async def test_llm_judge_evaluate():
    judge = LLMJudge(criteria="Quality and completeness")

    mock_output = {
        "score": 0.75,
        "reasoning": "Decent output",
        "feedback": "Add more detail",
    }

    with patch("fedotmas.optimize._scoring.run_meta_agent_call") as mock_call:
        from fedotmas.meta._adk_runner import LLMCallResult

        mock_call.return_value = LLMCallResult(
            raw_output=mock_output,
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )
        result = await judge.evaluate("test task", {"output": "some result"})

    assert isinstance(result, ScoringResult)
    assert result.score == 0.75
    assert result.feedback == "Add more detail"
    assert result.reasoning == "Decent output"
    assert judge.token_usage == (100, 50)


@pytest.mark.asyncio
async def test_llm_judge_clamps_score():
    judge = LLMJudge()

    mock_output = {"score": 1.5, "reasoning": "Over", "feedback": "None"}
    with patch("fedotmas.optimize._scoring.run_meta_agent_call") as mock_call:
        from fedotmas.meta._adk_runner import LLMCallResult

        mock_call.return_value = LLMCallResult(
            raw_output=mock_output,
            prompt_tokens=10,
            completion_tokens=5,
            elapsed=0.5,
        )
        result = await judge.evaluate("task", {})

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_llm_judge_clamps_negative_score():
    judge = LLMJudge()

    mock_output = {"score": -0.5, "reasoning": "Bad", "feedback": "Fix it"}
    with patch("fedotmas.optimize._scoring.run_meta_agent_call") as mock_call:
        from fedotmas.meta._adk_runner import LLMCallResult

        mock_call.return_value = LLMCallResult(
            raw_output=mock_output,
            prompt_tokens=10,
            completion_tokens=5,
            elapsed=0.5,
        )
        result = await judge.evaluate("task", {})

    assert result.score == 0.0


def test_llm_judge_token_usage_property():
    """token_usage property returns (prompt, completion) tuple."""
    judge = LLMJudge()
    assert judge.token_usage == (0, 0)
    judge._total_prompt_tokens = 42
    judge._total_completion_tokens = 17
    assert judge.token_usage == (42, 17)


def test_llm_judge_max_state_chars():
    """max_state_chars parameter is stored."""
    judge = LLMJudge(max_state_chars=500)
    assert judge._max_state_chars == 500
