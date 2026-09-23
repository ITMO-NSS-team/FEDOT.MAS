from __future__ import annotations

import pytest

from sampo_phase_5_policy import evaluate_policy_conformance
from sampo_phase_5_trace import read_trace


def _trace() -> dict:
    return {
        "tool_calls": [
            {
                "tool": "prepare_candidate_batch",
                "artifact_id": "a" * 64,
                "retrieval": {
                    "offset": 0,
                    "limit": 2,
                    "methods": ["bm25_token", "char_tfidf", "char_word_fusion", "construction_token_tfidf", "word_tfidf"],
                    "k": 5,
                    "fusion": "rrf",
                },
            },
            {"tool": "partition_candidate_batch", "ids": ["1", "2"], "artifact_id": "a" * 64},
            {
                "tool": "get_candidate_evidence",
                "ids": ["1"],
                "artifact_id": "a" * 64,
                "evidence": {"selection": "fused", "candidate_limit": 10},
            },
            {"tool": "save_review_decisions", "ids": ["1"], "artifact_id": "a" * 64},
            {"tool": "save_candidate_predictions", "ids": ["2"], "artifact_id": "a" * 64},
        ]
    }


def test_policy_conformance_accepts_disagreement_partition(monkeypatch):
    monkeypatch.setattr(
        "sampo_phase_5_policy._artifact",
        lambda _: {
            "examples": [
                {
                    "example_id": "1",
                    "method_candidates": {
                        "alpha": [{"method": "char_tfidf", "rank": 1}],
                        "beta": [{"method": "construction_token_tfidf", "rank": 1}],
                    },
                },
                {
                    "example_id": "2",
                    "method_candidates": {
                        "alpha": [
                            {"method": "char_tfidf", "rank": 1},
                            {"method": "construction_token_tfidf", "rank": 1},
                            {"method": "word_tfidf", "rank": 1},
                        ]
                    },
                },
            ]
        },
    )
    result = evaluate_policy_conformance(_trace(), ["1", "2"])
    assert result["pass"] is True
    assert result["expected_review_ids"] == ["1"]


def test_policy_conformance_rejects_fallback_all_ids(monkeypatch):
    monkeypatch.setattr(
        "sampo_phase_5_policy._artifact",
        lambda _: {
            "examples": [
                {
                    "example_id": "1",
                    "method_candidates": {
                        "alpha": [{"method": "char_tfidf", "rank": 1}],
                        "beta": [{"method": "word_tfidf", "rank": 1}],
                    },
                },
                {
                    "example_id": "2",
                    "method_candidates": {
                        "alpha": [{"method": "char_tfidf", "rank": 1}]
                    },
                },
            ]
        },
    )
    trace = _trace()
    trace["tool_calls"] = [trace["tool_calls"][0], trace["tool_calls"][1], {"tool": "save_candidate_predictions", "ids": ["1", "2"]}]
    result = evaluate_policy_conformance(trace, ["1", "2"])
    assert result["pass"] is False


def test_policy_conformance_rejects_mixed_artifacts(monkeypatch):
    monkeypatch.setattr(
        "sampo_phase_5_policy._artifact",
        lambda _: {
            "examples": [
                {"example_id": "1", "method_candidates": {"alpha": [{"method": "char_tfidf", "rank": 1}]}},
                {"example_id": "2", "method_candidates": {"alpha": [{"method": "char_tfidf", "rank": 1}]}},
            ]
        },
    )
    trace = _trace()
    trace["tool_calls"][2]["artifact_id"] = "b" * 64
    result = evaluate_policy_conformance(trace, ["1", "2"])
    assert result["pass"] is False
    assert any("prepare artifact_id" in reason for reason in result["reasons"])


def test_policy_conformance_rejects_duplicate_evidence(monkeypatch):
    monkeypatch.setattr(
        "sampo_phase_5_policy._artifact",
        lambda _: {
            "examples": [
                {
                    "example_id": "1",
                    "method_candidates": {
                        "alpha": [{"method": "char_tfidf", "rank": 1}],
                        "beta": [{"method": "bm25_token", "rank": 1}],
                    },
                },
                {"example_id": "2", "method_candidates": {"alpha": [{"method": "char_tfidf", "rank": 1}]}},
            ]
        },
    )
    trace = _trace()
    evidence = trace["tool_calls"][2]
    trace["tool_calls"].insert(3, dict(evidence))
    result = evaluate_policy_conformance(trace, ["1", "2"])
    assert result["pass"] is False
    assert any("exactly one evidence call" in reason for reason in result["reasons"])


@pytest.mark.parametrize("contents", ["", '{"tool_calls": ['])
def test_read_trace_marks_existing_empty_or_partial_file_unreadable(tmp_path, contents):
    trace = tmp_path / "trace.json"
    trace.write_text(contents, encoding="utf-8")

    result = read_trace(trace, attempts=1, delay_s=0)

    assert result["failures"] == ["trace_unreadable"]
