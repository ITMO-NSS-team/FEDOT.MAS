from __future__ import annotations

from sampo_phase_5_policy import evaluate_policy_conformance


def _trace() -> dict:
    return {
        "tool_calls": [
            {
                "tool": "prepare_candidate_batch",
                "artifact_id": "a" * 64,
                "retrieval": {
                    "offset": 0,
                    "limit": 2,
                    "methods": ["char_tfidf", "construction_token_tfidf", "word_tfidf"],
                    "k": 50,
                    "fusion": "rrf",
                },
            },
            {"tool": "partition_candidate_batch", "ids": ["1", "2"]},
            {
                "tool": "get_candidate_evidence",
                "ids": ["1"],
                "evidence": {"selection": "diverse_round_robin", "candidate_limit": 10},
            },
            {"tool": "save_review_decisions", "ids": ["1"]},
            {"tool": "save_candidate_predictions", "ids": ["2"]},
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
