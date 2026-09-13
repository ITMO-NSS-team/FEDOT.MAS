from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_sampo_benchmark import server


def test_retrieval_catalog_has_only_uniform_methods() -> None:
    assert server.list_methods() == {
        "methods": [
            "bm25_token",
            "char_tfidf",
            "char_word_fusion",
            "construction_token_tfidf",
            "word_tfidf",
        ],
        "max_batch_size": 100,
        "max_k": 50,
        "result_fields": ["example_id", "label", "rank", "score"],
    }
    assert not hasattr(server, "run_tfidf_baseline")


def test_retrieve_candidates_returns_method_local_rank_and_score(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_inputs",
        lambda filename="benchmark_inputs.csv": [
            {"example_id": "one", "raw_work_name": "source"}
        ],
    )
    monkeypatch.setattr(server, "_labels", lambda: ["alpha", "beta"])
    monkeypatch.setitem(
        server.RETRIEVERS,
        "char_tfidf",
        lambda examples, labels, k: [[("beta", 0.8), ("alpha", 0.2)]][:k],
    )

    assert server.retrieve_candidates(["one"], "char_tfidf", 2) == {
        "method": "char_tfidf",
        "candidates": [
            {
                "example_id": "one",
                "candidates": [
                    {"label": "beta", "rank": 1, "score": 0.8},
                    {"label": "alpha", "rank": 2, "score": 0.2},
                ],
            }
        ],
    }
