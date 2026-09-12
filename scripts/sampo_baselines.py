"""Public SAMPO baseline inference code.

This module intentionally accepts only public examples and allowed labels. It
does not open databases, mapper tables, or private ground truth files.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from difflib import SequenceMatcher

Prediction = list[str | None]
RankedPrediction = list[tuple[str, float]]


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


def normalized_exact(
    examples: list[str], labels: list[str], top_k: int = 3
) -> list[Prediction]:
    index: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        index[normalize(label)].append(label)
    return [
        (index.get(normalize(value), []) + [None] * top_k)[:top_k] for value in examples
    ]


def fuzzy(examples: list[str], labels: list[str], top_k: int = 3) -> list[Prediction]:
    normalized_labels = [normalize(label) for label in labels]
    results = []
    for example in examples:
        source = normalize(example)
        ranked = sorted(
            (
                (SequenceMatcher(None, source, label).quick_ratio(), index)
                for index, label in enumerate(normalized_labels)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        results.append([labels[index] for _, index in ranked[:top_k]])
    return results


def _char_ngrams(value: str, minimum: int = 3, maximum: int = 5) -> Counter[str]:
    value = f"^{normalize(value)}$"
    return Counter(
        value[start : start + size]
        for size in range(minimum, maximum + 1)
        for start in range(len(value) - size + 1)
    )


def _word_tokens(value: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _tfidf_ranked(
    examples: list[str], labels: list[str], term_fn: Callable[[str], Counter[str]], k: int
) -> list[RankedPrediction]:
    """Return deterministic sparse cosine TF-IDF rankings and scores."""
    if k < 1:
        raise ValueError("k must be at least 1")
    label_terms = [term_fn(label) for label in labels]
    document_frequency: Counter[str] = Counter()
    for terms in label_terms:
        document_frequency.update(terms.keys())
    total = len(labels)
    idf = {
        term: math.log((total + 1) / (frequency + 1)) + 1
        for term, frequency in document_frequency.items()
    }
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    label_norms = []
    for index, terms in enumerate(label_terms):
        weights = {term: count * idf[term] for term, count in terms.items()}
        label_norms.append(
            math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        )
        for term, weight in weights.items():
            postings[term].append((index, weight))

    results = []
    for example in examples:
        terms = term_fn(example)
        weights = {
            term: count * idf[term] for term, count in terms.items() if term in idf
        }
        source_norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        scores: defaultdict[int, float] = defaultdict(float)
        for term, weight in weights.items():
            for index, label_weight in postings[term]:
                scores[index] += weight * label_weight
        ranked = sorted(
            (
                (scores.get(index, 0.0) / (source_norm * label_norms[index]), labels[index])
                for index in range(len(labels))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        results.append([(label, score) for score, label in ranked[:k]])
    return results


def tfidf_char_ngrams_ranked(
    examples: list[str], labels: list[str], k: int
) -> list[RankedPrediction]:
    """Rank labels by character 3--5 gram TF-IDF cosine similarity."""
    return _tfidf_ranked(examples, labels, _char_ngrams, k)


def tfidf_word_ranked(
    examples: list[str], labels: list[str], k: int
) -> list[RankedPrediction]:
    """Rank labels by Unicode word-token TF-IDF cosine similarity."""
    return _tfidf_ranked(examples, labels, _word_tokens, k)


def tfidf_char_word_hybrid_ranked(
    examples: list[str], labels: list[str], k: int
) -> list[RankedPrediction]:
    """Fuse char and word rankings with equal-weight reciprocal-rank fusion."""
    candidate_count = len(labels)
    char_rankings = tfidf_char_ngrams_ranked(examples, labels, candidate_count)
    word_rankings = tfidf_word_ranked(examples, labels, candidate_count)
    results = []
    for char_ranking, word_ranking in zip(char_rankings, word_rankings):
        fused: defaultdict[str, float] = defaultdict(float)
        # Fixed RRF constant 60 and equal sources; no private-GT tuning.
        for ranking in (char_ranking, word_ranking):
            for rank, (label, _) in enumerate(ranking, start=1):
                fused[label] += 1 / (60 + rank)
        results.append(sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k])
    return results


def tfidf_char_ngrams(
    examples: list[str], labels: list[str], top_k: int = 3
) -> list[Prediction]:
    """Sparse cosine retrieval over TF-IDF character 3--5 grams."""
    ranked = tfidf_char_ngrams_ranked(examples, labels, top_k)
    return [
        [label for label, score in row if score > 0]
        + [None] * max(0, top_k - len([score for _, score in row if score > 0]))
        for row in ranked
    ]


BASELINES: dict[str, Callable[[list[str], list[str], int], list[Prediction]]] = {
    "normalized_exact": normalized_exact,
    "fuzzy": fuzzy,
    "tfidf_char_ngrams": tfidf_char_ngrams,
}
