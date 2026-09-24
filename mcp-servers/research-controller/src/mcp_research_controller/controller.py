"""Small, domain-neutral state store and action recommender for research work."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from threading import RLock

MAX_ITEMS = 100
MAX_ITEM_CHARS = 500
MAX_SEARCH_QUERIES = 12
REPEATED_INTENT_THRESHOLD = 3
FAILED_ATTEMPT_THRESHOLD = 5
REPEATED_STRATEGY_THRESHOLD = 3

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "find",
    "in",
    "is",
    "of",
    "on",
    "or",
    "search",
    "the",
    "to",
    "web",
    "with",
    "query",
    "queries",
    "exact",
    "title",
    "paper",
}
_STRATEGIES = {
    "browser": {"browser", "navigate", "navigation"},
    "document extraction": {"pdf", "document", "extract", "extraction"},
    "search": {"search", "query", "queries", "searching"},
    "scraping": {"scrape", "scraping", "scraper"},
}


@dataclass
class ResearchState:
    """Bounded state for a single research workstream."""

    goal: str
    findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    sources_checked: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    confidence: float | None = None
    remaining_budget: float | None = None


class ResearchController:
    """Maintain compact research state and recommend a next action."""

    def __init__(self) -> None:
        self._states: dict[str, ResearchState] = {}
        self._lock = RLock()

    def update_research_state(
        self,
        *,
        goal: str,
        research_id: str = "default",
        findings: list[str] | None = None,
        evidence: list[str] | None = None,
        sources_checked: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
        failed_attempts: list[str] | None = None,
        search_queries: list[str] | None = None,
        confidence: float | None = None,
        remaining_budget: float | None = None,
    ) -> dict[str, object]:
        """Merge new observations and replace the current unresolved-gap list."""
        clean_goal = _clean_text(goal)
        clean_id = _clean_text(research_id)
        if not clean_goal:
            raise ValueError("goal must not be empty")
        if not clean_id:
            raise ValueError("research_id must not be empty")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if remaining_budget is not None and remaining_budget < 0:
            raise ValueError("remaining_budget must not be negative")

        with self._lock:
            state = self._states.get(clean_id)
            if state is None:
                state = ResearchState(goal=clean_goal)
                self._states[clean_id] = state
            elif _comparison_text(state.goal) != _comparison_text(clean_goal):
                raise ValueError(
                    "research_id already belongs to a different goal; use a new id"
                )

            _merge_items(state.findings, findings)
            _merge_items(state.evidence, evidence)
            _merge_items(state.sources_checked, sources_checked)
            _append_events(state.failed_attempts, failed_attempts)
            _append_events(
                state.search_queries, search_queries, limit=MAX_SEARCH_QUERIES
            )
            if unresolved_questions is not None:
                state.unresolved_questions = _clean_items(unresolved_questions)
            if confidence is not None:
                state.confidence = float(confidence)
            if remaining_budget is not None:
                state.remaining_budget = float(remaining_budget)

            return {
                "research_id": clean_id,
                "status": "updated",
                "counts": _state_counts(state),
                "confidence": state.confidence,
                "remaining_budget": state.remaining_budget,
            }

    def get_next_action(self, research_id: str = "default") -> dict[str, object]:
        """Return a compact continue, change-strategy, or synthesize recommendation."""
        clean_id = _clean_text(research_id)
        with self._lock:
            state = self._states.get(clean_id)
            if state is None:
                return {
                    "research_id": clean_id,
                    "decision": "update_state",
                    "reason": "No research state is recorded.",
                    "next_actions": [
                        "Record the goal, evidence, sources, open questions, and budget."
                    ],
                }

            if _evidence_is_sufficient(state):
                return {
                    "research_id": clean_id,
                    "decision": "synthesize",
                    "reason": "Evidence is sufficient and no gaps remain.",
                    "next_actions": [
                        "Synthesize the findings with source support and note uncertainty."
                    ],
                }

            if state.remaining_budget == 0:
                return {
                    "research_id": clean_id,
                    "decision": "synthesize",
                    "reason": "No research budget remains; report current limits.",
                    "next_actions": [
                        "Synthesize available evidence and state unresolved gaps clearly."
                    ],
                }

            repeated_intent = _largest_similar_query_group(state.search_queries)
            if repeated_intent >= REPEATED_INTENT_THRESHOLD:
                return _change_strategy(
                    clean_id,
                    f"{repeated_intent} searches share the same semantic intent.",
                    state,
                )

            strategy_counts: Counter[str] = Counter()
            for attempt in state.failed_attempts:
                strategy = _strategy_name(attempt)
                if strategy is not None:
                    strategy_counts[strategy] += _reported_attempts(attempt)
            repeated_strategy = max(strategy_counts.values(), default=0)
            if (
                repeated_strategy >= REPEATED_STRATEGY_THRESHOLD
                or len(state.failed_attempts) >= FAILED_ATTEMPT_THRESHOLD
                or len(state.search_queries) >= MAX_SEARCH_QUERIES
            ):
                return _change_strategy(
                    clean_id,
                    "Repeated failures or a high search volume make the current approach low-value.",
                    state,
                )

            if not state.unresolved_questions:
                next_actions = [
                    "Check whether existing evidence supports the goal.",
                    "Add a focused evidence check only if a material gap remains.",
                ]
                reason = "No open questions are recorded, but evidence is not yet sufficient."
            else:
                next_actions = [
                    "Choose the highest-value unresolved question.",
                    "Try a distinct source or evidence method; record the result before more searches.",
                ]
                reason = "Open questions remain and the current search budget allows more work."
            return {
                "research_id": clean_id,
                "decision": "continue_search",
                "reason": reason,
                "next_actions": next_actions,
            }


def _change_strategy(
    research_id: str, reason: str, state: ResearchState
) -> dict[str, object]:
    next_actions = [
        "Stop repeating the same search intent or failed approach.",
        "Inspect sources already checked, or switch to a distinct evidence method.",
    ]
    if state.unresolved_questions:
        next_actions.append("Target the highest-value unresolved question.")
    return {
        "research_id": research_id,
        "decision": "change_strategy",
        "reason": reason,
        "next_actions": next_actions,
    }


def _evidence_is_sufficient(state: ResearchState) -> bool:
    if state.unresolved_questions or not state.evidence:
        return False
    if state.confidence is not None and state.confidence >= 0.8:
        return True
    return len(state.evidence) >= 2 and len(state.sources_checked) >= 2


def _state_counts(state: ResearchState) -> dict[str, int]:
    return {
        "findings": len(state.findings),
        "evidence": len(state.evidence),
        "sources_checked": len(state.sources_checked),
        "unresolved_questions": len(state.unresolved_questions),
        "failed_attempts": len(state.failed_attempts),
        "search_queries": len(state.search_queries),
    }


def _clean_items(values: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = _clean_text(value)
        key = _comparison_text(item)
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned[-MAX_ITEMS:]


def _merge_items(
    destination: list[str], values: list[str] | None, *, limit: int = MAX_ITEMS
) -> None:
    known = {_comparison_text(item) for item in destination}
    for item in _clean_items(values):
        key = _comparison_text(item)
        if key not in known:
            destination.append(item)
            known.add(key)
    if len(destination) > limit:
        del destination[:-limit]


def _append_events(
    destination: list[str], values: list[str] | None, *, limit: int = MAX_ITEMS
) -> None:
    for value in values or []:
        item = _clean_text(value)
        if item:
            destination.append(item)
    if len(destination) > limit:
        del destination[:-limit]


def _clean_text(value: str) -> str:
    return " ".join(str(value).split())[:MAX_ITEM_CHARS]


def _comparison_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _query_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    normalized: set[str] = set()
    for token in tokens:
        if token in _STOP_WORDS:
            continue
        if token == "ids":
            token = "id"
        elif token.endswith("ies") and len(token) > 5:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        normalized.add(token)
    return normalized


def _largest_similar_query_group(queries: list[str]) -> int:
    token_sets = [_query_tokens(query) for query in queries]
    parents = list(range(len(token_sets)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(token_sets)):
        if len(token_sets[left]) < 2:
            continue
        for right in range(left + 1, len(token_sets)):
            if len(token_sets[right]) < 2:
                continue
            shared = len(token_sets[left] & token_sets[right])
            union = len(token_sets[left] | token_sets[right])
            if shared >= 2 and union and shared / union >= 0.5:
                left_root, right_root = find(left), find(right)
                parents[right_root] = left_root

    counts = Counter(find(index) for index in range(len(parents)))
    return max(counts.values(), default=0)


def _strategy_name(attempt: str) -> str | None:
    lowered = attempt.casefold()
    for strategy, terms in _STRATEGIES.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            return strategy
    return None


def _reported_attempts(attempt: str) -> int:
    matches = re.findall(r"\b(\d+)\s*(?:times|attempts|retries)\b", attempt.casefold())
    return max([1, *(int(match) for match in matches)])
