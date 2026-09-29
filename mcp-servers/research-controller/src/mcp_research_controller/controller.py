"""Snapshot-driven, domain-neutral research state and action recommendations."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

STATE_VERSION = 1
MAX_ITEMS = 30
MAX_TEXT_CHARS = 240
MAX_QUERY_EVENTS = 50
MAX_TELEMETRY_EVENTS = 50
MAX_SEARCHES = 12
_RECOMMENDATION_ACTIONS = frozenset(
    {"continue_search", "change_strategy", "strategy_blocked", "synthesize"}
)

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "find",
    "for",
    "from",
    "give",
    "in",
    "information",
    "is",
    "latest",
    "look",
    "me",
    "of",
    "on",
    "or",
    "please",
    "query",
    "queries",
    "search",
    "searching",
    "source",
    "the",
    "to",
    "web",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "website",
    "article",
    "paper",
    "publication",
    "study",
    "title",
    "exact",
}
_ALIASES = {
    "accession": "identifier",
    "accessions": "identifier",
    "identification": "identifier",
    "identifier": "identifier",
    "identifiers": "identifier",
    "id": "identifier",
    "ids": "identifier",
    "chief": "ceo",
    "executive": "ceo",
    "officer": "ceo",
}
_STRATEGIES = {
    "browser": {"browser", "navigate", "navigation"},
    "document_extraction": {"pdf", "document", "extract", "extraction"},
    "search": {"search", "query", "queries", "searching"},
    "scraping": {"scrape", "scraping", "scraper"},
}
_LIST_FIELDS = (
    "findings",
    "evidence",
    "evidence_urls",
    "independent_sources",
    "sources_checked",
    "required_fields",
    "filled_fields",
    "unresolved_questions",
    "failed_approaches",
)
_TELEMETRY_DEFAULTS: dict[str, Any] = {
    "controller_calls": 0,
    "recommendation_count": 0,
    "strategy_blocked_events": 0,
    "followed_recommendations": 0,
    "unfollowed_recommendations": 0,
    "unknown_follow_through": 0,
    "recommendations": [],
    "intervention_outcomes": [],
    "pending_recommendation": None,
}


class ResearchController:
    """Pure snapshot transformer; callers own and persist ``research_state``."""

    def update_research_state(
        self,
        *,
        goal: str,
        research_state: dict[str, Any] | None = None,
        research_id: str | None = None,
        findings: list[str] | None = None,
        evidence: list[str] | None = None,
        evidence_urls: list[str] | None = None,
        independent_sources: list[str] | None = None,
        min_sources: int | None = None,
        require_independent_sources: bool | None = None,
        sources_checked: list[str] | None = None,
        required_fields: list[str] | None = None,
        filled_fields: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
        failed_attempts: list[str] | None = None,
        search_queries: list[str] | None = None,
        confidence: float | None = None,
        remaining_budget: float | None = None,
        last_recommendation_followed: bool | None = None,
    ) -> dict[str, Any]:
        """Apply new observations and return the updated JSON snapshot."""
        goal = _clean_text(goal)
        snapshot_id = (
            research_state.get("research_id")
            if isinstance(research_state, dict)
            else None
        )
        state_id = research_id or snapshot_id or "default"
        state_id = _clean_text(state_id)
        if not goal or not state_id:
            raise ValueError("goal and research_id must not be empty")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if remaining_budget is not None and remaining_budget < 0:
            raise ValueError("remaining_budget must not be negative")
        if min_sources is not None and (
            not isinstance(min_sources, int)
            or isinstance(min_sources, bool)
            or min_sources < 1
        ):
            raise ValueError("min_sources must be an integer >= 1")
        if require_independent_sources is not None and not isinstance(
            require_independent_sources, bool
        ):
            raise ValueError("require_independent_sources must be a boolean")

        state = _load_state(research_state, goal=goal, research_id=state_id)
        telemetry = state["telemetry"]
        telemetry["controller_calls"] += 1

        for name, values in (
            ("findings", findings),
            ("evidence", evidence),
            ("evidence_urls", evidence_urls),
            ("independent_sources", independent_sources),
            ("sources_checked", sources_checked),
            ("required_fields", required_fields),
            ("filled_fields", filled_fields),
        ):
            _merge(state[name], values)
        if min_sources is not None:
            state["min_sources"] = min_sources
        if require_independent_sources is not None:
            state["require_independent_sources"] = require_independent_sources
        if unresolved_questions is not None:
            state["unresolved_questions"] = _strings(unresolved_questions)

        queries = _events(search_queries)
        failures = _events(failed_attempts)
        state["search_count"] += len(queries)
        state["search_intents"].extend(_query_profile(query) for query in queries)
        state["search_intents"] = state["search_intents"][-MAX_QUERY_EVENTS:]
        state["failed_attempt_count"] += len(failures)
        for failure in failures:
            strategy = _strategy(failure)
            if strategy:
                counts = state["failed_strategy_counts"]
                counts[strategy] = counts.get(strategy, 0) + _reported_count(failure)
        state["failed_approaches"].extend(failures)
        state["failed_approaches"] = state["failed_approaches"][-MAX_ITEMS:]

        if confidence is not None:
            state["confidence"] = confidence
        if remaining_budget is not None:
            state["remaining_budget"] = remaining_budget
        _finish_recommendation(state, last_recommendation_followed)
        return {"status": "updated", "counts": _counts(state), "research_state": state}

    def get_next_action(self, research_state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the supplied snapshot and return action, reason, and new state."""
        state = _load_state(research_state)
        telemetry = state["telemetry"]
        telemetry["controller_calls"] += 1
        _finish_recommendation(state, None)

        action, reason, guidance = _recommend(state)
        telemetry["recommendation_count"] += 1
        event_id = telemetry["recommendation_count"]
        event = {
            "id": event_id,
            "action": action,
            "reason": reason,
            "searches_before": state["search_count"],
            "searches_after": None,
            "followed": None,
        }
        telemetry["recommendations"].append(event)
        telemetry["recommendations"] = telemetry["recommendations"][
            -MAX_TELEMETRY_EVENTS:
        ]
        telemetry["pending_recommendation"] = {
            "id": event_id,
            "action": action,
            "searches_before": state["search_count"],
        }
        if action == "strategy_blocked":
            telemetry["strategy_blocked_events"] += 1

        return {
            "action": action,
            "reason": reason,
            "guidance": guidance,
            "research_state": state,
        }


def _new_state(goal: str = "", research_id: str = "default") -> dict[str, Any]:
    state: dict[str, Any] = {
        "version": STATE_VERSION,
        "research_id": research_id,
        "goal": goal,
        **{name: [] for name in _LIST_FIELDS},
        "search_intents": [],
        "search_count": 0,
        "min_sources": 1,
        "require_independent_sources": False,
        "failed_attempt_count": 0,
        "failed_strategy_counts": {},
        "confidence": None,
        "remaining_budget": None,
        "telemetry": deepcopy(_TELEMETRY_DEFAULTS),
    }
    return state


def _load_state(
    snapshot: dict[str, Any] | None,
    *,
    goal: str | None = None,
    research_id: str | None = None,
) -> dict[str, Any]:
    if snapshot is None:
        if goal is None:
            raise ValueError("research_state is required; initialize it first")
        return _new_state(goal, research_id or "default")
    if not isinstance(snapshot, dict):
        raise TypeError("research_state must be a JSON object")

    state = _new_state()
    known_fields = set(state)
    state.update(
        {key: deepcopy(value) for key, value in snapshot.items() if key in known_fields}
    )
    if (
        not isinstance(state["version"], int)
        or isinstance(state["version"], bool)
        or state["version"] != STATE_VERSION
    ):
        raise ValueError("unsupported research_state version")
    if not isinstance(state["goal"], str) or not _clean_text(state["goal"]):
        raise ValueError("research_state goal must be a nonempty string")
    if not isinstance(state["research_id"], str) or not _clean_text(
        state["research_id"]
    ):
        raise ValueError("research_state research_id must be a nonempty string")
    state["goal"] = _clean_text(state["goal"])
    state["research_id"] = _clean_text(state["research_id"])
    if goal is not None and _key(state["goal"]) != _key(goal):
        raise ValueError(
            "research_id already belongs to a different goal; use a new id"
        )
    if research_id is not None and state["research_id"] != research_id:
        raise ValueError("research_state has a different research_id")

    confidence = state.get("confidence")
    if (
        isinstance(confidence, int | float)
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        and 0 <= confidence <= 1
    ):
        state["confidence"] = float(confidence)
    else:
        state["confidence"] = None
    budget = state.get("remaining_budget")
    if (
        isinstance(budget, int | float)
        and not isinstance(budget, bool)
        and math.isfinite(budget)
        and budget >= 0
    ):
        state["remaining_budget"] = budget
    else:
        state["remaining_budget"] = None

    for name in _LIST_FIELDS:
        state[name] = _strings(state.get(name, []))
    state["failed_approaches"] = state["failed_approaches"][-MAX_ITEMS:]
    raw_intents = state.get("search_intents")
    state["search_intents"] = (
        [
            {
                "terms": _strings(profile.get("terms")),
                "entities": _strings(profile.get("entities")),
            }
            for profile in raw_intents
            if isinstance(profile, dict) and _strings(profile.get("terms"))
        ][-MAX_QUERY_EVENTS:]
        if isinstance(raw_intents, list)
        else []
    )
    state["search_count"] = _nonnegative_int(state.get("search_count"))
    min_sources = state.get("min_sources", 1)
    state["min_sources"] = (
        min_sources
        if isinstance(min_sources, int)
        and not isinstance(min_sources, bool)
        and min_sources >= 1
        else 1
    )
    state["require_independent_sources"] = state.get(
        "require_independent_sources"
    ) is True
    state["failed_attempt_count"] = _nonnegative_int(
        state.get("failed_attempt_count")
    )
    strategies = state.get("failed_strategy_counts", {})
    state["failed_strategy_counts"] = (
        {
            _clean_text(key): _nonnegative_int(value)
            for key, value in strategies.items()
            if isinstance(key, str) and _clean_text(key)
        }
        if isinstance(strategies, dict)
        else {}
    )
    telemetry = deepcopy(_TELEMETRY_DEFAULTS)
    existing_telemetry = state.get("telemetry")
    if isinstance(existing_telemetry, dict):
        telemetry.update(existing_telemetry)
    for field in (
        "controller_calls",
        "recommendation_count",
        "strategy_blocked_events",
        "followed_recommendations",
        "unfollowed_recommendations",
        "unknown_follow_through",
    ):
        telemetry[field] = _nonnegative_int(telemetry[field])
    events = telemetry.get("recommendations")
    telemetry["recommendations"] = _sanitize_recommendations(events)
    outcomes = telemetry.get("intervention_outcomes")
    telemetry["intervention_outcomes"] = _sanitize_outcomes(outcomes)
    telemetry["pending_recommendation"] = _sanitize_pending_recommendation(
        telemetry.get("pending_recommendation")
    )
    state["telemetry"] = telemetry
    return state


def _recommend(state: dict[str, Any]) -> tuple[str, str, str]:
    if _evidence_is_sufficient(state):
        return (
            "synthesize",
            "evidence_requirements_satisfied",
            "Synthesize findings with citations and preserve the research_state in the handoff.",
        )
    if state["remaining_budget"] == 0:
        return (
            "synthesize",
            "search_budget_exhausted",
            "Synthesize current evidence and report unresolved gaps; do not imply completeness.",
        )

    similar_queries = _largest_similar_group(state["search_intents"])
    repeated_strategy = max(state["failed_strategy_counts"].values(), default=0)
    if (
        similar_queries >= 3
        or repeated_strategy >= 3
        or state["failed_attempt_count"] >= 5
        or state["search_count"] >= MAX_SEARCHES
    ):
        return (
            "strategy_blocked",
            "repeated_low_value_search",
            "Do not repeat this strategy. Use another evidence source/tool or synthesize current findings; this does not stop other research.",
        )
    if similar_queries >= 2 or repeated_strategy >= 2:
        return (
            "change_strategy",
            "repeated_low_value_search",
            "Change query framing or evidence method before another expensive search.",
        )
    if not state["unresolved_questions"]:
        return (
            "continue_search",
            "evidence_requirements_incomplete",
            "Check which reported requirements are missing; make one focused check if needed.",
        )
    return (
        "continue_search",
        "unresolved_questions",
        "Target the highest-value unresolved question and record the result.",
    )


def _evidence_is_sufficient(state: dict[str, Any]) -> bool:
    if state["unresolved_questions"] or not state["evidence"]:
        return False
    if set(state["required_fields"]) - set(state["filled_fields"]):
        return False
    if state["require_independent_sources"]:
        sources = set(state["independent_sources"])
    else:
        sources = set(state["independent_sources"])
        if not sources:
            sources = {_source_key(url) for url in state["evidence_urls"]} - {""}
    return len(sources) >= state["min_sources"]


def _finish_recommendation(state: dict[str, Any], followed: bool | None) -> None:
    if followed is not None and not isinstance(followed, bool):
        followed = None
    telemetry = state.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = deepcopy(_TELEMETRY_DEFAULTS)
        state["telemetry"] = telemetry
    pending = _sanitize_pending_recommendation(
        telemetry.get("pending_recommendation")
    )
    telemetry["pending_recommendation"] = pending
    if not pending:
        return
    telemetry["recommendations"] = _sanitize_recommendations(
        telemetry.get("recommendations")
    )
    telemetry["intervention_outcomes"] = _sanitize_outcomes(
        telemetry.get("intervention_outcomes")
    )
    for field in (
        "controller_calls",
        "recommendation_count",
        "strategy_blocked_events",
        "followed_recommendations",
        "unfollowed_recommendations",
        "unknown_follow_through",
    ):
        telemetry[field] = _nonnegative_int(telemetry.get(field))
    state["search_count"] = _nonnegative_int(state.get("search_count"))
    outcome = {
        "recommendation_id": pending["id"],
        "action": pending["action"],
        "searches_before": pending["searches_before"],
        "searches_after": state["search_count"],
        "followed": followed,
    }
    telemetry["intervention_outcomes"].append(outcome)
    telemetry["intervention_outcomes"] = telemetry["intervention_outcomes"][
        -MAX_TELEMETRY_EVENTS:
    ]
    for event in reversed(telemetry["recommendations"]):
        if event["id"] == pending["id"]:
            event["searches_after"] = state["search_count"]
            event["followed"] = followed
            break
    counter = {
        True: "followed_recommendations",
        False: "unfollowed_recommendations",
        None: "unknown_follow_through",
    }[followed]
    telemetry[counter] += 1
    telemetry["pending_recommendation"] = None


def _nonnegative_int(value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    return 0


def _sanitize_pending_recommendation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    recommendation_id = value.get("id")
    action = value.get("action")
    searches_before = value.get("searches_before")
    if (
        _nonnegative_int(recommendation_id, minimum=1) == 0
        or not isinstance(action, str)
        or action not in _RECOMMENDATION_ACTIONS
        or _nonnegative_int(searches_before) != searches_before
    ):
        return None
    return {
        "id": recommendation_id,
        "action": action,
        "searches_before": searches_before,
    }


def _sanitize_recommendations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    events = []
    for event in value:
        if not isinstance(event, dict):
            continue
        recommendation_id = event.get("id")
        action = event.get("action")
        searches_before = event.get("searches_before")
        if (
            _nonnegative_int(recommendation_id, minimum=1) == 0
            or not isinstance(action, str)
            or action not in _RECOMMENDATION_ACTIONS
            or _nonnegative_int(searches_before) != searches_before
        ):
            continue
        searches_after = event.get("searches_after")
        if searches_after is not None and _nonnegative_int(searches_after) != searches_after:
            searches_after = None
        followed = event.get("followed")
        if followed is not None and not isinstance(followed, bool):
            followed = None
        events.append(
            {
                "id": recommendation_id,
                "action": action,
                "reason": event.get("reason")
                if isinstance(event.get("reason"), str)
                else "",
                "searches_before": searches_before,
                "searches_after": searches_after,
                "followed": followed,
            }
        )
    return events[-MAX_TELEMETRY_EVENTS:]


def _sanitize_outcomes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    outcomes = []
    for event in value:
        if not isinstance(event, dict):
            continue
        recommendation_id = event.get("recommendation_id")
        action = event.get("action")
        searches_before = event.get("searches_before")
        searches_after = event.get("searches_after")
        followed = event.get("followed")
        if (
            _nonnegative_int(recommendation_id, minimum=1) == 0
            or not isinstance(action, str)
            or action not in _RECOMMENDATION_ACTIONS
            or _nonnegative_int(searches_before) != searches_before
            or _nonnegative_int(searches_after) != searches_after
            or (followed is not None and not isinstance(followed, bool))
        ):
            continue
        outcomes.append(
            {
                "recommendation_id": recommendation_id,
                "action": action,
                "searches_before": searches_before,
                "searches_after": searches_after,
                "followed": followed,
            }
        )
    return outcomes[-MAX_TELEMETRY_EVENTS:]


def _counts(state: dict[str, Any]) -> dict[str, int]:
    source_count = len(set(state["independent_sources"]))
    if not source_count:
        source_count = len({_source_key(url) for url in state["evidence_urls"]} - {""})
    return {
        "findings": len(state["findings"]),
        "evidence": len(state["evidence"]),
        "evidence_urls": len(state["evidence_urls"]),
        "independent_sources": source_count,
        "unresolved_questions": len(state["unresolved_questions"]),
        "failed_attempts": state["failed_attempt_count"],
        "searches": state["search_count"],
    }


def _merge(destination: list[str], values: list[str] | None) -> None:
    known = {_key(value) for value in destination}
    for value in _strings(values):
        if _key(value) not in known:
            destination.append(value)
            known.add(_key(value))
    del destination[:-MAX_ITEMS]


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    known: set[str] = set()
    for value in values:
        if (
            isinstance(value, str)
            and (text := _clean_text(value))
            and _key(text) not in known
        ):
            result.append(text)
            known.add(_key(text))
    return result[-MAX_ITEMS:]


def _events(values: list[str] | None) -> list[str]:
    return [text for value in values or [] if (text := _clean_text(value))]


def _query_profile(query: str) -> dict[str, list[str]]:
    query = unicodedata.normalize("NFKC", query)
    entities = re.findall(r"[\"']([^\"']+)[\"']", query)
    entities += re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query)
    entity_terms = {
        term for phrase in entities for word in _words(phrase) if (term := _term(word))
    }
    terms = {_term(token) for token in _words(query) if _term(token)}
    return {"terms": sorted(terms), "entities": sorted(entity_terms)}


def _words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)


def _term(token: str) -> str:
    if token in _STOP_WORDS or len(token) < 2:
        return ""
    if token in _ALIASES:
        return _ALIASES[token]
    if token.endswith("ies") and len(token) > 5:
        token = f"{token[:-3]}y"
    elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    return _ALIASES.get(token, token)


def _largest_similar_group(profiles: list[dict[str, Any]]) -> int:
    parents = list(range(len(profiles)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(profiles)):
        left_terms = set(profiles[left].get("terms", []))
        for right in range(left + 1, len(profiles)):
            right_terms = set(profiles[right].get("terms", []))
            shared = len(left_terms & right_terms)
            if min(len(left_terms), len(right_terms)) < 2 or shared < 1:
                continue
            coverage = shared / min(len(left_terms), len(right_terms))
            union = len(left_terms | right_terms)
            jaccard = shared / union if union else 0
            entity_match = bool(
                set(profiles[left].get("entities", []))
                & set(profiles[right].get("entities", []))
            )
            if (shared >= 2 and (coverage >= 0.7 or jaccard >= 0.42)) or (
                entity_match and coverage >= 0.5
            ):
                parents[root(right)] = root(left)
    return max(
        Counter(root(index) for index in range(len(parents))).values(), default=0
    )


def _strategy(attempt: str) -> str | None:
    words = set(_words(attempt))
    return next(
        (name for name, terms in _STRATEGIES.items() if words & terms),
        None,
    )


def _reported_count(attempt: str) -> int:
    matches = re.findall(r"\b(\d+)\s*(?:times|attempts|retries)\b", attempt.casefold())
    return max([1, *(int(match) for match in matches)])


def _source_key(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    except ValueError:
        return ""


def _clean_text(value: str) -> str:
    return " ".join(str(value).split())[:MAX_TEXT_CHARS]


def _key(value: str) -> str:
    return " ".join(value.split()).casefold()
