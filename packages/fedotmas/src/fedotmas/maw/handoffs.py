"""Helpers for validating optional, generated MAW artifact contracts."""

from __future__ import annotations

import json
import re
from typing import Any

from fedotmas.maw.models import ArtifactContract, ArtifactRequirement

EXECUTION_METADATA_KEY = "_fedotmas_execution"
ABSTENTION_STATE_KEY = "__fedotmas_completion"


def is_explicit_abstention(value: Any) -> bool:
    """Recognize the runtime's explicit terminal abstention protocol."""
    if isinstance(value, str):
        return value.strip().startswith("<abstain>") and "</abstain>" in value
    if isinstance(value, dict):
        return value.get("status") == "abstained" and isinstance(
            value.get("reason"), str
        )
    return False


_UNRESOLVED_TEXT = re.compile(
    r"^\s*unresolved\s*$|upstream.{0,100}explicitly unresolved|"
    r"\bno exact answer\b|\bno answer to verify\b|"
    r"\bexact optimum was not computed\b",
    re.IGNORECASE,
)
_ANSWER_NULL_KEYS = {"answer", "verified_answer", "value", "formatted_answer"}


def is_unresolved_answer(value: Any) -> bool:
    """Detect explicit terminal non-answers; structured null state takes priority."""
    parsed_candidates = parse_artifact_candidates(value)
    if parsed_candidates:

        if any(
            str(parsed.get("status", "")).casefold() == "unresolved"
            for parsed in parsed_candidates
        ):
            return True
        # A null nested candidate is not evidence that the actual answer is
        # unresolved. Only a top-level answer field has that meaning.
        for parsed in parsed_candidates:
            if any(parsed.get(key) is None for key in _ANSWER_NULL_KEYS if key in parsed):
                return True
    if isinstance(value, str):
        if _UNRESOLVED_TEXT.search(value):
            return True
        return False
    return False


def parse_artifact_candidates(value: Any) -> list[dict[str, Any]]:
    """Return top-level JSON object candidates in source order."""
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    candidates = []

    def add_array_items(item: Any) -> None:
        if isinstance(item, dict):
            candidates.append(item)
        elif isinstance(item, list):
            for child in item:
                add_array_items(child)

    index = 0
    while index < len(value):
        if value[index] not in "{[":
            index += 1
            continue
        try:
            candidate, end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(candidate, dict):
            candidates.append(candidate)
        elif isinstance(candidate, list):
            add_array_items(candidate)
        index += max(end, 1)
    return candidates


def parse_artifact(
    value: Any, contract: ArtifactContract | None = None
) -> dict[str, Any] | None:
    """Parse strict JSON or one unambiguous JSON object embedded in prose."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    candidates = parse_artifact_candidates(value)
    if contract is not None:
        fields = list(
            dict.fromkeys([*contract.required_fields, *contract.identity_fields])
        )
        matching = [
            candidate
            for candidate in candidates
            if not _missing_fields_in_artifact(candidate, fields)
        ]
        if matching:
            return matching[-1]
    return candidates[0] if len(candidates) == 1 else None


def _missing_fields_in_artifact(
    artifact: dict[str, Any], fields: list[str]
) -> list[str]:
    return [field for field in fields if not _field_is_present(artifact, field)]


def missing_contract_fields(
    value: Any, required_fields: list[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    artifact = parse_artifact(value)
    if artifact is None:
        return None, list(required_fields) or ["structured artifact"]
    missing = [
        field for field in required_fields if not _field_is_present(artifact, field)
    ]
    return artifact, missing


def field_values(value: Any, path: str) -> tuple[list[Any], bool]:
    """Resolve a dotted path with ``[]`` list expansion, preserving every item."""
    segments = path.split(".")
    if not segments or any(not segment for segment in segments):
        return [], False
    current = [value]
    for segment in segments:
        repeated = segment.endswith("[]")
        key = segment[:-2] if repeated else segment
        if not key:
            return [], False
        next_values = []
        for item in current:
            if not isinstance(item, dict) or key not in item:
                return [], False
            found = item[key]
            if repeated:
                if not isinstance(found, list) or not found:
                    return [], False
                next_values.extend(found)
            else:
                next_values.append(found)
        current = next_values
    return current, bool(current)


def field_value(value: Any, path: str) -> Any:
    """Return a path value, keeping repeated identities as a list."""
    values, valid = field_values(value, path)
    if not valid:
        return None
    return values if "[]" in path else values[0]


def _field_is_present(artifact: dict[str, Any], path: str) -> bool:
    values, valid = field_values(artifact, path)
    return valid and all(not _is_blank(item) for item in values)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return not value
    return False


def describe_requirement(
    state: dict[str, Any], requirement: ArtifactRequirement
) -> tuple[str, list[str], dict[str, Any] | None]:
    value = state.get(requirement.source_key)
    fields = list(
        dict.fromkeys([*requirement.required_fields, *requirement.identity_fields])
    )
    candidates = parse_artifact_candidates(value)
    complete = [
        item for item in candidates if not _missing_fields_in_artifact(item, fields)
    ]
    artifact = complete[-1] if complete else parse_artifact(value)
    missing = (
        _missing_fields_in_artifact(artifact, fields)
        if artifact is not None
        else (fields or ["structured artifact"])
    )
    identity = (
        {
            key: field_value(artifact, key)
            for key in requirement.identity_fields
            if artifact is not None and _field_is_present(artifact, key)
        }
        if artifact is not None
        else None
    )
    return str(value) if value is not None else "", missing, identity


def validate_output_contract(value: Any, contract: ArtifactContract) -> list[str]:
    """Describe absent fields without modifying or discarding the artifact."""
    fields = list(dict.fromkeys([*contract.required_fields, *contract.identity_fields]))
    artifact = parse_artifact(value, contract)
    if artifact is None:
        return fields or ["structured artifact"]
    missing = _missing_fields_in_artifact(artifact, fields)
    return missing


def append_execution_issue(state: dict[str, Any], issue: dict[str, Any]) -> None:
    metadata = state.get(EXECUTION_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
        state[EXECUTION_METADATA_KEY] = metadata
    issues = metadata.setdefault("handoff_issues", [])
    if not isinstance(issues, list):
        return
    identity = _issue_identity(issue)
    for existing in issues:
        if isinstance(existing, dict) and _issue_identity(existing) == identity:
            existing.update(issue)
            existing["resolved"] = False
            return
    issues.append({**issue, "resolved": False})


def resolve_execution_issue(state: dict[str, Any], issue: dict[str, Any]) -> None:
    """Mark the matching historical handoff issue resolved after revalidation."""
    metadata = state.get(EXECUTION_METADATA_KEY)
    issues = metadata.get("handoff_issues") if isinstance(metadata, dict) else None
    if not isinstance(issues, list):
        return
    identity = _issue_identity(issue)
    for existing in issues:
        if isinstance(existing, dict) and _issue_identity(existing) == identity:
            existing["resolved"] = True


def _issue_identity(issue: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        issue.get(key) for key in ("kind", "agent", "source_key", "output_key")
    )


def unresolved_execution_issues(state: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = state.get(EXECUTION_METADATA_KEY)
    issues = metadata.get("handoff_issues") if isinstance(metadata, dict) else None
    if not isinstance(issues, list):
        return []
    return [
        issue
        for issue in issues
        if isinstance(issue, dict) and issue.get("resolved") is not True
    ]
