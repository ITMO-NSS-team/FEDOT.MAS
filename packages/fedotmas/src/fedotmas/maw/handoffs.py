"""Helpers for validating optional, generated MAW artifact contracts."""

from __future__ import annotations

import json
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


def parse_artifact(value: Any) -> dict[str, Any] | None:
    """Return a structured artifact mapping; scalar answer strings are rejected."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def missing_contract_fields(
    value: Any, required_fields: list[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    artifact = parse_artifact(value)
    if artifact is None:
        return None, list(required_fields) or ["structured artifact"]
    missing = [
        field
        for field in required_fields
        if not _field_is_present(artifact, field)
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
    artifact, missing = missing_contract_fields(value, fields)
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
    _artifact, missing = missing_contract_fields(value, fields)
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
    return tuple(issue.get(key) for key in ("kind", "agent", "source_key", "output_key"))


def unresolved_execution_issues(state: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = state.get(EXECUTION_METADATA_KEY)
    issues = metadata.get("handoff_issues") if isinstance(metadata, dict) else None
    if not isinstance(issues, list):
        return []
    return [issue for issue in issues if isinstance(issue, dict) and issue.get("resolved") is not True]
