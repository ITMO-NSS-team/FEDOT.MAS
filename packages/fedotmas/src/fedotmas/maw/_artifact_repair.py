"""Lossless local operations for bounded, tools-free artifact repair."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from fedotmas.maw.handoffs import parse_artifact
from fedotmas.maw.models import ArtifactContract


def _json_text(value: str) -> str:
    text = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def close_outer_object(value: Any) -> dict[str, Any] | None:
    """Append only a missing root brace; never complete strings or nested values."""
    if not isinstance(value, str):
        return None
    text = _json_text(value)
    if not text.startswith("{"):
        return None
    stack: list[str] = []
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != {"}": "{", "]": "["}[char]:
                return None
    if quoted or stack != ["{"]:
        return None
    try:
        result = json.loads(text + "}")
    except ValueError:
        return None
    return result if isinstance(result, dict) else None


def repair_source(value: Any, contract: ArtifactContract | None) -> dict | None:
    """Do not promote a nested object out of a malformed JSON document."""
    if isinstance(value, str):
        text = _json_text(value)
        if text.startswith(("{", "[")):
            try:
                result = json.loads(text)
            except ValueError:
                return None
            return (
                result if isinstance(result, dict) else parse_artifact(value, contract)
            )
    return parse_artifact(value, contract)


def _pointer(path: Any) -> list[str]:
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise ValueError("Repair paths must be JSON pointers")
    if re.search(r"~(?![01])", path):
        raise ValueError("Invalid JSON pointer escape")
    return (
        [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
        if path
        else []
    )


def apply_field_copies(source: dict, plan: dict, contract: ArtifactContract) -> dict:
    """Copy only existing values from an immutable source; keep all original data."""
    if set(plan) != {"field_copies"}:
        raise ValueError("Repair plan must contain only field_copies")
    copies = plan["field_copies"]
    if not isinstance(copies, list) or len(copies) > 24:
        raise ValueError("Repair plan must contain at most 24 field copies")
    fields = [*contract.required_fields, *contract.identity_fields]
    result = deepcopy(source)
    targets: set[tuple[str, ...]] = set()
    for operation in copies:
        if not isinstance(operation, dict) or set(operation) != {"from", "to"}:
            raise ValueError("Each repair copy requires only from/to pointers")
        source_path, target_path = (
            _pointer(operation["from"]),
            _pointer(operation["to"]),
        )
        target_name = ".".join(target_path)
        if not target_path or not any(
            field == target_name
            or field.startswith(target_name + ".")
            or field.startswith(target_name + "[]")
            for field in fields
        ):
            raise ValueError("Repair destination must supply a declared contract field")
        if tuple(target_path) in targets:
            raise ValueError("Duplicate repair destination")
        targets.add(tuple(target_path))
        value: Any = source
        for part in source_path:
            if isinstance(value, dict):
                value = value[part]
            elif (
                isinstance(value, list) and part.isdecimal() and str(int(part)) == part
            ):
                value = value[int(part)]
            else:
                raise ValueError("Repair source path does not exist")
        target = result
        for part in target_path[:-1]:
            if part not in target:
                target[part] = {}
            if not isinstance(target[part], dict):
                raise ValueError("Repair cannot replace an existing container")
            target = target[part]
        key = target_path[-1]
        if key in target and target[key] != value:
            raise ValueError("Repair cannot overwrite an existing value")
        target[key] = deepcopy(value)
    return result
