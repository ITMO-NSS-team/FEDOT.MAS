from __future__ import annotations

import copy


def patch_schema_openai_strict(schema: dict) -> dict:
    """Recursively add ``additionalProperties: false`` and full ``required``
    to every object in a JSON Schema, making it compatible with OpenAI strict mode.

    Returns a patched copy; the original is not mutated.
    """
    schema = copy.deepcopy(schema)

    _patch_object(schema)

    for definition in schema.get("$defs", {}).values():
        _patch_object(definition)
        definition.pop("title", None)

    return schema


def _patch_object(obj: dict) -> None:
    """Patch a single schema object node in place."""
    if obj.get("type") != "object":
        return
    props = obj.get("properties")
    if not props:
        return
    obj["additionalProperties"] = False
    obj["required"] = list(props.keys())


def needs_strict_schema(model_name: str) -> bool:
    """Return ``True`` for models that require strict-compatible schemas.

    Uses a whitelist approach: patch all models except Gemini,
    because strict-compatible schemas are harmless for providers
    that don't enforce them.
    """
    gemini_prefixes = ("gemini/", "gemini-", "vertex_ai/")
    return not any(model_name.startswith(p) for p in gemini_prefixes)
