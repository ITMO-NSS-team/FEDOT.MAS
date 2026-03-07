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


def inject_model_enum(schema: dict, allowed_models: list[str]) -> dict:
    """Add ``enum`` constraint to the ``model`` field in agent definitions.

    Looks for ``MAWAgentConfig`` and ``AgentPoolEntry`` in ``$defs`` and injects
    ``enum: allowed_models`` into their ``model`` property, preserving nullable
    (``anyOf``) wrappers.

    Returns a patched copy; the original is not mutated.
    """
    schema = copy.deepcopy(schema)
    target_defs = ("MAWAgentConfig", "AgentPoolEntry", "MASAgentConfig")

    for def_name, definition in schema.get("$defs", {}).items():
        if def_name not in target_defs:
            continue
        props = definition.get("properties", {})
        model_prop = props.get("model")
        if model_prop is None:
            continue
        _inject_enum_into_prop(model_prop, allowed_models)

    return schema


def _inject_enum_into_prop(prop: dict, allowed_models: list[str]) -> None:
    """Inject ``enum`` into a property schema, handling ``anyOf`` (nullable)."""
    # Direct string type: {"type": "string"} or {"type": "string", ...}
    if prop.get("type") == "string":
        prop["enum"] = allowed_models
        return

    # Nullable via anyOf: {"anyOf": [{"type": "string"}, {"type": "null"}]}
    any_of = prop.get("anyOf")
    if any_of:
        for variant in any_of:
            if isinstance(variant, dict) and variant.get("type") == "string":
                variant["enum"] = allowed_models
                return


def needs_strict_schema(model_name: str) -> bool:
    """Return ``True`` for models that require strict-compatible schemas.

    Uses a whitelist approach: patch all models except Gemini,
    because strict-compatible schemas are harmless for providers
    that don't enforce them.
    """
    gemini_prefixes = ("gemini/", "gemini-", "vertex_ai/")
    return not any(model_name.startswith(p) for p in gemini_prefixes)
