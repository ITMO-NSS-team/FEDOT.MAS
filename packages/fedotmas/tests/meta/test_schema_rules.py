"""Schema patching rules — pure functions, no mocks."""

from __future__ import annotations

import copy

import pytest

from fedotmas.meta.schema_utils import (
    inject_model_enum,
    needs_strict_schema,
    patch_schema_openai_strict,
)


def _make_base_schema(*, nullable_model: bool = False) -> dict:
    """Build a minimal JSON Schema with AgentConfig and AgentPoolEntry defs."""
    if nullable_model:
        model_prop = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    else:
        model_prop = {"type": "string"}

    return {
        "$defs": {
            "AgentConfig": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "model": copy.deepcopy(model_prop),
                    "instruction": {"type": "string"},
                    "output_key": {"type": "string"},
                },
            },
            "AgentPoolEntry": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "model": copy.deepcopy(model_prop),
                    "instruction": {"type": "string"},
                },
            },
        },
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {"$ref": "#/$defs/AgentConfig"},
            },
        },
    }


class TestInjectModelEnum:
    """Rules 1-4: inject_model_enum behaviour."""

    def test_enum_injected_into_agent_config(self, allowed_models):
        schema = _make_base_schema()
        result = inject_model_enum(schema, allowed_models)
        model_prop = result["$defs"]["AgentConfig"]["properties"]["model"]
        assert model_prop["enum"] == allowed_models

    def test_enum_injected_into_agent_pool_entry(self, allowed_models):
        schema = _make_base_schema()
        result = inject_model_enum(schema, allowed_models)
        model_prop = result["$defs"]["AgentPoolEntry"]["properties"]["model"]
        assert model_prop["enum"] == allowed_models

    def test_immutability(self, allowed_models):
        schema = _make_base_schema()
        original = copy.deepcopy(schema)
        inject_model_enum(schema, allowed_models)
        assert schema == original

    def test_nullable_model_enum_on_string_variant(self, allowed_models):
        schema = _make_base_schema(nullable_model=True)
        result = inject_model_enum(schema, allowed_models)
        any_of = result["$defs"]["AgentConfig"]["properties"]["model"]["anyOf"]
        string_variant = next(v for v in any_of if v.get("type") == "string")
        null_variant = next(v for v in any_of if v.get("type") == "null")
        assert string_variant["enum"] == allowed_models
        assert "enum" not in null_variant


class TestPatchSchemaStrict:
    """Rules 5-6: patch_schema_openai_strict behaviour."""

    def test_adds_additional_properties_and_required(self):
        schema = _make_base_schema()
        result = patch_schema_openai_strict(schema)
        agent_def = result["$defs"]["AgentConfig"]
        assert agent_def["additionalProperties"] is False
        assert set(agent_def["required"]) == set(agent_def["properties"].keys())

    def test_strict_immutability(self):
        schema = _make_base_schema()
        original = copy.deepcopy(schema)
        patch_schema_openai_strict(schema)
        assert schema == original


class TestNeedsStrictSchema:
    """Rule 7: Gemini whitelist."""

    @pytest.mark.parametrize("model", [
        "gemini/gemini-2.0-flash",
        "gemini-2.0-flash",
        "vertex_ai/gemini-2.0-flash",
    ])
    def test_gemini_no_strict(self, model):
        assert needs_strict_schema(model) is False

    @pytest.mark.parametrize("model", [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4-20250514",
        "deepseek/deepseek-chat",
    ])
    def test_non_gemini_needs_strict(self, model):
        assert needs_strict_schema(model) is True


class TestEnumPlusStrictCombo:
    """Rule 8: enum survives strict patching."""

    def test_enum_survives_strict(self, allowed_models):
        schema = _make_base_schema()
        schema = inject_model_enum(schema, allowed_models)
        result = patch_schema_openai_strict(schema)
        model_prop = result["$defs"]["AgentConfig"]["properties"]["model"]
        assert model_prop["enum"] == allowed_models
        assert result["$defs"]["AgentConfig"]["additionalProperties"] is False
