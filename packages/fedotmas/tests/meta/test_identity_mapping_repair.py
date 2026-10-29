import copy

import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock

from fedotmas._settings import ModelConfig
from fedotmas.maw.models import MAWConfig
from fedotmas.meta import _adk_runner
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.meta._identity_repair import (
    identity_repair_payload,
    validate_identity_only_repair,
)


def pipeline():
    return {
        "agents": [
            {
                "name": "source",
                "instruction": "Find birthplaces",
                "output_key": "places",
                "output_contract": {
                    "required_fields": ["birthplaces"],
                    "identity_fields": ["birthplaces[].president"],
                },
            },
            {
                "name": "coordinates",
                "instruction": "Enrich each birthplace with coordinates, preserving every president.",
                "output_key": "coords",
                "input_requirements": [
                    {
                        "source_key": "places",
                        "required_fields": ["birthplaces"],
                        "identity_fields": ["birthplaces[].president"],
                    }
                ],
                "output_contract": {
                    "required_fields": ["birthplaces_with_coords"],
                    "identity_fields": ["birthplaces_with_coords[].president"],
                },
            },
            {
                "name": "answerer",
                "instruction": "Answer using {coords}",
                "output_key": "answer",
            },
        ],
        "pipeline": {
            "type": "sequential",
            "children": [
                {"agent_name": n} for n in ["source", "coordinates", "answerer"]
            ],
        },
    }


def repaired(raw):
    result = copy.deepcopy(raw)
    result["agents"][1]["input_requirements"][0]["identity_field_mapping"] = [
        {
            "source_path": "birthplaces[].president",
            "output_path": "birthplaces_with_coords[].president",
        }
    ]
    return result


def test_ambiguous_mapping_still_fails_and_can_be_repaired_without_changing_plan():
    raw = pipeline()
    with pytest.raises(ValidationError) as caught:
        MAWConfig.model_validate(raw)
    original = identity_repair_payload(caught.value)
    assert original == raw
    validate_identity_only_repair(original, repaired(raw))


@pytest.mark.parametrize(
    "change", ["instruction", "required_field", "model", "identity", "pipeline"]
)
def test_targeted_repair_cannot_rewrite_or_weaken_plan(change):
    raw = pipeline()
    fixed = repaired(raw)
    if change == "instruction":
        fixed["agents"][1]["instruction"] = "Guess coordinates"
    elif change == "required_field":
        fixed["agents"][0]["output_contract"]["required_fields"] = []
    elif change == "model":
        fixed["agents"][1]["model"] = "different-model"
    elif change == "identity":
        fixed["agents"][0]["output_contract"]["identity_fields"] = []
    else:
        fixed["pipeline"]["children"] = list(reversed(fixed["pipeline"]["children"]))
    with pytest.raises(ValueError):
        validate_identity_only_repair(raw, fixed)


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [False, True])
async def test_identity_retry_is_targeted_bounded_and_retains_usage(
    monkeypatch, rewrite
):
    messages = []
    raw = pipeline()
    fixed = repaired(raw)
    if rewrite:
        fixed["agents"][1]["instruction"] = "Guess instead"

    async def execute(**kwargs):
        messages.append(kwargs["user_message"])
        kwargs["usage_totals"].update(prompt=10, completion=5, elapsed=1)
        if len(messages) == 1:
            MAWConfig.model_validate(raw)
        return LLMCallResult(fixed, 10, 5, 1)

    monkeypatch.setattr(_adk_runner, "_execute_meta_call", execute)
    monkeypatch.setattr(_adk_runner.asyncio, "sleep", AsyncMock())
    call = run_meta_agent_call(
        agent_name="meta",
        instruction="Plan",
        user_message="Task",
        output_schema=MAWConfig,
        output_key="config",
        model=ModelConfig(model="stub"),
        temperature=0,
        max_retries=2,
    )
    if rewrite:
        with pytest.raises(ValueError) as error:
            await call
        assert error.value.prompt_tokens == 20
    else:
        result = await call
        assert result.prompt_tokens == 20 and result.completion_tokens == 10
    assert len(messages) == 2
    assert "TARGETED IDENTITY MAPPING REPAIR" in messages[1]
    assert "birthplaces[].president" in messages[1]
