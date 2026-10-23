from functools import partial
from unittest.mock import AsyncMock

import pytest
from fedotmas._settings import ModelConfig
from fedotmas.maw.contracts import handoff_contract_errors
from fedotmas.maw.handoffs import describe_requirement, validate_output_contract
from fedotmas.maw.models import MAWConfig
from fedotmas.meta import _adk_runner
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.meta._helpers import validate_pipeline_output


def pipeline(produced=None, required=None):
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "source",
                    "instruction": "Extract locations",
                    "output_key": "locations",
                    "tools": ["document"],
                    "output_contract": {
                        "required_fields": produced
                        if produced is not None
                        else ["locations"]
                    },
                },
                {
                    "name": "consumer",
                    "instruction": "Map places",
                    "output_key": "answer",
                    "input_requirements": [
                        {
                            "source_key": "locations",
                            "required_fields": required
                            if required is not None
                            else ["place_name"],
                        }
                    ],
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [{"agent_name": "source"}, {"agent_name": "consumer"}],
            },
            "final_answer_agent": "consumer",
        }
    )


def test_reproduces_gaia_root_vs_list_contract_mismatch():
    errors = handoff_contract_errors(pipeline())
    assert len(errors) == 1
    assert (
        "place_name" in errors[0] and "source" in errors[0] and "consumer" in errors[0]
    )


@pytest.mark.parametrize(
    ("produced", "required"),
    [
        (["locations[].place_name"], ["locations[].place_name"]),
        (["locations[].place_name", "locations[].year"], ["locations"]),
        (["location.name"], ["location"]),
        (["place_name"], ["place_name"]),
    ],
)
def test_matching_paths_and_guaranteed_containers(produced, required):
    assert not handoff_contract_errors(pipeline(produced, required))


def test_declaring_container_does_not_guarantee_its_children():
    assert handoff_contract_errors(pipeline(["locations"], ["locations[].place_name"]))


@pytest.mark.parametrize(
    "produced, required, artifact",
    [
        ("urls[]", "urls", {"urls": ["https://example.com"]}),
        ("items[].urls[]", "items[].urls", {"items": [{"urls": ["u"]}]}),
        ("items[].urls[]", "items", {"items": [{"urls": ["u"]}]}),
    ],
)
def test_repeated_leaf_guarantees_container(produced, required, artifact):
    config = pipeline([produced], [required])
    assert validate_output_contract(artifact, config.agents[0].output_contract) == []
    assert (
        describe_requirement(
            {"locations": artifact}, config.agents[1].input_requirements[0]
        )[1]
        == []
    )
    validate_pipeline_output(config.model_dump(), available_tools={"document"})


@pytest.mark.parametrize(
    "produced, required",
    [("urls", "urls[]"), ("urls[].value", "urls.value"), ("urls_extra[]", "urls")],
)
def test_array_path_coverage_does_not_weaken_or_confuse_shapes(produced, required):
    assert handoff_contract_errors(pipeline([produced], [required]))


def test_missing_producer_contract_and_text_user_query():
    config = pipeline()
    config.agents[0].output_contract = None
    assert "needs an output_contract" in handoff_contract_errors(config)[0]
    config.agents[1].input_requirements[0].source_key = "user_query"
    assert "plain task text" in handoff_contract_errors(config)[0]


def test_unknown_tools_rejected_before_execution():
    config = pipeline(["place_name"], ["place_name"])
    with pytest.raises(ValueError, match="unavailable MCP servers"):
        validate_pipeline_output(config.model_dump(), available_tools={"sandbox-light"})


@pytest.mark.asyncio
async def test_contract_feedback_retries_generation_and_keeps_billed_usage(monkeypatch):
    messages = []
    outputs = [
        pipeline(),
        pipeline(["locations[].place_name"], ["locations[].place_name"]),
    ]

    async def execute(**kwargs):
        messages.append(kwargs["user_message"])
        kwargs["usage_totals"].update(prompt=17, completion=4, elapsed=2.0)
        return LLMCallResult(
            raw_output=outputs[len(messages) - 1].model_dump(),
            prompt_tokens=17,
            completion_tokens=4,
            elapsed=2.0,
        )

    monkeypatch.setattr(_adk_runner, "_execute_meta_call", execute)
    monkeypatch.setattr(_adk_runner.asyncio, "sleep", AsyncMock())
    result = await run_meta_agent_call(
        agent_name="meta_agent",
        instruction="Design a pipeline",
        user_message="Task",
        output_schema=MAWConfig,
        output_key="config",
        model=ModelConfig(model="stub"),
        temperature=0,
        max_retries=1,
        output_validator=partial(
            validate_pipeline_output, available_tools={"document"}
        ),
    )
    assert len(messages) == 2 and "place_name" in messages[1]
    assert "Invalid generated pipeline" in messages[1]
    assert result.prompt_tokens == 34 and result.completion_tokens == 8
    assert result.elapsed == 4.0
