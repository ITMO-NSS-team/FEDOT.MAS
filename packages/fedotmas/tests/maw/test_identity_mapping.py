import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ValidationError

from fedotmas.control.fixes._fix_agent_configs import repair_agent_configs
from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw.handoffs import merge_input_requirements, unresolved_execution_issues
from fedotmas.maw.models import ArtifactRequirement, IdentityFieldMapping, MAWConfig


def config():
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "source",
                    "instruction": "Extract only identified people",
                    "output_key": "people",
                    "output_contract": {
                        "required_fields": ["people"],
                        "identity_fields": ["people[].id"],
                    },
                },
                {
                    "name": "inspector",
                    "instruction": "Return web_counts for these people: {people}",
                    "output_key": "counts",
                    "input_requirements": [
                        {
                            "source_key": "people",
                            "required_fields": ["people"],
                            "identity_fields": ["people[].id"],
                            "identity_field_mapping": [
                                {
                                    "source_path": "people[].id",
                                    "output_path": "web_counts[].id",
                                }
                            ],
                        }
                    ],
                    "output_contract": {
                        "required_fields": ["web_counts"],
                        "identity_fields": ["web_counts[].id"],
                    },
                },
                {
                    "name": "final",
                    "instruction": "Answer from {counts}",
                    "output_key": "answer",
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"agent_name": n} for n in ("source", "inspector", "final")
                ],
            },
        }
    )


def test_output_normalization_does_not_add_the_old_container():
    cfg = config()
    assert cfg.agents[1].output_contract.identity_fields == ["web_counts[].id"]
    assert MAWConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_ambiguous_collection_rename_is_rejected_before_running_workers():
    raw = config().model_dump()
    raw["agents"][1]["input_requirements"][0]["identity_field_mapping"] = []
    with pytest.raises(ValueError, match="Declare identity_field_mapping"):
        MAWConfig.model_validate(raw)


@pytest.mark.parametrize(
    "mapping",
    [
        {"absent": "web_counts[].id"},
        {"people[].id": "id"},
        {"people[].id": "web_counts..id"},
        {"people[].id": ""},
        {"people[].id": "web_counts[].id", "people[].name": "web_counts[].id"},
    ],
)
def test_invalid_identity_routes_are_rejected(mapping):
    with pytest.raises(ValidationError):
        ArtifactRequirement(
            source_key="people",
            identity_fields=["people[].id", "people[].name"],
            identity_field_mapping=[
                IdentityFieldMapping(source_path=k, output_path=v)
                for k, v in mapping.items()
            ],
        )


def test_split_requirements_keep_mapping_and_reject_conflicting_routes():
    requirement = config().agents[1].input_requirements[0]
    merged = merge_input_requirements([requirement, requirement.model_copy(deep=True)])
    assert len(merged) == 1
    assert merged[0].identity_field_mapping == [
        IdentityFieldMapping(source_path="people[].id", output_path="web_counts[].id")
    ]
    with pytest.raises(ValueError, match="Conflicting identity"):
        merge_input_requirements(
            [
                requirement,
                ArtifactRequirement(
                    source_key="people", identity_fields=["people[].id"]
                ),
            ]
        )


def test_duplicate_routes_and_merging_implicit_targets_are_rejected():
    mapping = IdentityFieldMapping(source_path="source.id", output_path="target.id")
    with pytest.raises(ValueError, match="Duplicate"):
        ArtifactRequirement(
            source_key="source",
            identity_fields=["source.id"],
            identity_field_mapping=[mapping, mapping],
        )
    with pytest.raises(ValueError, match="merge distinct"):
        ArtifactRequirement(
            source_key="source",
            identity_fields=["source.id", "target.id"],
            identity_field_mapping=[mapping],
        )


@pytest.mark.asyncio
async def test_meta_retries_ambiguous_collection_and_accepts_explicit_routes(
    monkeypatch,
):
    from functools import partial
    from unittest.mock import AsyncMock
    from fedotmas._settings import ModelConfig
    from fedotmas.meta import _adk_runner
    from fedotmas.meta._adk_runner import LLMCallResult
    from fedotmas.meta._helpers import validate_pipeline_output

    good = config().model_dump()
    bad = config().model_dump()
    bad["agents"][1]["input_requirements"][0]["identity_field_mapping"] = []
    outputs = iter([bad, good])
    messages = []

    async def execute(**kwargs):
        messages.append(kwargs["user_message"])
        kwargs["usage_totals"].update(prompt=12, completion=3, elapsed=1.0)
        return LLMCallResult(
            raw_output=next(outputs), prompt_tokens=12, completion_tokens=3, elapsed=1.0
        )

    monkeypatch.setattr(_adk_runner, "_execute_meta_call", execute)
    monkeypatch.setattr(_adk_runner.asyncio, "sleep", AsyncMock())
    result = await _adk_runner.run_meta_agent_call(
        agent_name="meta",
        instruction="Design",
        user_message="Count works",
        output_schema=MAWConfig,
        output_key="config",
        model=ModelConfig(model="stub"),
        temperature=0,
        max_retries=1,
        output_validator=partial(validate_pipeline_output, available_tools=set()),
    )
    assert len(messages) == 2
    assert "identity_field_mapping" in messages[1]
    assert result.prompt_tokens == 24 and result.completion_tokens == 6
    assert MAWConfig.model_validate(result.raw_output).agents[
        1
    ].output_contract.identity_fields == ["web_counts[].id"]


class Stub(BaseLlm):
    model: str = "offline"
    answer: str

    async def generate_content_async(self, request, stream=False):
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.answer)])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ids, succeeds",
    [
        (["A", "B"], True),
        (["A", "C"], False),
        (["A"], False),
        (["B", "A"], False),
        ([None, "B"], False),
    ],
)
async def test_runner_checks_every_identity_across_renamed_collections(
    monkeypatch, ids, succeeds
):
    answers = iter(
        [
            '{"people":[{"id":"A"},{"id":"B"}]}',
            json.dumps(
                {
                    "web_counts": [
                        {"id": identity, "count": n + 3}
                        for n, identity in enumerate(ids)
                    ]
                }
            ),
            "7",
        ]
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: Stub(answer=next(answers)))
    tree = builder.build(config(), autonomous=False)
    result = await run_pipeline(tree, "Count works for the identified people")
    issues = unresolved_execution_issues(result.state)
    assert (not issues) == succeeds
    if not succeeds:
        assert any(
            item["kind"] in {"entity_continuity_mismatch", "incomplete_artifact"}
            for item in issues
        )
    assert "people" not in json.loads(result.state["counts"])


@pytest.mark.asyncio
async def test_scalar_identity_mapping_is_copied_from_the_right_source(monkeypatch):
    cfg = (
        config()
        .agents[1]
        .model_copy(
            update={
                "input_requirements": [
                    ArtifactRequirement(
                        source_key="paper",
                        identity_fields=["doi"],
                        identity_field_mapping=[
                            IdentityFieldMapping(
                                source_path="doi", output_path="source.doi"
                            )
                        ],
                    )
                ],
                "output_contract": config()
                .agents[1]
                .output_contract.model_copy(
                    update={
                        "required_fields": ["answer"],
                        "identity_fields": ["source.doi"],
                    }
                ),
            }
        )
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: Stub(answer="unused"))
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    ctx = MagicMock()
    ctx.state = {"paper": {"doi": "10/example"}, "counts": '{"answer":42}'}
    await agent.after_agent_callback(ctx)
    assert json.loads(ctx.state["counts"]) == {
        "answer": 42,
        "source": {"doi": "10/example"},
    }
    assert not unresolved_execution_issues(ctx.state)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["drop", "reroute"])
async def test_recovery_cannot_disable_or_redirect_identity_routes(change):
    cfg = config()
    ctx = SimpleNamespace(state={"config": cfg.model_dump_json()})
    requirement = cfg.agents[1].input_requirements[0].model_dump()
    requirement["identity_field_mapping"] = (
        []
        if change == "drop"
        else [{"source_path": "people[].id", "output_path": "unchecked[].id"}]
    )
    before = ctx.state["config"]
    result = await repair_agent_configs(
        ctx,
        [{"agent_name": "inspector", "input_requirements": [requirement]}],
        "repair",
    )
    assert result.startswith("Error")
    assert ctx.state["config"] == before


@pytest.mark.asyncio
async def test_recovery_can_relocate_output_and_identity_route_together():
    cfg = config()
    ctx = SimpleNamespace(state={"config": cfg.model_dump_json()})
    requirement = cfg.agents[1].input_requirements[0].model_dump()
    requirement["identity_field_mapping"] = [
        {"source_path": "people[].id", "output_path": "counts[].id"}
    ]
    result = await repair_agent_configs(
        ctx,
        [
            {
                "agent_name": "inspector",
                "instruction": "Return counts with IDs",
                "input_requirements": [requirement],
                "output_contract": {
                    "required_fields": ["counts"],
                    "identity_fields": ["counts[].id"],
                },
                "output_field_renames": [
                    {"old_path": "web_counts", "new_path": "counts"},
                    {"old_path": "web_counts[].id", "new_path": "counts[].id"},
                ],
            }
        ],
        "Relocate without losing identities",
    )
    assert result.startswith("Updated"), result
    updated = MAWConfig.model_validate_json(ctx.state["config"])
    assert updated.agents[1].output_contract.identity_fields == ["counts[].id"]
