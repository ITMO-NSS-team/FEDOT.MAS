"""Replay captured b05 generation failures through real ADK, without network."""

import copy
from functools import partial
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from fedotmas._settings import ModelConfig
from fedotmas.maw.models import MAWConfig
from fedotmas.meta import _adk_runner
from fedotmas.meta._evidence_repair import validate_evidence_only_repair
from fedotmas.meta._helpers import validate_pipeline_output

CASES = json.loads(
    (Path(__file__).parent / "fixtures/evidence_generation_failures.json").read_text()
)
TOOLS = {
    "media",
    "document",
    "download",
    "web-scraping",
    "websearch-tavily",
    "sandbox-light",
}


class Responses(BaseLlm):
    model: str = "offline"
    outputs: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, request, stream=False):
        raw = self.outputs[len(self.requests)]
        self.requests.append(request.model_copy(deep=True))
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=json.dumps(raw))]
            ),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=11, candidates_token_count=7
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["task_id"][:8])
async def test_real_adk_source_retry_retains_payload_and_usage(monkeypatch, case):
    validator = partial(
        validate_pipeline_output, available_tools=TOOLS, task=case["task"]
    )
    with pytest.raises(ValueError):
        validator(copy.deepcopy(case["original"]))
    model = Responses(outputs=[case["original"], case["repaired"]])
    monkeypatch.setattr(_adk_runner, "make_llm", lambda _: model)
    monkeypatch.setattr(_adk_runner.asyncio, "sleep", AsyncMock())
    monkeypatch.delenv("FEDOTMAS_META_RESPONSE_DIR", raising=False)
    result = await _adk_runner.run_meta_agent_call(
        agent_name="meta",
        instruction="Plan",
        user_message="TASK: " + case["task"],
        evidence_task=case["task"],
        output_schema=MAWConfig,
        output_key="config",
        model=ModelConfig(model="offline"),
        temperature=0,
        max_retries=1,
        output_validator=validator,
    )
    assert len(model.requests) == 2
    message = str(model.requests[1].contents)
    assert "TARGETED SOURCE DECLARATION REPAIR" in message
    assert "CONFIGURATION TO REPAIR" in message
    assert case["original"]["agents"][0]["name"] in message
    validator(result.raw_output)
    validate_evidence_only_repair(case["original"], result.raw_output, case["task"])
    assert (result.prompt_tokens, result.completion_tokens) == (22, 14)


@pytest.mark.parametrize(
    "change",
    [
        "drop_source",
        "drop_fact",
        "change_source",
        "instruction",
        "tools",
        "model",
        "pipeline",
        "input_field",
        "identity",
    ],
)
def test_repair_cannot_weaken_obligations_or_rewrite_plan(change):
    case = CASES[3]
    fixed = copy.deepcopy(case["repaired"])
    reader = fixed["agents"][1]
    if change == "drop_source":
        reader["evidence_requirements"] = []
    elif change == "drop_fact":
        reader["evidence_requirements"][0]["required_fields"].pop()
    elif change == "change_source":
        # Start with an already-valid source, which must not be changed.
        case = {**case, "original": copy.deepcopy(case["repaired"])}
        reader["evidence_requirements"][0]["source"] = "chemistry"
    elif change == "instruction":
        reader["instruction"] = "Guess the surname"
    elif change == "tools":
        reader["tools"] = ["web-scraping"]
    elif change == "model":
        reader["model"] = "openai/gpt-4o-mini"
    elif change == "pipeline":
        fixed["pipeline"]["children"].reverse()
    elif change == "input_field":
        reader["input_requirements"][0]["required_fields"] = []
    else:
        reader["output_contract"]["identity_fields"] = ["source_url"]
    with pytest.raises(ValueError):
        validate_evidence_only_repair(case["original"], fixed, case["task"])


@pytest.mark.asyncio
async def test_failed_targeted_repair_never_uses_further_retries(monkeypatch):
    case = CASES[3]
    weakened = copy.deepcopy(case["repaired"])
    weakened["agents"][1]["evidence_requirements"] = []
    model = Responses(outputs=[case["original"], weakened])
    monkeypatch.setattr(_adk_runner, "make_llm", lambda _: model)
    monkeypatch.setattr(_adk_runner.asyncio, "sleep", AsyncMock())
    with pytest.raises(ValueError) as error:
        await _adk_runner.run_meta_agent_call(
            agent_name="meta",
            instruction="Plan",
            user_message=case["task"],
            evidence_task=case["task"],
            output_schema=MAWConfig,
            output_key="config",
            model=ModelConfig(model="offline"),
            temperature=0,
            max_retries=3,
            output_validator=partial(
                validate_pipeline_output, available_tools=TOOLS, task=case["task"]
            ),
        )
    assert len(model.requests) == 2
    assert error.value.prompt_tokens == 22


@pytest.mark.parametrize(
    "locator", ["user_query", "search_results.selected_url", "other_url"]
)
def test_repair_preserves_valid_reader_locator(locator):
    case = CASES[3]
    original = copy.deepcopy(case["repaired"])
    fixed = copy.deepcopy(original)
    reader = fixed["agents"][1]
    reader["evidence_requirements"][0]["source_field"] = locator
    reader["output_contract"]["required_fields"].append(locator)
    with pytest.raises(ValueError, match="own output field|valid locator"):
        validate_evidence_only_repair(original, fixed, case["task"])


def test_missing_attachment_obligation_can_be_added_without_rewriting_plan():
    task = "Read the supplied file.\nHost file path: /tmp/input.csv"
    original = {
        "agents": [
            {
                "name": "reader",
                "instruction": "Read the supplied file",
                "tools": ["document"],
                "output_key": "answer",
            }
        ],
        "pipeline": {"agent_name": "reader"},
    }
    fixed = copy.deepcopy(original)
    fixed["agents"][0].update(
        evidence_requirements=[
            {
                "task_quote": "Host file path: /tmp/input.csv",
                "source": "/tmp/input.csv",
                "source_kind": "file",
                "source_field": "file_path",
                "required_fields": ["result"],
            }
        ],
        output_contract={"required_fields": ["file_path", "result"]},
    )
    validate_pipeline_output(fixed, available_tools=TOOLS, task=task)
    validate_evidence_only_repair(original, fixed, task)
