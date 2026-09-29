"""Offline regressions for the GUI's final-answer contract."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from fedotmas import MAWConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module("server.app")
sanitize_config = importlib.import_module("server.normalize").sanitize_config
result_payload = importlib.import_module("server.results").result_payload
RunIn = importlib.import_module("server.schemas").RunIn


class ScriptedLlm(BaseLlm):
    answers: list[str]

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=self.answers.pop(0))]
            )
        )


async def run_gui(monkeypatch, kind, config, answers):
    builder = importlib.import_module(f"fedotmas.{kind}.builder")
    llm = ScriptedLlm(model="openai/test", answers=answers)
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: llm)
    response = await app.run(
        RunIn(
            kind=kind, query="Answer.", tools=[], model="openai/gpt-4o", config=config
        )
    )
    events = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        events.extend(
            json.loads(line[6:])
            for line in chunk.splitlines()
            if line.startswith("data: ")
        )
    assert not [e for e in events if e["type"] == "error"], events
    return next(e for e in events if e["type"] == "done")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,status",
    [("42", "completed"), ('{"status":"unresolved","answer":null}', "incomplete")],
)
async def test_maw_endpoint_preserves_answer_and_execution_status(
    monkeypatch, answer, status
):
    done = await run_gui(
        monkeypatch,
        "maw",
        {
            "agents": [
                {"name": "critic", "instruction": "Answer.", "output_key": "answer"}
            ],
            "pipeline": {"type": "agent", "agent_name": "critic"},
        },
        [answer],
    )
    assert done["answer"] == answer
    assert done["output_key"] == "answer"
    assert done["status"] == status
    assert "_fedotmas_execution" in done["state"]


@pytest.mark.asyncio
async def test_mas_endpoint_records_coordinator_answer(monkeypatch):
    agent = {"instruction": "Answer.", "description": "Answer."}
    done = await run_gui(
        monkeypatch,
        "mas",
        {
            "coordinator": {**agent, "name": "coordinator"},
            "workers": [
                {**agent, "name": "worker", "output_key": "coordinator_result"}
            ],
        },
        ["42"],
    )
    assert done["answer"] == "42"
    assert done["output_key"] == "coordinator_result_"
    assert done["status"] == "completed"


@pytest.mark.parametrize(
    "value,answer",
    [(0, "0"), (False, "false"), ({"answer": 42}, '{"answer": 42}'), (None, "")],
)
def test_structured_answer_and_missing_output(value, answer):
    payload = result_payload(
        {"answer": value, "metadata": "long" * 100}, "answer", "completed"
    )
    assert payload["answer"] == answer
    assert payload["status"] == ("completed" if answer else "incomplete")
    assert result_payload({"other": "42"}, None, "limited")["status"] == "limited"
    assert result_payload({"other": "42"}, None, "completed")["answer"] == ""


@pytest.mark.asyncio
async def test_normalized_handoff_survives_reload_and_execution(monkeypatch):
    config = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "data-producer",
                    "instruction": "Produce.",
                    "output_key": "raw-data",
                },
                {
                    "name": "final-critic",
                    "instruction": "Use {raw-data?}.",
                    "output_key": "answer",
                    "input_requirements": [
                        {"source_key": "raw-data", "required_fields": ["value"]}
                    ],
                },
            ],
            "final_answer_agent": "final-critic",
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "data-producer"},
                    {"type": "agent", "agent_name": "final-critic"},
                ],
            },
        }
    )
    normalized = sanitize_config(config, "maw", available_tools={})
    reloaded = MAWConfig.model_validate_json(normalized.model_dump_json())
    assert reloaded.agents[1].input_requirements[0].source_key == "raw_data"
    assert reloaded.agents[1].instruction == "Use {raw_data?}."
    assert reloaded.final_answer_agent == "final_critic"
    done = await run_gui(
        monkeypatch, "maw", reloaded.model_dump(), ['{"value":42}', "42"]
    )
    assert done["answer"] == "42"
    assert done["status"] == "completed"


def test_normalization_rewrites_complete_placeholders_only():
    config = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "producer",
                    "instruction": "Produce.",
                    "output_key": "raw-data",
                },
                {
                    "name": "consumer",
                    "instruction": "Use {raw-data?}; keep {raw-dataset?}.",
                    "output_key": "answer",
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "producer"},
                    {"type": "agent", "agent_name": "consumer"},
                ],
            },
        }
    )
    normalized = sanitize_config(config, "maw", available_tools={})
    assert normalized.agents[1].instruction == "Use {raw_data?}; keep {raw-dataset?}."


def test_normalization_rejects_colliding_output_keys():
    config = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "producer",
                    "instruction": "Produce.",
                    "output_key": "raw-data",
                },
                {
                    "name": "consumer",
                    "instruction": "Consume.",
                    "output_key": "raw_data",
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "producer"},
                    {"type": "agent", "agent_name": "consumer"},
                ],
            },
        }
    )
    with pytest.raises(ValueError, match="Duplicate output_key"):
        sanitize_config(config, "maw", available_tools={})
