"""Deterministic smoke test for the rubber-recipe master-orchestrator topology."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

from fedotmas import MASConfig
from fedotmas.core.runner import run_pipeline
from fedotmas.mas.builder import build_routing_system
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

HERE = Path(__file__).resolve().parent


class ScriptedLlm(BaseLlm):
    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        yield LlmResponse(content=self.responses.pop(0))


def call(name: str, args: dict[str, object]) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part.from_function_call(name=name, args=args)],
    )


def text(value: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=value)])


async def test_master_calls_specialists_and_retains_control() -> None:
    config = MASConfig.model_validate_json(
        (HERE / "config.json").read_text(encoding="utf-8")
    )
    llms = {
        "rubber_property_master": ScriptedLlm(
            model="test/master",
            responses=[
                call(
                    "predict_rubber_properties",
                    {
                        "nr_smr20_phr": 55,
                        "sbr1502_phr": 45,
                        "carbon_black_n220_phr": 55,
                        "zinc_oxide_phr": 5,
                        "stearic_acid_phr": 2,
                        "tmq_antioxidant_phr": 1.5,
                        "antiozonant_6ppd_phr": 1.5,
                        "process_oil_phr": 5,
                        "sulfur_phr": 5,
                        "tmtd_accelerator_phr": 2,
                        "reclaim_phr": 5,
                    },
                ),
                call(
                    "formulation_validator",
                    {"request": "Format the numerical candidate."},
                ),
                call(
                    "rubber_chemistry_reviewer",
                    {"request": "Review the candidate chemistry."},
                ),
                call(
                    "prediction_auditor",
                    {"request": "Audit predictions and uncertainty."},
                ),
                text("Recipe and predicted properties."),
            ],
        ),
        "formulation_validator": ScriptedLlm(
            model="test/formulation", responses=[text("Recipe formatted.")]
        ),
        "rubber_chemistry_reviewer": ScriptedLlm(
            model="test/chemistry", responses=[text("Chemistry reviewed.")]
        ),
        "prediction_auditor": ScriptedLlm(
            model="test/audit", responses=[text("Predictions audited.")]
        ),
    }

    def resolve(model_name: str | None, _worker_models: object) -> BaseLlm:
        assert model_name == "host/gpt-5.6-terra"
        caller = resolve.current_agent
        return llms[caller]

    # The builder resolves models while traversing workers, then coordinator.
    order = iter(
        [
            "formulation_validator",
            "rubber_chemistry_reviewer",
            "prediction_auditor",
            "rubber_property_master",
        ]
    )

    def ordered_resolve(model_name: str | None, worker_models: object) -> BaseLlm:
        resolve.current_agent = next(order)
        return resolve(model_name, worker_models)

    resolve.current_agent = ""
    with patch("fedotmas.mas.builder._resolve_llm", side_effect=ordered_resolve):
        root = build_routing_system(config, autonomous=False)

    result = await run_pipeline(root, "Predict properties of the supplied recipe.")

    assert root.name == "rubber_property_master"
    assert [agent.name for agent in root.sub_agents] == [
        "formulation_validator",
        "rubber_chemistry_reviewer",
        "prediction_auditor",
    ]
    assert config.coordinator.tools == ["rubber-recipe-predictor"]
    master_requests = llms["rubber_property_master"].requests
    assert len(master_requests) == 5
    tool_response = master_requests[1].contents[-1].parts[0].function_response
    assert tool_response is not None
    prediction = tool_response.response["structuredContent"]
    assert prediction["recipe_input_unchanged"]["nr_smr20_phr"] == 55.0
    assert prediction["recipe_input_unchanged"]["carbon_black_n220_phr"] == 55.0
    assert result.state["formulation_output"] == "Recipe formatted."
    assert result.state["chemistry_review"] == "Chemistry reviewed."
    assert result.state["prediction_audit"] == "Predictions audited."
    assert result.state["property_prediction"] == "Recipe and predicted properties."


def test_runner_loads_complete_recipe_and_overrides_all_models():
    import runpy

    runner = runpy.run_path(str(HERE / "run_mas.py"), run_name="test_runner")
    config, query = runner["load_inputs"]("openai/test")
    assert config.coordinator.model == "openai/test"
    assert all(worker.model == "openai/test" for worker in config.workers)
    assert "NR SMR-20 — 55 phr" in query
    assert "регенерат — 5 phr" in query
    assert config.coordinator.output_key == "property_prediction"
