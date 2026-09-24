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
        "rubber_recipe_master": ScriptedLlm(
            model="test/master",
            responses=[
                call(
                    "predict_rubber_recipe",
                    {
                        "thermal_conductivity_min_w_mk": 0.45,
                        "oil_swelling_max_pct_1006h": 20.0,
                        "water_swelling_max_pct_1006h": 28.0,
                        "specific_gravity_max": 1.23,
                        "nr_min_phr": 25,
                        "nr_max_phr": 75,
                        "n220_min_phr": 20,
                        "n220_max_phr": 80,
                        "search_step_phr": 1,
                        "center_nr_phr": 50,
                    },
                ),
                call(
                    "formulation_specialist",
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
        "formulation_specialist": ScriptedLlm(
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
            "formulation_specialist",
            "rubber_chemistry_reviewer",
            "prediction_auditor",
            "rubber_recipe_master",
        ]
    )

    def ordered_resolve(model_name: str | None, worker_models: object) -> BaseLlm:
        resolve.current_agent = next(order)
        return resolve(model_name, worker_models)

    resolve.current_agent = ""
    with patch("fedotmas.mas.builder._resolve_llm", side_effect=ordered_resolve):
        root = build_routing_system(config, autonomous=False)

    result = await run_pipeline(root, "Generate a rubber recipe and properties.")

    assert root.name == "rubber_recipe_master"
    assert [agent.name for agent in root.sub_agents] == [
        "formulation_specialist",
        "rubber_chemistry_reviewer",
        "prediction_auditor",
    ]
    assert config.coordinator.tools == ["rubber-recipe-predictor"]
    master_requests = llms["rubber_recipe_master"].requests
    assert len(master_requests) == 5
    tool_response = master_requests[1].contents[-1].parts[0].function_response
    assert tool_response is not None
    prediction = tool_response.response["structuredContent"]
    assert prediction["recipe"]["nr_smr20_phr"] == 58.0
    assert prediction["recipe"]["carbon_black_n220_phr"] == 60.0
    assert result.state["formulation_output"] == "Recipe formatted."
    assert result.state["chemistry_review"] == "Chemistry reviewed."
    assert result.state["prediction_audit"] == "Predictions audited."
    assert result.state["recipe_prediction"] == "Recipe and predicted properties."
