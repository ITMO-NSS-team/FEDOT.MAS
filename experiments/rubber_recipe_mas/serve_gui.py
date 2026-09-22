from __future__ import annotations

import argparse
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import uvicorn
from fedotmas import MAS, MASConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field, PrivateAttr

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "terra_subscription_run"
SESSION_DB = RUN_DIR / "gui_sessions.db"
MODEL = "host/gpt-5.6-terra"
PREDICTOR_SERVER = "rubber-recipe-predictor"

OUTPUT_FILES = {
    "formulation_specialist": RUN_DIR / "formulation_output.md",
    "rubber_chemistry_reviewer": RUN_DIR / "chemistry_review.md",
    "prediction_auditor": RUN_DIR / "prediction_audit.md",
}


class TerraRunReplayLlm(BaseLlm):
    """Replay one completed Terra MAS run through the real ADK agent tree."""

    agent_name: str
    output_text: str = ""
    routing_requests: dict[str, str] = Field(default_factory=dict)
    numerical_tool: dict[str, Any] = Field(default_factory=dict)
    recipe_prediction: str = ""
    worker_order: list[str] = Field(default_factory=list)
    _tool_called: bool = PrivateAttr(default=False)
    _tool_result: str = PrivateAttr(default="")
    _next_worker_index: int = PrivateAttr(default=0)

    def _capture_tool_result(self, llm_request: LlmRequest) -> None:
        if self._tool_result:
            return
        for content in reversed(llm_request.contents):
            for part in content.parts or []:
                response = part.function_response
                if response is None or response.name != self.numerical_tool.get("name"):
                    continue
                raw_result: Any = response.response
                if isinstance(raw_result, dict):
                    raw_result = raw_result.get(
                        "structuredContent", raw_result.get("result", raw_result)
                    )
                if isinstance(raw_result, str):
                    self._tool_result = raw_result
                else:
                    self._tool_result = json.dumps(
                        raw_result, ensure_ascii=False, indent=2
                    )
                return

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        if self.agent_name != "rubber_recipe_master":
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=self.output_text)],
                )
            )
            return

        if self.numerical_tool and not self._tool_called:
            self._tool_called = True
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name=self.numerical_tool["name"],
                            args=self.numerical_tool["arguments"],
                        )
                    ],
                )
            )
            return

        self._capture_tool_result(llm_request)
        next_worker = (
            self.worker_order[self._next_worker_index]
            if self._next_worker_index < len(self.worker_order)
            else None
        )
        if next_worker is not None:
            self._next_worker_index += 1
            content = types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name=next_worker,
                        args={
                            "request": self.routing_requests[next_worker].replace(
                                "{tool_result}", self._tool_result
                            )
                        },
                    )
                ],
            )
        else:
            content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=self.recipe_prediction)],
            )
        yield LlmResponse(content=content)


def build_replay_models(config: MASConfig) -> dict[str, BaseLlm]:
    routing = json.loads((RUN_DIR / "001_routing.json").read_text(encoding="utf-8"))
    routing_requests = {
        call["agent_name"]: call["request_template"] for call in routing["calls"]
    }
    worker_order = [call["agent_name"] for call in routing["calls"]]
    models: dict[str, BaseLlm] = {
        "rubber_recipe_master": TerraRunReplayLlm(
            model=MODEL,
            agent_name="rubber_recipe_master",
            routing_requests=routing_requests,
            numerical_tool=routing["tool_call"],
            worker_order=worker_order,
            recipe_prediction=(RUN_DIR / "recipe_prediction.md").read_text(
                encoding="utf-8"
            ),
        )
    }
    for worker in config.workers:
        models[worker.name] = TerraRunReplayLlm(
            model=MODEL,
            agent_name=worker.name,
            output_text=OUTPUT_FILES[worker.name].read_text(encoding="utf-8"),
        )
    return models


def create_app():
    config = MASConfig.model_validate_json(
        (HERE / "config.json").read_text(encoding="utf-8")
    )
    models = build_replay_models(config)
    resolution_order = iter(
        [worker.name for worker in config.workers] + [config.coordinator.name]
    )

    def resolve_model(model_name: str | None, worker_models: object) -> BaseLlm:
        del worker_models
        if model_name != MODEL:
            raise ValueError(f"Unexpected model: {model_name}")
        return models[next(resolution_order)]

    mas = MAS(mcp_servers=[PREDICTOR_SERVER])
    with patch("fedotmas.mas.builder._resolve_llm", side_effect=resolve_model):
        return mas.serve(
            config,
            name="rubber_recipe_mas_terra_replay",
            session_service_uri=f"sqlite:///{SESSION_DB.as_posix()}",
            web=True,
            host="0.0.0.0",
            port=8765,
            auto_create_session=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the completed Terra MAS run")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
