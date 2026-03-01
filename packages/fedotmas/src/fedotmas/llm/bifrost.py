"""Bifrost LLM provider — local OpenAI-compatible gateway."""

from __future__ import annotations

import os

from openai import AsyncOpenAI
from pydantic import Field
from typing_extensions import override

from fedotmas.llm._openai_compat import OpenAICompatibleLlm


class BifrostLlm(OpenAICompatibleLlm):
    """Direct access to a local Bifrost gateway (vLLM / Ollama proxy)."""

    api_base: str = Field(
        default_factory=lambda: os.getenv(
            "FEDOTMAS_BIFROST_BASE_URL", "http://localhost:8080/api/v1"
        )
    )

    def _build_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key or "bifrost",
        )

    @override
    def _resolve_model(self, raw: str) -> str:
        return raw.removeprefix("bifrost/")

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return [r"bifrost/.*"]
