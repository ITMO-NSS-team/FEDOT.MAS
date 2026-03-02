from __future__ import annotations

import os

from openai import AsyncOpenAI
from pydantic import Field
from typing_extensions import override

from fedotmas.common.logging import get_logger
from fedotmas.llm._openai_compat import OpenAICompatibleLlm

_log = get_logger("fedotmas.llm.bifrost")


class BifrostLlm(OpenAICompatibleLlm):
    """Direct access to a local Bifrost gateway (vLLM / Ollama proxy)."""

    api_base: str = Field(
        default_factory=lambda: os.getenv(
            "FEDOTMAS_BIFROST_BASE_URL", "http://localhost:9090/v1"
        )
    )

    def _build_client(self) -> AsyncOpenAI:
        key = self.api_key or "bifrost"
        if not self.api_key:
            _log.debug("No Bifrost API key provided, using placeholder 'bifrost'")
        return AsyncOpenAI(
            base_url=self.api_base,
            api_key=key,
        )

    @override
    def _resolve_model(self, raw_model: str) -> str:
        return raw_model.removeprefix("bifrost/")

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return [r"bifrost/.*"]
