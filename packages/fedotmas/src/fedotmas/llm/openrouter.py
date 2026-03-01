"""OpenRouter LLM provider — direct access via ``openai.AsyncOpenAI``."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI
from typing_extensions import override

from fedotmas.llm._openai_compat import OpenAICompatibleLlm


class OpenRouterLlm(OpenAICompatibleLlm):
    """Direct OpenRouter access with native headers and provider routing."""

    api_base: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None
    provider_order: list[str] | None = None
    provider_sort: str | None = None
    quantizations: list[str] | None = None

    def _build_client(self) -> AsyncOpenAI:
        headers: dict[str, str] = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key or os.getenv("OPENROUTER_API_KEY", ""),
            default_headers=headers or None,
        )

    def _extra_create_kwargs(self) -> dict[str, Any]:
        provider: dict[str, Any] = {}
        if self.provider_order:
            provider["order"] = self.provider_order
        if self.provider_sort:
            provider["sort"] = self.provider_sort
        if self.quantizations:
            provider["quantizations"] = self.quantizations
        return {"extra_body": {"provider": provider}} if provider else {}

    @override
    def _resolve_model(self, raw: str) -> str:
        return raw.removeprefix("openrouter/")

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return [r"openrouter/.*"]
