"""Custom OpenAI-compatible LLM providers for Google ADK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm

from fedotmas.llm._openai_compat import OpenAICompatibleLlm
from fedotmas.llm.bifrost import BifrostLlm
from fedotmas.llm.openrouter import OpenRouterLlm

if TYPE_CHECKING:
    from fedotmas.config.settings import ModelConfig

__all__ = [
    "BifrostLlm",
    "OpenAICompatibleLlm",
    "OpenRouterLlm",
    "make_llm",
]


def make_llm(cfg: ModelConfig) -> BaseLlm:
    """Create the appropriate ``BaseLlm`` based on ``cfg.provider``."""
    if cfg.provider == "openrouter":
        return OpenRouterLlm(
            model=cfg.model,
            api_key=cfg.api_key,
            api_base=cfg.api_base or "https://openrouter.ai/api/v1",
        )
    if cfg.provider == "bifrost":
        kwargs: dict[str, str | None] = {"model": cfg.model, "api_key": cfg.api_key}
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        return BifrostLlm(**kwargs)

    # Default: LiteLlm
    llm_kwargs: dict[str, str | None] = {}
    if cfg.api_base:
        llm_kwargs["api_base"] = cfg.api_base
    if cfg.api_key:
        llm_kwargs["api_key"] = cfg.api_key
    return LiteLlm(model=cfg.model, **llm_kwargs)
