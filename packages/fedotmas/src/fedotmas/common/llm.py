from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.models.lite_llm import LiteLlm

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

    from fedotmas.config.settings import ModelConfig

__all__ = ["make_llm"]


def make_llm(cfg: ModelConfig) -> BaseLlm:
    """Create a ``LiteLlm`` instance from *cfg*."""
    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    return LiteLlm(model=cfg.model, **kwargs)
