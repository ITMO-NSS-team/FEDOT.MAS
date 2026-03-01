from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from fedotmas.config.settings import (
    ModelConfig,
    get_meta_model,
    get_meta_temperature,
    get_worker_models,
    resolve_model_config,
)

T = TypeVar("T", bound=BaseModel)


def resolve_meta_and_workers(
    meta_model: str | ModelConfig | None,
    worker_models: list[str | ModelConfig] | None,
    temperature: float | None,
) -> tuple[ModelConfig, list[ModelConfig], float]:
    """Resolve meta model, worker models, and temperature with env fallback."""
    resolved_meta = (
        resolve_model_config(meta_model)
        if meta_model
        else resolve_model_config(get_meta_model())
    )
    resolved_workers = (
        [resolve_model_config(m) for m in worker_models]
        if worker_models
        else [resolve_model_config(m) for m in get_worker_models()]
    )
    resolved_temp = temperature if temperature is not None else get_meta_temperature()
    return resolved_meta, resolved_workers, resolved_temp


def format_server_descriptions(descriptions: dict[str, str]) -> str:
    """Format MCP server descriptions for LLM prompts."""
    if not descriptions:
        return "No MCP tools available."
    return "\n".join(f"- **{name}**: {desc}" for name, desc in descriptions.items())


def parse_llm_output(raw: Any, schema: type[T]) -> T:
    """Parse raw LLM output (dict or JSON string) into a Pydantic model."""
    if isinstance(raw, dict):
        return schema.model_validate(raw)
    if isinstance(raw, str):
        return schema.model_validate_json(raw)
    raise TypeError(f"Unexpected LLM output type: {type(raw)}")
