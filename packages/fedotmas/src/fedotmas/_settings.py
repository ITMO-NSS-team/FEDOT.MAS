from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import find_dotenv, load_dotenv

from fedotmas.common.logging import get_logger

# Working directory first: installed as a dependency, only the consumer's project
# holds a .env, never the package tree.  The bare call is the clone fallback.
_DOTENV_PATH = find_dotenv(usecwd=True) or find_dotenv()
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH)

_log = get_logger("fedotmas.settings")


DEFAULT_META_MODEL = "qwen/qwen3.6-finetuned"
DEFAULT_WORKER_MODELS: list[str] = ["openai/gpt-5-mini"]
DEFAULT_META_TEMPERATURE = 0.3
DEFAULT_MAX_LOOP_ITERATIONS = 3


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single LLM model endpoint."""

    model: str  # provider/model-name, e.g. "openai/gpt-4o"
    api_base: str | None = None  # custom endpoint URL
    api_key: str | None = None  # per-model API key
    extra_body: dict[str, Any] | None = None  # provider-specific request body


def resolve_model_config(value: str | ModelConfig) -> ModelConfig:
    """Convert a plain string to ModelConfig, picking up env defaults."""
    if isinstance(value, ModelConfig):
        return value
    return ModelConfig(
        model=value,
        api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        extra_body=_extra_body_from_env(),
    )


def _env_list(name: str) -> list[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _env_optional_flag(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _provider_routing_from_env(prefix: str = "FEDOTMAS_PROVIDER") -> dict[str, Any]:
    provider: dict[str, Any] = {}
    for field in ("order", "only", "ignore", "quantizations"):
        values = _env_list(f"{prefix}_{field.upper()}")
        if values:
            provider[field] = values
    for field in (
        "allow_fallbacks",
        "require_parameters",
        "zdr",
        "enforce_distillable_text",
    ):
        value = _env_optional_flag(f"{prefix}_{field.upper()}")
        if value is not None:
            provider[field] = value
    sort_by = os.getenv(f"{prefix}_SORT_BY")
    sort_partition = os.getenv(f"{prefix}_SORT_PARTITION")
    if sort_by or sort_partition:
        sort: dict[str, str] = {}
        if sort_by:
            sort["by"] = sort_by
        if sort_partition:
            sort["partition"] = sort_partition
        provider["sort"] = sort
    else:
        value = os.getenv(f"{prefix}_SORT")
        if value:
            provider["sort"] = value

    for field in ("data_collection",):
        value = os.getenv(f"{prefix}_{field.upper()}")
        if value:
            provider[field] = value
    return provider


def _extra_body_from_env(prefix: str = "FEDOTMAS_PROVIDER") -> dict[str, Any] | None:
    provider = _provider_routing_from_env(prefix)
    if not provider:
        return None
    return {"provider": provider}


def get_meta_model() -> str:
    return (
        os.getenv("FEDOTMAS_META_AGENT_MODEL")
        or os.getenv("FEDOTMAS_DEFAULT_MODEL")
        or DEFAULT_META_MODEL
    )


def get_reflection_model() -> str:
    return os.getenv("FEDOTMAS_REFLECTION_MODEL") or get_meta_model()


def get_worker_models() -> list[str]:
    env = os.getenv("FEDOTMAS_WORKER_MODELS")
    if env:
        models = [m.strip() for m in env.split(",") if m.strip()]
        if models:
            return models
        _log.warning("FEDOTMAS_WORKER_MODELS={!r} names no models, falling back", env)
    default = os.getenv("FEDOTMAS_DEFAULT_MODEL")
    if default:
        return [default]
    return list(DEFAULT_WORKER_MODELS)


def get_meta_temperature() -> float:
    env = os.getenv("FEDOTMAS_META_AGENT_TEMPERATURE")
    if not env:
        return DEFAULT_META_TEMPERATURE
    try:
        return float(env)
    except ValueError:
        raise ValueError(
            f"Invalid FEDOTMAS_META_AGENT_TEMPERATURE='{env}', expected a float"
        ) from None


def validate_model_name(model: str | None) -> str | None:
    """Validate that a model name includes a provider prefix."""
    if model is not None and "/" not in model:
        raise ValueError(
            f"Model '{model}' must include a provider prefix, "
            f"e.g. 'openai/{model}' or 'openrouter/{model}'"
        )
    return model


def get_max_loop_iterations() -> int:
    env = os.getenv("FEDOTMAS_DEFAULT_MAX_LOOP_ITERATIONS")
    if not env:
        return DEFAULT_MAX_LOOP_ITERATIONS
    try:
        return int(env)
    except ValueError:
        raise ValueError(
            f"Invalid FEDOTMAS_DEFAULT_MAX_LOOP_ITERATIONS='{env}', expected an integer"
        ) from None
