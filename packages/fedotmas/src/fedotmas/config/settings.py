from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_META_MODEL = "openai/gpt-oss-120b"
DEFAULT_WORKER_MODELS: list[str] = ["openai/gpt-oss-120b"]
DEFAULT_META_TEMPERATURE = 0.3
DEFAULT_MAX_LOOP_ITERATIONS = 3
VALID_PROXIES = ("openrouter", "bifrost", "litellm")


def get_default_proxy() -> str | None:
    """Read FEDOTMAS_DEFAULT_PROXY env var. Returns None for litellm."""
    env = os.getenv("FEDOTMAS_DEFAULT_PROXY")
    if not env:
        return None
    env = env.strip().lower()
    if env == "litellm":
        return None  # litellm represented as None internally
    if env not in VALID_PROXIES:
        raise ValueError(
            f"Invalid FEDOTMAS_DEFAULT_PROXY='{env}', "
            f"expected one of: {', '.join(VALID_PROXIES)}"
        )
    return env


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single LLM model endpoint."""

    model: str                        # provider/model-name, e.g. "openai/gpt-4o"
    api_base: str | None = None       # custom endpoint URL
    api_key: str | None = None        # per-model API key
    proxy: str | None = None          # "openrouter" | "bifrost" | None (=litellm)


def resolve_model_config(value: str | ModelConfig) -> ModelConfig:
    """Convert a plain string to ModelConfig if needed."""
    if isinstance(value, ModelConfig):
        return value
    return ModelConfig(model=value, proxy=get_default_proxy())


def get_meta_model() -> str:
    return os.getenv("FEDOTMAS_META_MODEL") or DEFAULT_META_MODEL


def get_worker_models() -> list[str]:
    env = os.getenv("FEDOTMAS_WORKER_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
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
