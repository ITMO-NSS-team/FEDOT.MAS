from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONFIG_NAME = "config.toml"


@dataclass(frozen=True)
class ModelSettings:
    default: str = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class MetaAgentSettings:
    model: str = ""
    temperature: float = 0.3


@dataclass(frozen=True)
class PipelineSettings:
    max_loop_iterations: int = 3


@dataclass(frozen=True)
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    meta_agent: MetaAgentSettings = field(default_factory=MetaAgentSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)

    @property
    def default_model(self) -> str:
        return os.getenv("FEDOTMAS_DEFAULT_MODEL") or self.model.default

    @property
    def meta_agent_model(self) -> str:
        return (
            os.getenv("FEDOTMAS_META_AGENT_MODEL")
            or self.meta_agent.model
            or self.default_model
        )

    @property
    def meta_agent_temperature(self) -> float:
        env = os.getenv("FEDOTMAS_META_AGENT_TEMPERATURE")
        return float(env) if env else self.meta_agent.temperature

    @property
    def max_loop_iterations(self) -> int:
        env = os.getenv("FEDOTMAS_DEFAULT_MAX_LOOP_ITERATIONS")
        return int(env) if env else self.pipeline.max_loop_iterations


def _find_config() -> Path | None:
    """Walk up from cwd looking for ``config.toml``."""
    cur = Path.cwd()
    for parent in (cur, *cur.parents):
        candidate = parent / _DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from a TOML file.

    Resolution order for each value:
        1. Environment variable (``FEDOTMAS_*``)
        2. Value in the TOML file
        3. Built-in default
    """
    if path is not None:
        config_path = Path(path)
    else:
        config_path = _find_config()

    if config_path is None or not config_path.is_file():
        return Settings()

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    model_raw = raw.get("model", {})
    meta_raw = raw.get("meta_agent", {})
    pipeline_raw = raw.get("pipeline", {})

    return Settings(
        model=ModelSettings(
            default=model_raw.get("default", ModelSettings.default),
        ),
        meta_agent=MetaAgentSettings(
            model=meta_raw.get("model", MetaAgentSettings.model),
            temperature=meta_raw.get("temperature", MetaAgentSettings.temperature),
        ),
        pipeline=PipelineSettings(
            max_loop_iterations=pipeline_raw.get(
                "max_loop_iterations", PipelineSettings.max_loop_iterations
            ),
        ),
    )


settings = load_settings()
