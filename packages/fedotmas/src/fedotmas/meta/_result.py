from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fedotmas._settings import ModelConfig
from fedotmas.maw.models import MAWConfig


@dataclass
class MetaAgentResult:
    config: MAWConfig | Any
    worker_models: list[ModelConfig] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed: float = 0.0
