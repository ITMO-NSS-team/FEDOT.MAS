from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import Checkpoint


@dataclass
class RunError:
    agent_name: str
    message: str


@dataclass
class ControlledRun:
    config: MAWConfig
    status: Literal["success", "error"]
    state: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    error: RunError | None = None

    @property
    def result(self) -> dict[str, Any]:
        """Final pipeline state (convenience alias)."""
        return self.state
