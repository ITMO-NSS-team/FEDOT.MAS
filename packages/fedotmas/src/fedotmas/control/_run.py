from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from fedotmas.plugins._checkpoint import Checkpoint


@dataclass
class RunError:
    agent_name: str
    message: str


@dataclass
class PipelineStep:
    """A top-level pipeline step that is about to execute."""

    name: str  # ADK agent name ("researcher", "par_1", "loop_2")
    index: int  # position in the top-level sequence
    state: dict[str, Any]  # accumulated state from completed prior steps
    agent: MAWAgentConfig | None = (
        None  # agent config for this step (None for composite steps)
    )


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
