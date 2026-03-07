from __future__ import annotations

from pydantic import BaseModel, model_validator

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.mas.models")


class MASAgentConfig(BaseModel):
    """Configuration for an agent in a routing-based multi-agent system.

    The ``description`` field is critical — ADK AutoFlow uses it to decide
    which agent to route tasks to.
    """

    name: str
    description: str
    instruction: str
    model: str | None = None
    tools: list[str] = []
    output_key: str | None = None

    @model_validator(mode="after")
    def _normalize_model(self) -> MASAgentConfig:
        if self.model is not None and "/" not in self.model:
            _log.warning(
                "Model '{}' has no provider prefix, assuming 'openai/{}'",
                self.model,
                self.model,
            )
            self.model = f"openai/{self.model}"
        return self


class MASConfig(BaseModel):
    """Configuration for a routing-based multi-agent system.

    The ``coordinator`` is the root agent that uses ADK AutoFlow to
    dynamically route tasks to ``workers`` based on their descriptions.
    """

    coordinator: MASAgentConfig
    workers: list[MASAgentConfig]

    @model_validator(mode="after")
    def _validate_config(self) -> MASConfig:
        if len(self.workers) < 1:
            raise ValueError("At least one worker is required")

        # Unique names across coordinator + workers
        names = [self.coordinator.name] + [w.name for w in self.workers]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"Duplicate agent name: '{name}'")
            seen.add(name)

        return self
