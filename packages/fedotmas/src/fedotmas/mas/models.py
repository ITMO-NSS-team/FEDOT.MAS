from __future__ import annotations

from pydantic import BaseModel, model_validator

from fedotmas._settings import validate_model_name


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
        validate_model_name(self.model)
        return self


class MASConfig(BaseModel):
    """Configuration for a routing-based multi-agent system.

    The ``coordinator`` is the root agent that uses ADK AutoFlow to
    dynamically route tasks to ``workers`` based on their descriptions.
    """

    coordinator: MASAgentConfig
    workers: list[MASAgentConfig]

    def __str__(self) -> str:
        model = f"  model: {self.coordinator.model}" if self.coordinator.model else ""
        lines = [f"Coordinator: {self.coordinator.name}{model}", "Workers:"]
        for w in self.workers:
            lines.append(f"  {w.name:<20} {w.description!r}")
        return "\n".join(lines)

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
