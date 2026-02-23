from __future__ import annotations

from typing import Any

from fedotmas_synapse.checkpoint import CheckpointCallback
from fedotmas_synapse.otel import OtelEventCallback
from fedotmas_synapse.session import MongoSessionService


class SynapsePlugin:
    """Facade that configures all Synapse integrations for MAS."""

    def __init__(self, mongo_uri: str, otel_endpoint: str | None = None) -> None:
        self.session_service = MongoSessionService(mongo_uri)
        self.otel_callback = OtelEventCallback() if otel_endpoint else None
        self.checkpoint = CheckpointCallback(mongo_uri)

    def mas_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to MAS() constructor."""
        kw: dict[str, Any] = {"session_service": self.session_service}
        if self.otel_callback:
            kw["event_callback"] = self.otel_callback
        kw["before_agent_callbacks"] = [self.checkpoint.before_agent]
        kw["after_agent_callbacks"] = [self.checkpoint.after_agent]
        return kw
