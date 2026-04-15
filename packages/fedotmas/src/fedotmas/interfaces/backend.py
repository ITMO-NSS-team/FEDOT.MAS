from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fedotmas.interfaces.runner import RunnerProtocol


@runtime_checkable
class BackendProtocol(Protocol):
    """Factory protocol for creating backend-specific runners."""

    def create_runner(
        self,
        *,
        session_service: Any | None = None,
        memory_service: Any | None = None,
    ) -> RunnerProtocol:
        """Create a runner instance for executing agent trees."""
        ...
