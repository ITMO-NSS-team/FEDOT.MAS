from __future__ import annotations

from typing import Any

from fedotmas.interfaces.runner import RunnerProtocol


class ADKBackend:
    """Google ADK backend for FEDOT.MAS."""

    def create_runner(
        self,
        *,
        session_service: Any | None = None,
        memory_service: Any | None = None,
    ) -> RunnerProtocol:
        from fedotmas.backends.adk.runner import ADKRunner

        return ADKRunner(
            session_service=session_service,
            memory_service=memory_service,
        )
