from __future__ import annotations

from typing import Any

from fedotmas.backends import get_backend
from fedotmas.common.logging import get_logger
from fedotmas.interfaces.agent import AgentTree
from fedotmas.interfaces.middleware import MiddlewareProtocol
from fedotmas.interfaces.runner import PipelineResult

_log = get_logger("fedotmas.core.runner")

# Re-export PipelineResult for backward compatibility
__all__ = ["PipelineResult", "run_pipeline"]


async def run_pipeline(
    agent_tree: AgentTree,
    user_query: str,
    *,
    session_service: Any | None = None,
    memory_service: Any | None = None,
    middlewares: list[MiddlewareProtocol] | None = None,
    backend_plugins: list[Any] | None = None,
    initial_state: dict[str, Any] | None = None,
    backend: str = "adk",
) -> PipelineResult:
    """Execute an agent tree via the specified backend and return the final state.

    This is a convenience wrapper that creates a backend runner and delegates.
    """
    be = get_backend(backend)
    runner = be.create_runner(
        session_service=session_service,
        memory_service=memory_service,
    )
    return await runner.run_pipeline(
        agent_tree,
        user_query,
        initial_state=initial_state,
        middlewares=middlewares,
        backend_plugins=backend_plugins,
    )
