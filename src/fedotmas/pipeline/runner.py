from __future__ import annotations

import uuid
from typing import Any

from google.adk import Runner
from google.adk.agents.base_agent import BaseAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.pipeline.runner")


async def run_pipeline(
    agent: BaseAgent,
    user_query: str,
    *,
    app_name: str = "fedotmas",
    user_id: str = "user",
    session_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute an ADK agent tree and return the final session state.

    Args:
        agent: Root agent (output of ``builder.build``).
        user_query: The user's task.
        app_name: Application name for the ADK runner.
        user_id: User identifier for the session.
        session_id: Optional session id (auto-generated if omitted).
        initial_state: Extra keys to inject into ``session.state`` before
            execution (``user_query`` is always set automatically).

    Returns:
        The full ``session.state`` dict after pipeline execution.
    """
    _log.info("Creating session | app={} user={}", app_name, user_id)
    session_service = InMemorySessionService()
    session_id = session_id or uuid.uuid4().hex

    # Pre-populate state with user_query + any caller-supplied keys.
    state: dict[str, Any] = {"user_query": user_query}
    if initial_state:
        state.update(initial_state)

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
    )

    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_query)],
    )

    _log.info("Pipeline run started | agent={}", agent.name)
    # Consume every event; we only care about the final state.
    async for _event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        pass

    # Re-fetch the session to get the fully-updated state.
    final_session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session.id,
    )
    result = dict(final_session.state) if final_session else state
    _log.info("Pipeline run complete | state_keys={}", len(result))
    return result
