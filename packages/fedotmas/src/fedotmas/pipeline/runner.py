from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.adk import Runner
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.memory import BaseMemoryService
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.pipeline.runner")

EventCallback = Callable[[Event], Awaitable[None] | None]


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    state: dict[str, Any] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed: float = 0.0


async def run_pipeline(
    agent: BaseAgent,
    user_query: str,
    *,
    session_service: BaseSessionService | None = None,
    memory_service: BaseMemoryService | None = None,
    plugins: list[BasePlugin] | None = None,
    event_callback: EventCallback | None = None,
    app_name: str = "fedotmas",
    user_id: str = "user",
    session_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
) -> PipelineResult:
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
    _log.debug("Creating session | app={} user={}", app_name, user_id)
    session_service = session_service or InMemorySessionService()
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

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_query)],
    )

    _log.info("Pipeline run started | pipeline={}", agent.name)
    total_prompt = 0
    total_completion = 0
    pipeline_start = time.monotonic()

    async with Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
        memory_service=memory_service,
        plugins=plugins or [],
    ) as runner:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
        ):
            if event.partial:
                continue

            if event_callback:
                result = event_callback(event)
                if result is not None:
                    await result

            # Tool calls
            for fc in event.get_function_calls():
                _log.info(
                    "Tool call | agent={} tool={} args={}", event.author, fc.name, fc.args
                )

            # Tool responses
            for fr in event.get_function_responses():
                _log.info("Tool result | agent={} tool={}", event.author, fr.name)

            # Token usage
            if event.usage_metadata:
                um = event.usage_metadata
                prompt = um.prompt_token_count or 0
                completion = um.candidates_token_count or 0
                total_prompt += prompt
                total_completion += completion
                if prompt or completion:
                    _log.info(
                        "Tokens | agent={} prompt={} completion={}",
                        event.author,
                        prompt,
                        completion,
                    )

            # Text response (final, no function calls)
            if event.content and event.content.parts and not event.get_function_calls():
                texts = [p.text for p in event.content.parts if p.text]
                if texts:
                    preview = texts[0][:200]
                    _log.trace("Response | agent={} text={}", event.author, preview)

            # State changes
            if event.actions.state_delta:
                _log.info(
                    "State update | agent={} keys={}",
                    event.author,
                    list(event.actions.state_delta.keys()),
                )

            # Errors
            if event.error_code:
                _log.error(
                    "LLM error | agent={} code={} msg={}",
                    event.author,
                    event.error_code,
                    event.error_message,
                )

    total_elapsed = time.monotonic() - pipeline_start
    _log.info(
        "Pipeline complete | total_elapsed={:.1f}s total_prompt={} total_completion={}",
        total_elapsed,
        total_prompt,
        total_completion,
    )

    # Re-fetch the session to get the fully-updated state.
    final_session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session.id,
    )
    return PipelineResult(
        state=dict(final_session.state) if final_session else state,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        elapsed=total_elapsed,
    )
