from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk import Runner
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.memory import BaseMemoryService
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types

from fedotmas.common.logging import get_logger
from fedotmas.plugins import WebSearchLimitExceeded

_log = get_logger("fedotmas.core.runner")

SEARCH_LIMIT_RECOVERY_PROMPT = (
    "SearchLimitExceeded: web/search exploration budget is exhausted. "
    "Stop exploration immediately. Do not call any more web, browser, or search "
    "tools. Synthesize the best possible final answer from the evidence already "
    "available in the conversation and session state. If evidence is incomplete, "
    "state the best supported answer concisely."
)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    state: dict[str, Any] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed: float = 0.0


async def run_pipeline(
    agent_or_app: BaseAgent | App,
    user_query: str,
    *,
    session_service: BaseSessionService | None = None,
    memory_service: BaseMemoryService | None = None,
    plugins: list[BasePlugin] | None = None,
    app_name: str = "fedotmas",
    user_id: str = "user",
    session_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
) -> PipelineResult:
    """Execute an ADK agent tree and return the final session state.

    Args:
        agent_or_app: Root agent or ``App`` (output of ``builder.build`` or
            ``MAS.build_app``).  When an ``App`` is passed, its bundled
            plugins are used and the ``plugins``/``app_name`` parameters
            are ignored.
        user_query: The user's task.
        plugins: ADK plugins registered on the Runner. Only used when
            *agent_or_app* is a ``BaseAgent`` (legacy path).
        app_name: Application name for the ADK runner. Only used when
            *agent_or_app* is a ``BaseAgent`` (legacy path).
        user_id: User identifier for the session.
        session_id: Optional session id (auto-generated if omitted).
        initial_state: Extra keys to inject into ``session.state`` before
            execution (``user_query`` is always set automatically).

    Returns:
        The full ``session.state`` dict after pipeline execution.
    """
    if isinstance(agent_or_app, App):
        app = agent_or_app
        effective_name = app.name
    else:
        app = App(
            name=app_name,
            root_agent=agent_or_app,
            plugins=list(plugins or []),
        )
        effective_name = app_name

    _log.debug("Creating session | app={} user={}", effective_name, user_id)
    session_service = session_service or InMemorySessionService()
    session_id = session_id or uuid.uuid4().hex

    # Pre-populate state with user_query + any caller-supplied keys.
    state: dict[str, Any] = {"user_query": user_query}
    if initial_state:
        state.update(initial_state)

    session = await session_service.create_session(
        app_name=effective_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_query)],
    )

    _log.info("Pipeline run started | pipeline={}", app.root_agent.name)
    total_prompt = 0
    total_completion = 0
    pipeline_start = time.monotonic()

    async with Runner(
        app=app,
        session_service=session_service,
        memory_service=memory_service,
    ) as runner:
        try:
            total_prompt, total_completion = await _consume_runner_events(
                runner=runner,
                user_id=user_id,
                session_id=session.id,
                message=message,
                total_prompt=total_prompt,
                total_completion=total_completion,
            )
        except BaseException as exc:
            if not _is_search_limit_exceeded(exc):
                raise

            _log.warning(
                "Search limit exceeded; requesting final answer from current evidence"
            )
            recovery_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=SEARCH_LIMIT_RECOVERY_PROMPT)],
            )
            total_prompt, total_completion = await _consume_runner_events(
                runner=runner,
                user_id=user_id,
                session_id=session.id,
                message=recovery_message,
                total_prompt=total_prompt,
                total_completion=total_completion,
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
        app_name=effective_name,
        user_id=user_id,
        session_id=session.id,
    )
    if final_session is None:
        raise RuntimeError(
            f"Session '{session.id}' lost after pipeline execution — results unavailable"
        )
    return PipelineResult(
        state=dict(final_session.state),
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        elapsed=total_elapsed,
    )


async def _consume_runner_events(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    message: types.Content,
    total_prompt: int,
    total_completion: int,
) -> tuple[int, int]:
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.partial:
            continue

        # Token accumulation (business logic — stays in runner)
        if event.usage_metadata:
            um = event.usage_metadata
            total_prompt += um.prompt_token_count or 0
            total_completion += um.candidates_token_count or 0

        # Error handling (control flow — stays in runner)
        if event.error_code:
            _log.error(
                "LLM error | agent={} code={} msg={}",
                event.author,
                event.error_code,
                event.error_message,
            )
            raise RuntimeError(
                f"Agent '{event.author}' failed with error {event.error_code}: "
                f"{event.error_message}"
            )

    return total_prompt, total_completion


def _is_search_limit_exceeded(exc: BaseException) -> bool:
    if isinstance(exc, WebSearchLimitExceeded):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_search_limit_exceeded(item) for item in exc.exceptions)
    return False
