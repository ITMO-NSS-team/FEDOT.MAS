from __future__ import annotations

import asyncio
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
from tenacity import RetryError

from fedotmas.common.logging import get_logger
from fedotmas.plugins import WebSearchLimitExceeded, WebSearchLimitPlugin

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
    #: Agents that spent their whole token budget without emitting content.
    #: The pipeline carries on past them, so callers need this to tell "the
    #: model answered wrongly" from "the model never got to answer".
    truncated_agents: list[str] = field(default_factory=list)


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
    timeout: float | None = None,
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
        timeout: Optional wall-clock budget (seconds) for pipeline *execution*.
            On expiry the run is stopped and the partial ``session.state``
            accumulated so far is returned instead of raising — so any
            sub-answers already produced can still be salvaged.

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

    root_name = app.root_agent.name  # ty: ignore[unresolved-attribute]
    _log.info("Pipeline run started | pipeline={}", root_name)
    total_prompt = 0
    total_completion = 0
    truncated_agents: list[str] = []
    pipeline_start = time.monotonic()

    async with Runner(
        app=app,
        session_service=session_service,
        memory_service=memory_service,
    ) as runner:
        try:
            total_prompt, total_completion = await _consume_with_timeout(
                runner=runner,
                user_id=user_id,
                session_id=session.id,
                message=message,
                total_prompt=total_prompt,
                total_completion=total_completion,
                truncated_agents=truncated_agents,
                timeout=timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            _log.warning(
                "Pipeline execution exceeded {}s budget; salvaging partial state",
                timeout,
            )
        except BaseException as exc:
            if not _is_search_limit_exceeded(exc):
                raise

            _log.warning(
                "Search limit exceeded; requesting final answer from current evidence"
            )
            # Disable web/search/browser tools for the finalization turn so the
            # agent cannot re-trigger the budget (which would re-raise uncaught)
            # or loop on error results burning the remaining time budget.
            _enter_finalize_mode(app.plugins)
            recovery_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=SEARCH_LIMIT_RECOVERY_PROMPT)],
            )
            try:
                total_prompt, total_completion = await _consume_with_timeout(
                    runner=runner,
                    user_id=user_id,
                    session_id=session.id,
                    message=recovery_message,
                    total_prompt=total_prompt,
                    total_completion=total_completion,
                    truncated_agents=truncated_agents,
                    timeout=timeout,
                )
            except (asyncio.TimeoutError, TimeoutError):
                _log.warning("Finalization turn timed out; salvaging partial state")
            except BaseException as exc2:
                if _is_search_limit_exceeded(exc2):
                    _log.warning(
                        "Finalization turn still hit budget; salvaging partial state"
                    )
                else:
                    raise

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
        truncated_agents=truncated_agents,
    )


async def _consume_with_timeout(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    message: types.Content,
    total_prompt: int,
    total_completion: int,
    truncated_agents: list[str],
    timeout: float | None,
) -> tuple[int, int]:
    """Consume runner events, optionally bounded by *timeout* seconds.

    On timeout the consuming coroutine is cancelled (stopping the pipeline) and
    ``asyncio.TimeoutError`` propagates; the caller salvages the partial state
    already persisted to the session service.
    """
    coro = _consume_runner_events(
        runner=runner,
        user_id=user_id,
        session_id=session_id,
        message=message,
        total_prompt=total_prompt,
        total_completion=total_completion,
        truncated_agents=truncated_agents,
    )
    if timeout is None or timeout <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout)


async def _consume_runner_events(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    message: types.Content,
    total_prompt: int,
    total_completion: int,
    truncated_agents: list[str],
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
            if event.error_code == types.FinishReason.MAX_TOKENS:
                # The agent spent its whole budget without emitting content, so
                # ADK reports an error rather than a short answer.  That is one
                # step falling short, not a reason to discard what every earlier
                # step produced: let the pipeline carry on with this output
                # empty, the same way a timeout salvages partial state.
                _log.warning(
                    "Agent '{}' produced no content within its token budget; "
                    "continuing with an empty result for this step",
                    event.author,
                )
                if event.author:
                    truncated_agents.append(event.author)
                continue

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


def _is_search_limit_exceeded(
    exc: BaseException, _seen: set[int] | None = None
) -> bool:
    """True if *exc* is, or wraps, a ``WebSearchLimitExceeded``.

    The exception reaches us wrapped — a worker-model retry produces
    ``RetryError[WebSearchLimitExceeded]`` — so we must peel ``RetryError``,
    exception groups, and ``__cause__``/``__context__`` chains, not just check
    the outermost type.
    """
    if _seen is None:
        _seen = set()
    if exc is None or id(exc) in _seen:
        return False
    _seen.add(id(exc))

    if isinstance(exc, WebSearchLimitExceeded):
        return True
    if isinstance(exc, RetryError):
        try:
            inner = exc.last_attempt.exception()
        except Exception:
            inner = None
        if isinstance(inner, BaseException) and _is_search_limit_exceeded(inner, _seen):
            return True
    if isinstance(exc, BaseExceptionGroup):
        if any(_is_search_limit_exceeded(item, _seen) for item in exc.exceptions):
            return True
    for nxt in (exc.__cause__, exc.__context__):
        if isinstance(nxt, BaseException) and _is_search_limit_exceeded(nxt, _seen):
            return True
    return False


def _enter_finalize_mode(plugins: list[BasePlugin]) -> None:
    """Switch web-search/scraping limit plugins into finalization mode.

    In this mode the plugins block every web/search/browser tool call (returning
    a terse "answer now" result instead of raising), so the post-budget
    finalization turn can produce an answer without re-tripping the limit.
    """
    for plugin in plugins:
        if isinstance(plugin, WebSearchLimitPlugin):
            plugin.finalizing = True
