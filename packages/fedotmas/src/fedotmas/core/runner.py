from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from google.adk import Runner
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.memory import BaseMemoryService
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types

from fedotmas.common.logging import get_logger
from fedotmas.maw.handoffs import unresolved_execution_issues

_log = get_logger("fedotmas.core.runner")

# ADK keeps its own copy private (base_llm_flow._NO_CONTENT_ERROR_CODE).
_NO_CONTENT_ERROR_CODE = "MODEL_RETURNED_NO_CONTENT"


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    state: dict[str, Any] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed: float = 0.0
    #: Agents that finished a step without emitting content: they spent their
    #: whole token budget, or the model returned an empty response.
    #: The pipeline carries on past them, so callers need this to tell "the
    #: model answered wrongly" from "the model never got to answer".
    truncated_agents: list[str] = field(default_factory=list)
    status: Literal["completed", "timed_out", "failed", "limited", "incomplete"] = (
        "completed"
    )


class PipelineExecutionError(RuntimeError):
    """Pipeline failure that retains state and token usage seen before it."""

    def __init__(self, cause: Exception, result: PipelineResult) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.result = result


@dataclass
class _TokenUsage:
    """Mutable counters so cancellation cannot discard consumed event usage."""

    prompt: int = 0
    completion: int = 0


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
            On expiry the run is stopped and partial state is returned with
            ``status='timed_out'`` for diagnostics.

    Returns:
        A ``PipelineResult`` containing full session state and execution status.
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
    usage = _TokenUsage()
    truncated_agents: list[str] = []
    pipeline_start = time.monotonic()
    failure: Exception | None = None
    timed_out = False

    async with Runner(
        app=app,
        session_service=session_service,
        memory_service=memory_service,
    ) as runner:
        try:
            await _consume_with_timeout(
                runner=runner,
                user_id=user_id,
                session_id=session.id,
                message=message,
                usage=usage,
                truncated_agents=truncated_agents,
                timeout=timeout,
            )
        except TimeoutError:
            timed_out = True
            _log.warning(
                "Pipeline execution exceeded {}s budget; preserving partial state",
                timeout,
            )
        except Exception as exc:  # noqa: BLE001 - retain partial accounting for any agent failure
            failure = exc

    total_elapsed = time.monotonic() - pipeline_start
    _log.info(
        "Pipeline complete | total_elapsed={:.1f}s total_prompt={} total_completion={}",
        total_elapsed,
        usage.prompt,
        usage.completion,
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
    metadata = final_session.state.get("_fedotmas_execution", {})
    if not isinstance(metadata, dict):
        metadata = {}
    status: Literal["completed", "timed_out", "failed", "limited", "incomplete"]
    if failure is not None:
        status = "failed"
    elif timed_out:
        status = "timed_out"
    elif metadata.get("limited_agents"):
        status = "limited"
    elif unresolved_execution_issues(final_session.state):
        status = "incomplete"
    else:
        status = "completed"
    result = PipelineResult(
        state=dict(final_session.state),
        total_prompt_tokens=usage.prompt,
        total_completion_tokens=usage.completion,
        elapsed=total_elapsed,
        truncated_agents=truncated_agents,
        status=status,
    )
    if failure is not None:
        raise PipelineExecutionError(failure, result) from failure
    return result


async def _consume_with_timeout(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    message: types.Content,
    usage: _TokenUsage,
    truncated_agents: list[str],
    timeout: float | None,
) -> None:
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
        usage=usage,
        truncated_agents=truncated_agents,
    )
    if timeout is None or timeout <= 0:
        await coro
    else:
        await asyncio.wait_for(coro, timeout=timeout)


async def _consume_runner_events(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    message: types.Content,
    usage: _TokenUsage,
    truncated_agents: list[str],
) -> None:
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
            usage.prompt += um.prompt_token_count or 0
            usage.completion += um.candidates_token_count or 0

        # Error handling (control flow — stays in runner)
        if event.error_code:
            if event.error_code == types.FinishReason.MAX_TOKENS:
                # LiteLLM flags any non-STOP finish reason, content or not, so
                # check for content rather than trusting the code.  Thought
                # parts do not count: ADK writes output_key only from
                # non-thought text (llm_agent.py, __handle_output_key).
                parts = (event.content.parts if event.content else None) or []
                has_content = any(p.text and not p.thought for p in parts)
                # One step falling short is not a reason to discard what every
                # earlier step produced: carry on with this output empty, the
                # same way a timeout salvages partial state.
                if has_content:
                    _log.warning(
                        "Agent '{}' hit its token budget; its answer is cut short",
                        event.author,
                    )
                else:
                    _log.warning(
                        "Agent '{}' produced no content within its token budget; "
                        "continuing with an empty result for this step",
                        event.author,
                    )
                    _record_empty_step(truncated_agents, event.author)
                continue

            if event.error_code == _NO_CONTENT_ERROR_CODE:
                # Same salvage as an exhausted budget: seen after a tool result
                # the model could not use (a base64 screenshot as text).
                _log.warning(
                    "Agent '{}' returned an empty response; "
                    "continuing with an empty result for this step",
                    event.author,
                )
                _record_empty_step(truncated_agents, event.author)
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


def _record_empty_step(truncated_agents: list[str], author: str | None) -> None:
    # The field names which steps came up empty, not how often.
    if author and author not in truncated_agents:
        truncated_agents.append(author)
