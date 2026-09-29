from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from fedotmas._settings import ModelConfig
from fedotmas.common.llm import make_llm
from fedotmas.common.logging import get_logger
from fedotmas.meta._helpers import validate_allowed_models

_log = get_logger("fedotmas.meta._adk_runner")


@dataclass
class LLMCallResult:
    """Result of a single ADK LlmAgent call."""

    raw_output: Any
    prompt_tokens: int
    completion_tokens: int
    elapsed: float


class _MetaUsageTracker(BasePlugin):
    """Collect provider usage before ADK validates structured responses."""

    def __init__(self, usage_totals: dict[str, int | float]) -> None:
        super().__init__("fedotmas_meta_usage")
        self._usage_totals = usage_totals

    async def after_model_callback(self, *, callback_context, llm_response):
        finish_reason = str(getattr(llm_response, "finish_reason", "")).lower()
        if "max_tokens" in finish_reason or "length" in finish_reason:
            self._usage_totals["truncation_warning"] = 1
            _log.warning(
                "Meta-agent output was truncated at the configured output-token "
                "limit; consider increasing FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS."
            )
        usage = getattr(llm_response, "usage_metadata", None)
        if usage is not None:
            self._usage_totals["prompt"] = self._usage_totals.get("prompt", 0) + (
                usage.prompt_token_count or 0
            )
            self._usage_totals["completion"] = self._usage_totals.get(
                "completion", 0
            ) + (usage.candidates_token_count or 0)

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        self._usage_totals["prompt"] = self._usage_totals.get("prompt", 0) + int(
            getattr(error, "prompt_tokens", 0)
        )
        self._usage_totals["completion"] = self._usage_totals.get(
            "completion", 0
        ) + int(getattr(error, "completion_tokens", 0))


async def run_meta_agent_call(
    *,
    agent_name: str,
    instruction: str,
    user_message: str,
    output_schema: type[BaseModel],
    output_key: str,
    model: ModelConfig,
    temperature: float,
    session_service: BaseSessionService | None = None,
    max_retries: int = 2,
    allowed_models: list[str] | None = None,
    plugins: list[BasePlugin] | None = None,
    timeout_s: float | None = None,
) -> LLMCallResult:
    """Run a single ADK LlmAgent call and return the structured result.

    Used by both single-stage ``generate_pipeline_config`` and the two-stage
    ``PoolGenerator`` / ``PipelineGenerator``.

    Retries up to *max_retries* times on ``RuntimeError`` or
    ``ValidationError`` (e.g. invalid JSON from LLM) with exponential backoff.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    last_error: Exception | None = None
    effective_message = user_message
    failed_prompt_tokens = 0
    failed_completion_tokens = 0
    failed_elapsed = 0.0
    effective_timeout_s = _resolve_timeout(timeout_s)
    max_output_tokens = _resolve_max_output_tokens()
    for attempt in range(max_retries + 1):
        attempt_usage: dict[str, int | float] = {
            "prompt": 0,
            "completion": 0,
            "elapsed": 0.0,
        }
        try:
            call = _execute_meta_call(
                agent_name=agent_name,
                instruction=instruction,
                user_message=effective_message,
                output_schema=output_schema,
                output_key=output_key,
                model=model,
                temperature=temperature,
                session_service=session_service,
                allowed_models=allowed_models,
                plugins=plugins,
                max_output_tokens=max_output_tokens,
                usage_totals=attempt_usage,
            )
            if effective_timeout_s is None:
                result = await call
            else:
                async with asyncio.timeout(effective_timeout_s):
                    result = await call
            return LLMCallResult(
                raw_output=result.raw_output,
                prompt_tokens=failed_prompt_tokens + result.prompt_tokens,
                completion_tokens=failed_completion_tokens + result.completion_tokens,
                elapsed=failed_elapsed + result.elapsed,
            )
        except (RuntimeError, ValueError, TypeError, TimeoutError) as e:
            last_error = e
            failed_prompt_tokens += int(attempt_usage.get("prompt", 0))
            failed_completion_tokens += int(attempt_usage.get("completion", 0))
            failed_elapsed += float(attempt_usage.get("elapsed", 0.0))
            # Retain usage when all structured-output attempts fail so the
            # caller can include billed generations in its run diagnostics.
            error_with_usage = cast(Any, e)
            error_with_usage.prompt_tokens = failed_prompt_tokens
            error_with_usage.completion_tokens = failed_completion_tokens
            error_with_usage.elapsed = failed_elapsed
            if isinstance(e, TimeoutError):
                _log.error(
                    "{} timed out after {:.1f}s; not retrying the same prompt",
                    agent_name,
                    effective_timeout_s or 0.0,
                )
                raise
            if _is_output_truncation(e) and not attempt_usage.get("truncation_warning"):
                _log.warning(
                    "Meta-agent output was truncated at the configured output-token "
                    "limit; consider increasing FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS."
                )
            if attempt < max_retries:
                delay = 2**attempt
                _log.warning(
                    "{} attempt {}/{} failed: {}, retrying in {}s...",
                    agent_name,
                    attempt + 1,
                    max_retries + 1,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                effective_message = _retry_message(user_message, e)
            else:
                _log.error(
                    "{} failed after {} attempts: {}",
                    agent_name,
                    max_retries + 1,
                    e,
                )
    if last_error is None:
        raise RuntimeError(f"{agent_name}: retry loop exited without result or error")
    raise last_error


async def _execute_meta_call(
    *,
    agent_name: str,
    instruction: str,
    user_message: str,
    output_schema: type[BaseModel],
    output_key: str,
    model: ModelConfig,
    temperature: float,
    session_service: BaseSessionService | None = None,
    allowed_models: list[str] | None = None,
    plugins: list[BasePlugin] | None = None,
    max_output_tokens: int | None = None,
    usage_totals: dict[str, int | float] | None = None,
) -> LLMCallResult:
    """Core execution logic for a single meta-agent LLM call."""
    _log.info(
        "{} | model={} temperature={}",
        agent_name,
        model.model,
        temperature,
    )

    llm = make_llm(model)

    agent = LlmAgent(
        name=agent_name,
        model=llm,
        instruction=instruction,
        output_schema=output_schema,
        output_key=output_key,
        generate_content_config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=False,
            )
        ),
    )

    session_service = session_service or InMemorySessionService()
    session_id = uuid.uuid4().hex
    app_name = f"fedotmas_{agent_name}"

    session = await session_service.create_session(
        app_name=app_name,
        user_id="system",
        session_id=session_id,
        state={},
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    total_prompt = 0
    total_completion = 0
    start = time.monotonic()
    usage_tracker = (
        _MetaUsageTracker(usage_totals) if usage_totals is not None else None
    )

    runner = Runner(
        app_name=app_name,
        agent=agent,
        plugins=[*([usage_tracker] if usage_tracker else []), *(plugins or [])],
        session_service=session_service,
    )

    await runner.__aenter__()
    primary_error: BaseException | None = None
    suppressed = False
    try:
        async for event in runner.run_async(
            user_id="system",
            session_id=session.id,
            new_message=message,
        ):
            if event.partial:
                continue

            if event.usage_metadata:
                um = event.usage_metadata
                prompt = um.prompt_token_count or 0
                completion = um.candidates_token_count or 0
                total_prompt += prompt
                total_completion += completion
                if prompt or completion:
                    _log.info("Tokens | prompt={} completion={}", prompt, completion)

            if event.content and event.content.parts:
                texts = [p.text for p in event.content.parts if p.text]
                if texts:
                    _log.debug("Response preview | text={}", texts[0][:200])

            if event.error_code:
                _log.error(
                    "LLM error | agent={} code={} msg={}",
                    agent_name,
                    event.error_code,
                    event.error_message,
                )
                raise RuntimeError(
                    f"{agent_name} LLM error {event.error_code}: {event.error_message}"
                )
    except BaseException as exc:  # noqa: BLE001 - preserve primary errors on cleanup
        primary_error = exc
    finally:
        try:
            suppressed = await runner.__aexit__(
                type(primary_error) if primary_error else None,
                primary_error,
                primary_error.__traceback__ if primary_error else None,
            )
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            _log.warning(
                "Meta-agent runner cleanup failed after primary error: {}",
                cleanup_error,
            )
    if primary_error is not None and not suppressed:
        if usage_totals is not None and not (
            usage_totals["prompt"] or usage_totals["completion"]
        ):
            usage_totals["prompt"] = total_prompt
            usage_totals["completion"] = total_completion
        if usage_totals is not None:
            usage_totals["elapsed"] = time.monotonic() - start
        raise primary_error

    elapsed = time.monotonic() - start
    if usage_totals is not None:
        if not (usage_totals["prompt"] or usage_totals["completion"]):
            usage_totals["prompt"] = total_prompt
            usage_totals["completion"] = total_completion
        usage_totals["elapsed"] = elapsed
    _log.info(
        "{} complete | elapsed={:.1f}s prompt={} completion={}",
        agent_name,
        elapsed,
        total_prompt,
        total_completion,
    )

    # Retrieve the structured output from session state.
    final_session = await session_service.get_session(
        app_name=app_name,
        user_id="system",
        session_id=session.id,
    )
    if final_session is None:
        raise RuntimeError(
            f"{agent_name}: session lost after execution — results unavailable"
        )

    raw_output = final_session.state.get(output_key)
    _log.debug(
        "Raw output | key={} type={} preview={}",
        output_key,
        type(raw_output).__name__,
        str(raw_output)[:500],
    )
    if raw_output is None:
        raise RuntimeError(
            f"{agent_name} did not produce '{output_key}' in session state"
        )

    if allowed_models:
        validate_allowed_models(raw_output, allowed_models)

    tracked_prompt = int(usage_totals.get("prompt", 0)) if usage_totals else 0
    tracked_completion = int(usage_totals.get("completion", 0)) if usage_totals else 0
    return LLMCallResult(
        raw_output=raw_output,
        prompt_tokens=tracked_prompt or total_prompt,
        completion_tokens=tracked_completion or total_completion,
        elapsed=elapsed,
    )


def _retry_message(user_message: str, error: Exception) -> str:
    details = str(error)
    if _is_output_truncation(error):
        feedback = (
            "The previous structured response was truncated by the output-token "
            "limit. Regenerate the complete valid configuration concisely. If this "
            "continues, increase FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS.\n"
            f"Provider error: {details}"
        )
        return f"{user_message}\n\n{feedback}"
    if (
        "validationerror" in details.lower()
        or "references unknown agent" in details.lower()
    ):
        feedback = (
            "The previous structured configuration was invalid:\n"
            f"{details}\n"
            "Regenerate the complete config and correct every validation error. "
            "For pipeline references, use agent_name values exactly as declared "
            "in the agents list; do not add or alter agent names."
        )
    else:
        feedback = f"PREVIOUS ATTEMPT FAILED: {details}\nPlease fix this error in your response."
    return f"{user_message}\n\n{feedback}"


def _is_output_truncation(error: BaseException) -> bool:
    details = str(error).lower()
    return (
        "max_tokens" in details
        or "finish_reason=length" in details
        or "finish_reason='length'" in details
    )


def _resolve_timeout(timeout_s: float | None) -> float | None:
    if timeout_s is not None:
        return timeout_s if timeout_s > 0 else None

    value = os.getenv("FEDOTMAS_META_AGENT_TIMEOUT_S", "180")
    try:
        resolved = float(value)
    except ValueError:
        _log.warning("Invalid FEDOTMAS_META_AGENT_TIMEOUT_S={!r}; using 180", value)
        resolved = 180.0
    return resolved if resolved > 0 else None


def _resolve_max_output_tokens() -> int | None:
    value = os.getenv("FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS", "8192")
    try:
        resolved = int(value)
    except ValueError:
        _log.warning(
            "Invalid FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS={!r}; using 8192",
            value,
        )
        resolved = 8192
    return resolved if resolved > 0 else None
