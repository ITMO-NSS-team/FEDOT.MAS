from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import ModelConfig
from fedotmas.meta.schema_utils import needs_strict_schema, patch_schema_openai_strict

_log = get_logger("fedotmas.meta._adk_runner")


@dataclass
class LLMCallResult:
    """Result of a single ADK LlmAgent call."""

    raw_output: Any
    prompt_tokens: int
    completion_tokens: int
    elapsed: float


def _fix_schema_callback(*, callback_context: Any, llm_request: Any) -> None:  # noqa: ARG001
    """Patch response_schema for OpenAI-compatible models (strict mode)."""
    try:
        model = llm_request.model or ""
        schema = llm_request.config and llm_request.config.response_schema
        if not schema or not needs_strict_schema(model):
            return None

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = schema.model_json_schema()
        elif isinstance(schema, BaseModel):
            schema = schema.__class__.model_json_schema()
        elif not isinstance(schema, dict):
            return None

        llm_request.config.response_schema = patch_schema_openai_strict(schema)
    except Exception as e:
        _log.error("Schema patching failed, proceeding without patch: {}", e)
    return None


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
) -> LLMCallResult:
    """Run a single ADK LlmAgent call and return the structured result.

    Used by both single-stage ``generate_pipeline_config`` and the two-stage
    ``PoolGenerator`` / ``PipelineGenerator``.

    Retries up to *max_retries* times on ``RuntimeError`` or
    ``ValidationError`` (e.g. invalid JSON from LLM) with exponential backoff.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await _execute_meta_call(
                agent_name=agent_name,
                instruction=instruction,
                user_message=user_message,
                output_schema=output_schema,
                output_key=output_key,
                model=model,
                temperature=temperature,
                session_service=session_service,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            last_error = e
            if attempt < max_retries:
                delay = 2**attempt
                _log.warning(
                    "{} attempt {}/{} failed: {}, retrying in {}s...",
                    agent_name, attempt + 1, max_retries + 1, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                _log.error(
                    "{} failed after {} attempts: {}",
                    agent_name, max_retries + 1, e,
                )
    raise last_error  # type: ignore[misc]


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
) -> LLMCallResult:
    """Core execution logic for a single meta-agent LLM call."""
    _log.info(
        "{} | model={} temperature={}",
        agent_name,
        model.model,
        temperature,
    )

    llm_kwargs: dict[str, Any] = {}
    if model.api_base:
        llm_kwargs["api_base"] = model.api_base
    if model.api_key:
        llm_kwargs["api_key"] = model.api_key

    agent = LlmAgent(
        name=agent_name,
        model=LiteLlm(model=model.model, **llm_kwargs),
        instruction=instruction,
        output_schema=output_schema,
        output_key=output_key,
        generate_content_config=types.GenerateContentConfig(
            temperature=temperature,
        ),
        before_model_callback=_fix_schema_callback,  # type: ignore[arg-type]
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

    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    total_prompt = 0
    total_completion = 0
    start = time.monotonic()

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
            raise RuntimeError(
                f"{agent_name} LLM error {event.error_code}: {event.error_message}"
            )

    elapsed = time.monotonic() - start
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

    return LLMCallResult(
        raw_output=raw_output,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        elapsed=elapsed,
    )
