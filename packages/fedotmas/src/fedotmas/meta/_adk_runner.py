from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from fedotmas.backends import get_backend
from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.interfaces.runner import SingleAgentResult
from fedotmas.meta._helpers import validate_allowed_models

_log = get_logger("fedotmas.meta._adk_runner")


# Keep LLMCallResult as a compatibility alias
LLMCallResult = SingleAgentResult


async def run_meta_agent_call(
    *,
    agent_name: str,
    instruction: str,
    user_message: str,
    output_schema: type[BaseModel],
    output_key: str,
    model: ModelConfig,
    temperature: float,
    session_service: Any | None = None,
    max_retries: int = 2,
    allowed_models: list[str] | None = None,
    plugins: list[Any] | None = None,
    backend: str = "adk",
) -> SingleAgentResult:
    """Run a single LLM agent call and return the structured result.

    Used by both single-stage ``generate_pipeline_config`` and the two-stage
    ``PoolGenerator`` / ``PipelineGenerator``.

    Retries up to *max_retries* times on ``RuntimeError`` or
    ``ValidationError`` (e.g. invalid JSON from LLM) with exponential backoff.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    last_error: Exception | None = None
    effective_message = user_message
    for attempt in range(max_retries + 1):
        try:
            be = get_backend(backend)
            runner = be.create_runner(session_service=session_service)
            result = await runner.run_single_agent(
                agent_name=agent_name,
                instruction=instruction,
                user_message=effective_message,
                model=model,
                temperature=temperature,
                output_schema=output_schema,
                output_key=output_key,
                backend_plugins=plugins,
            )

            if result.raw_output is None:
                raise RuntimeError(
                    f"{agent_name} did not produce '{output_key}' in session state"
                )

            if allowed_models:
                validate_allowed_models(result.raw_output, allowed_models)

            return result
        except (RuntimeError, ValueError, TypeError) as e:
            last_error = e
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
                effective_message = (
                    f"{user_message}\n\n"
                    f"PREVIOUS ATTEMPT FAILED: {e}\n"
                    f"Please fix this error in your response."
                )
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
