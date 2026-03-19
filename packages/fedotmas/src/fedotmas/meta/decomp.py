from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.common.llm import make_llm

_log = get_logger("fedotmas.meta.decomposition_agent")

AGENT_NAME = "decomposition_agent"
OUTPUT_KEY = "result"

DECOMPOSITION_PROMPT = """You are a task decomposition expert for multi-agent systems benchmarks.

Given a complex benchmark question, break it down into clear, atomic subtasks that can be solved independently or sequentially.

Rules:
- Each subtask must be specific and actionable
- Subtasks should cover all aspects needed to answer the original question
- Keep subtasks concise (1-2 sentences each)
- Order subtasks logically (dependencies first)
- Return ONLY valid JSON in the format: {"subtasks": [...]}
"""


# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────
class SubtasksOutput(BaseModel):
    subtasks: list[str]


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────
@dataclass
class DecompositionResult:
    """Result of a decomposition agent call."""

    subtasks: list[str]
    prompt_tokens: int
    completion_tokens: int
    elapsed: float


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────
async def decompose_question(
    question: str,
    model: ModelConfig,
    temperature: float = 0.0,
    session_service: BaseSessionService | None = None,
    max_retries: int = 2,
    allowed_models: list[str] | None = None,
) -> DecompositionResult:
    """Decompose a benchmark question into subtasks.

    Retries up to *max_retries* times on RuntimeError / ValueError
    with exponential backoff.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await _execute_decomposition(
                question=question,
                model=model,
                temperature=temperature,
                session_service=session_service,
                allowed_models=allowed_models,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** attempt
                _log.warning(
                    "{} attempt {}/{} failed: {}, retrying in {}s...",
                    AGENT_NAME,
                    attempt + 1,
                    max_retries + 1,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                _log.error(
                    "{} failed after {} attempts: {}",
                    AGENT_NAME,
                    max_retries + 1,
                    e,
                )

    if last_error is None:
        raise RuntimeError(f"{AGENT_NAME}: retry loop exited without result or error")
    raise last_error


# ─────────────────────────────────────────────
# CORE EXECUTION
# ─────────────────────────────────────────────
async def _execute_decomposition(
    *,
    question: str,
    model: ModelConfig,
    temperature: float,
    session_service: BaseSessionService | None = None,
    allowed_models: list[str] | None = None,
) -> DecompositionResult:
    _log.info(
        "{} | model={} temperature={}",
        AGENT_NAME,
        model.model,
        temperature,
    )

    llm = make_llm(model)


    agent = LlmAgent(
        name=AGENT_NAME,
        model=llm,
        instruction=DECOMPOSITION_PROMPT,
        output_schema=SubtasksOutput,
        output_key=OUTPUT_KEY,
        generate_content_config=types.GenerateContentConfig(
            temperature=temperature,
        ),

    )

    session_service = session_service or InMemorySessionService()
    session_id = uuid.uuid4().hex
    app_name = f"fedotmas_{AGENT_NAME}"

    session = await session_service.create_session(
        app_name=app_name,
        user_id="system",
        session_id=session_id,
        state={},
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=question)],
    )

    total_prompt = 0
    total_completion = 0
    start = time.monotonic()

    async with Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    ) as runner:
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
                    AGENT_NAME,
                    event.error_code,
                    event.error_message,
                )
                raise RuntimeError(
                    f"{AGENT_NAME} LLM error {event.error_code}: {event.error_message}"
                )

    elapsed = time.monotonic() - start
    _log.info(
        "{} complete | elapsed={:.1f}s prompt={} completion={}",
        AGENT_NAME,
        elapsed,
        total_prompt,
        total_completion,
    )

    # Retrieve structured output from session state
    final_session = await session_service.get_session(
        app_name=app_name,
        user_id="system",
        session_id=session.id,
    )
    if final_session is None:
        raise RuntimeError(
            f"{AGENT_NAME}: session lost after execution — results unavailable"
        )

    raw_output = final_session.state.get(OUTPUT_KEY)
    _log.debug(
        "Raw output | key={} type={} preview={}",
        OUTPUT_KEY,
        type(raw_output).__name__,
        str(raw_output)[:500],
    )
    if raw_output is None:
        raise RuntimeError(
            f"{AGENT_NAME} did not produce '{OUTPUT_KEY}' in session state"
        )

    # raw_output is either a dict or already a SubtasksOutput instance
    if isinstance(raw_output, SubtasksOutput):
        subtasks = raw_output.subtasks
    elif isinstance(raw_output, dict):
        subtasks = SubtasksOutput(**raw_output).subtasks
    else:
        raise RuntimeError(
            f"{AGENT_NAME}: unexpected output type {type(raw_output)}"
        )

    _log.info("{} | subtasks_count={}", AGENT_NAME, len(subtasks))

    return DecompositionResult(
        subtasks=subtasks,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        elapsed=elapsed,
    )


# ─────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────
'''
async def main() -> None:
    question = (
        "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgarispatients from Jan-May 2018 as listed on the NIH website?: What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?"
    )

    model = ModelConfig(model="openai/gpt-oss-120b:free")
    

    result = await decompose_question(
        question=question,
        model=model,
        temperature=0.0,
        max_retries=2,
    )

    print(f"Subtasks ({len(result.subtasks)}):")
    for i, subtask in enumerate(result.subtasks, 1):
        print(f"  {i}. {subtask}")
    print(f"\nTokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")
    print(f"Elapsed: {result.elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())'''