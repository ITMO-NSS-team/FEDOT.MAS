from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from fedotmas._settings import ModelConfig, get_meta_model, resolve_model_config
from fedotmas.common.logging import get_logger
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.optimize._prompts import (
    JUDGE_SYSTEM_PROMPT,
    SYNTHETIC_EXAMPLES_SYSTEM_PROMPT,
)
from fedotmas.optimize._state import Task

_log = get_logger("fedotmas.optimize._scoring")


@dataclass
class ScoringResult:
    score: float
    feedback: str
    reasoning: str


@runtime_checkable
class Scorer(Protocol):
    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult: ...


class _JudgeOutput(BaseModel):
    score: float = Field(description="Quality score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the score")
    feedback: str = Field(
        description="Actionable feedback for improving the pipeline output"
    )


class _SyntheticExamplesOutput(BaseModel):
    examples: list[str] = Field(
        description="Slightly rephrased requests with unchanged meaning"
    )


class LLMJudge:
    def __init__(
        self,
        *,
        criteria: str | None = None,
        model: str | ModelConfig | None = None,
        max_state_chars: int | None = None,
        temperature: float = 0.1,
        synthetic_temperature: float = 0.7,
    ) -> None:
        self._criteria = criteria or "Overall quality, completeness, and correctness."
        if model is None:
            self._model = resolve_model_config(get_meta_model())
        else:
            self._model = resolve_model_config(model)

        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._max_state_chars = max_state_chars
        self._temperature = temperature
        self._synthetic_temperature = synthetic_temperature

    @property
    def token_usage(self) -> tuple[int, int]:
        return (self._total_prompt_tokens, self._total_completion_tokens)

    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult:
        sections = [
            f"## Task\n{task.input}",
            f"## Pipeline output\n{_format_state(state, self._max_state_chars)}",
        ]
        if task.expected is not None:
            sections.append(f"## Expected answer\n{task.expected}")
        sections.append(f"## Evaluation criteria\n{self._criteria}")
        user_message = "\n\n".join(sections)

        result: LLMCallResult = await run_meta_agent_call(
            agent_name="judge",
            instruction=JUDGE_SYSTEM_PROMPT,
            user_message=user_message,
            output_schema=_JudgeOutput,
            output_key="judge_result",
            model=self._model,
            temperature=self._temperature,
        )

        self._total_prompt_tokens += result.prompt_tokens
        self._total_completion_tokens += result.completion_tokens

        output = _JudgeOutput.model_validate(result.raw_output)
        score = max(0.0, min(1.0, output.score))

        _log.info("Judge | score={:.2f} reasoning={}", score, output.reasoning[:100])

        return ScoringResult(
            score=score,
            feedback=output.feedback,
            reasoning=output.reasoning,
        )

    async def generate_synthetic_examples(
        self, query: str, *, count: int = 1
    ) -> list[str]:
        """Generate meaning-preserving query variants for robustness testing.

        The judge model is reused deliberately: a GUI or an optimization workflow
        needs only one evaluator-model setting. Generated variants do not include
        answers and are not evaluated automatically.
        """
        source = query.strip()
        if not source:
            raise ValueError("query must not be empty")
        if not 1 <= count <= 10:
            raise ValueError("count must be between 1 and 10")

        user_message = json.dumps(
            {"source_request": source, "count": count}, ensure_ascii=False
        )
        result: LLMCallResult = await run_meta_agent_call(
            agent_name="synthetic_examples",
            instruction=SYNTHETIC_EXAMPLES_SYSTEM_PROMPT,
            user_message=user_message,
            output_schema=_SyntheticExamplesOutput,
            output_key="synthetic_examples",
            model=self._model,
            temperature=self._synthetic_temperature,
        )

        self._total_prompt_tokens += result.prompt_tokens
        self._total_completion_tokens += result.completion_tokens

        output = _SyntheticExamplesOutput.model_validate(result.raw_output)
        examples = _unique_examples(output.examples, source)
        if not examples:
            raise ValueError("model returned no distinct synthetic examples")
        return examples[:count]


def _format_state(state: dict[str, Any], max_chars: int | None = None) -> str:
    parts: list[str] = []
    for key, value in state.items():
        text = str(value)
        if max_chars is not None and len(text) > max_chars:
            _log.debug(
                "State key '{}' truncated: {} -> {} chars",
                key,
                len(text),
                max_chars,
            )
            text = text[:max_chars] + "... (truncated)"
        parts.append(f"### {key}\n{text}")
    return "\n\n".join(parts) if parts else "(empty state)"


def _unique_examples(examples: list[str], source: str) -> list[str]:
    """Remove empty, unchanged, and duplicate variants without altering wording."""
    source_key = " ".join(source.split()).casefold()
    seen = {source_key}
    result: list[str] = []
    for value in examples:
        text = value.strip()
        key = " ".join(text.split()).casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
