from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from fedotmas._settings import ModelConfig, resolve_model_config, get_meta_model
from fedotmas.common.logging import get_logger
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.optimize._prompts import JUDGE_SYSTEM_PROMPT

_log = get_logger("fedotmas.optimize._scoring")


@dataclass
class ScoringResult:
    score: float
    feedback: str
    reasoning: str


@runtime_checkable
class Scorer(Protocol):
    async def evaluate(self, task: str, state: dict[str, Any]) -> ScoringResult: ...


class _JudgeOutput(BaseModel):
    score: float = Field(description="Quality score from 0.0 to 1.0")
    reasoning: str = Field(description="Explanation of the score")
    feedback: str = Field(
        description="Actionable feedback for improving the pipeline output"
    )


class LLMJudge:
    def __init__(
        self,
        *,
        criteria: str | None = None,
        model: str | ModelConfig | None = None,
        max_state_chars: int = 2000,
        temperature: float = 0.1,
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

    @property
    def token_usage(self) -> tuple[int, int]:
        return (self._total_prompt_tokens, self._total_completion_tokens)

    async def evaluate(self, task: str, state: dict[str, Any]) -> ScoringResult:
        user_message = (
            f"## Task\n{task}\n\n"
            f"## Pipeline output\n{_format_state(state, self._max_state_chars)}\n\n"
            f"## Evaluation criteria\n{self._criteria}"
        )

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


def _format_state(state: dict[str, Any], max_chars: int = 2000) -> str:
    parts: list[str] = []
    for key, value in state.items():
        text = str(value)
        if len(text) > max_chars:
            _log.debug(
                "State key '{}' truncated: {} -> {} chars",
                key,
                len(text),
                max_chars,
            )
            text = text[:max_chars] + "... (truncated)"
        parts.append(f"### {key}\n{text}")
    return "\n\n".join(parts) if parts else "(empty state)"
