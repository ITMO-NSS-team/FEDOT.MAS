from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from fedotmas._settings import ModelConfig, resolve_model_config, get_meta_model
from fedotmas.common.logging import get_logger
from fedotmas.meta._adk_runner import run_meta_agent_call
from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from fedotmas.optimize._prompts import REFLECTION_SYSTEM_PROMPT, MERGE_SYSTEM_PROMPT
from fedotmas.optimize._state import Candidate

_log = get_logger("fedotmas.optimize._proposer")


@dataclass
class ReflectionExample:
    task: str
    agent_instruction: str
    agent_output: str | None
    pipeline_output: dict[str, Any]
    score: float
    feedback: str


class _ReflectionOutput(BaseModel):
    improved_instruction: str = Field(
        description="The improved instruction for the agent"
    )


class _MergeOutput(BaseModel):
    merged_instruction: str = Field(
        description="The merged instruction combining the best of both"
    )


class Proposer:
    def __init__(
        self,
        *,
        model: str | ModelConfig | None = None,
        max_output_chars: int = 3000,
    ) -> None:
        if model is None:
            self._model = resolve_model_config(get_meta_model())
        else:
            self._model = resolve_model_config(model)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._max_output_chars = max_output_chars

    @property
    def token_usage(self) -> tuple[int, int]:
        return (self.total_prompt_tokens, self.total_completion_tokens)

    async def propose_mutation(
        self,
        candidate: Candidate,
        agent_names: list[str],
        tasks: list[str],
    ) -> MAWConfig:
        config = candidate.config

        for agent_name in agent_names:
            agent = _find_agent(config, agent_name)
            if agent is None:
                continue

            examples = _build_reflection_examples(candidate, agent, tasks)
            if not examples:
                continue

            new_instruction = await self._reflect(
                agent_name, agent.instruction, examples, self._max_output_chars
            )
            if new_instruction and new_instruction != agent.instruction:
                new_agent = MAWAgentConfig(
                    name=agent.name,
                    instruction=new_instruction,
                    model=agent.model,
                    output_key=agent.output_key,
                    tools=list(agent.tools),
                )
                config = config.replace_agent(agent_name, new_agent)

        return config

    async def propose_merge(
        self,
        candidate_a: Candidate,
        candidate_b: Candidate,
        tasks: list[str],
    ) -> MAWConfig:
        config_a = candidate_a.config
        config_b = candidate_b.config

        result_config = config_a

        seen: set[str] = set()
        all_names: list[str] = []
        for a in config_a.agents:
            if a.name not in seen:
                all_names.append(a.name)
                seen.add(a.name)
        for a in config_b.agents:
            if a.name not in seen:
                all_names.append(a.name)
                seen.add(a.name)

        for name in all_names:
            agent_a = _find_agent(config_a, name)
            agent_b = _find_agent(config_b, name)

            if agent_a is not None and agent_b is not None:
                if agent_a.instruction == agent_b.instruction:
                    continue

                task_context = "\n".join(f"- {t}" for t in tasks[:5])
                merged = await self._merge(
                    name,
                    agent_a.instruction,
                    agent_b.instruction,
                    task_context,
                )
                if merged:
                    new_agent = MAWAgentConfig(
                        name=agent_a.name,
                        instruction=merged,
                        model=agent_a.model,
                        output_key=agent_a.output_key,
                        tools=list(agent_a.tools),
                    )
                    result_config = result_config.replace_agent(name, new_agent)

            elif agent_b is not None and agent_a is None:
                result_config = result_config.insert_after(
                    result_config.agents[-1].name, agent_b
                )

        return result_config

    async def _reflect(
        self,
        agent_name: str,
        current_instruction: str,
        examples: list[ReflectionExample],
        max_output_chars: int = 3000,
    ) -> str | None:
        examples_text = _format_reflection_examples(examples, max_output_chars)
        user_message = (
            f"## Agent: {agent_name}\n\n"
            f"## Current instruction\n{current_instruction}\n\n"
            f"## Evaluation examples\n{examples_text}"
        )

        try:
            result = await run_meta_agent_call(
                agent_name=f"reflector_{agent_name}",
                instruction=REFLECTION_SYSTEM_PROMPT,
                user_message=user_message,
                output_schema=_ReflectionOutput,
                output_key="reflection_result",
                model=self._model,
                temperature=0.7,
            )
            self.total_prompt_tokens += result.prompt_tokens
            self.total_completion_tokens += result.completion_tokens

            output = _ReflectionOutput.model_validate(result.raw_output)
            _log.info(
                "Reflection | agent={} instruction_len={}→{}",
                agent_name,
                len(current_instruction),
                len(output.improved_instruction),
            )
            return output.improved_instruction
        except Exception as e:
            _log.warning("Reflection failed for {}: {}", agent_name, e)
            return None

    async def _merge(
        self,
        agent_name: str,
        instruction_a: str,
        instruction_b: str,
        task_context: str,
    ) -> str | None:
        user_message = (
            f"## Agent: {agent_name}\n\n"
            f"## Instruction A\n{instruction_a}\n\n"
            f"## Instruction B\n{instruction_b}\n\n"
            f"## Task context\n{task_context}"
        )

        try:
            result = await run_meta_agent_call(
                agent_name=f"merger_{agent_name}",
                instruction=MERGE_SYSTEM_PROMPT,
                user_message=user_message,
                output_schema=_MergeOutput,
                output_key="merge_result",
                model=self._model,
                temperature=0.5,
            )
            self.total_prompt_tokens += result.prompt_tokens
            self.total_completion_tokens += result.completion_tokens

            output = _MergeOutput.model_validate(result.raw_output)
            _log.info(
                "Merge | agent={} merged_len={}",
                agent_name,
                len(output.merged_instruction),
            )
            return output.merged_instruction
        except Exception as e:
            _log.warning("Merge failed for {}: {}", agent_name, e)
            return None


def _find_agent(config: MAWConfig, name: str) -> MAWAgentConfig | None:
    for a in config.agents:
        if a.name == name:
            return a
    return None


def _build_reflection_examples(
    candidate: Candidate, agent: MAWAgentConfig, tasks: list[str]
) -> list[ReflectionExample]:
    examples: list[ReflectionExample] = []
    for task in tasks:
        if task not in candidate.scores:
            continue
        state = candidate.states.get(task, {})
        agent_output = state.get(agent.output_key)
        examples.append(
            ReflectionExample(
                task=task,
                agent_instruction=agent.instruction,
                agent_output=str(agent_output) if agent_output is not None else None,
                pipeline_output=state,
                score=candidate.scores[task],
                feedback=candidate.feedbacks.get(task, ""),
            )
        )
    return examples


def _format_reflection_examples(
    examples: list[ReflectionExample], max_output_chars: int = 3000
) -> str:
    parts: list[str] = []
    for i, ex in enumerate(examples, 1):
        agent_out = ex.agent_output or "(no output)"
        if len(agent_out) > max_output_chars:
            agent_out = agent_out[:max_output_chars] + "... (truncated)"
        parts.append(
            f"### Example {i}\n"
            f"**Task:** {ex.task}\n"
            f"**Agent output:** {agent_out}\n"
            f"**Score:** {ex.score:.2f}\n"
            f"**Feedback:** {ex.feedback}"
        )
    return "\n\n".join(parts)
