from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from fedotmas._settings import ModelConfig, resolve_model_config, get_meta_model
from fedotmas.common.logging import get_logger
from fedotmas.meta._adk_runner import run_meta_agent_call
from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from fedotmas.optimize._config import OptimizationConfig
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


def _with_instruction(agent: MAWAgentConfig, instruction: str) -> MAWAgentConfig:
    return MAWAgentConfig(
        name=agent.name,
        instruction=instruction,
        model=agent.model,
        output_key=agent.output_key,
        tools=list(agent.tools),
    )


class Proposer:
    def __init__(
        self,
        config: OptimizationConfig | None = None,
        *,
        model: str | ModelConfig | None = None,
    ) -> None:
        if model is None:
            self._model = resolve_model_config(get_meta_model())
        else:
            self._model = resolve_model_config(model)

        cfg = config or OptimizationConfig()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._max_output_chars = cfg.max_output_chars
        self._temperature_reflect = cfg.temperature_reflect
        self._temperature_merge = cfg.temperature_merge
        self._max_merge_context_tasks = cfg.max_merge_context_tasks
        self._llm_timeout = cfg.llm_timeout

    @property
    def token_usage(self) -> tuple[int, int]:
        return (self._total_prompt_tokens, self._total_completion_tokens)

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
                config = config.replace_agent(
                    agent_name, _with_instruction(agent, new_instruction)
                )

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
        all_names = _unique_agent_names(config_a, config_b)

        for name in all_names:
            agent_a = _find_agent(config_a, name)
            agent_b = _find_agent(config_b, name)

            if agent_a is not None and agent_b is not None:
                if agent_a.instruction == agent_b.instruction:
                    continue

                task_context = "\n".join(
                    f"- {t}" for t in tasks[: self._max_merge_context_tasks]
                )
                merged = await self._merge(
                    name,
                    agent_a.instruction,
                    agent_b.instruction,
                    task_context,
                )
                if merged:
                    result_config = result_config.replace_agent(
                        name, _with_instruction(agent_a, merged)
                    )

            elif agent_b is not None and agent_a is None:
                result_config = result_config.insert_after(
                    result_config.agents[-1].name, agent_b
                )

        return result_config

    async def propose_genealogy_merge(
        self,
        ancestor: Candidate,
        child_a: Candidate,
        child_b: Candidate,
        tasks: list[str],
    ) -> MAWConfig:
        """Component-level crossover using common ancestor.

        For each agent:
        - If only one child changed the instruction -> take that child's version.
        - If both changed -> fallback to LLM merge for that agent.
        - If neither changed -> keep ancestor's version.
        """
        config_anc = ancestor.config
        config_a = child_a.config
        config_b = child_b.config
        result_config = config_a
        all_names = _unique_agent_names(config_a, config_b, config_anc)

        for name in all_names:
            agent_anc = _find_agent(config_anc, name)
            agent_a = _find_agent(config_a, name)
            agent_b = _find_agent(config_b, name)

            if agent_a is None and agent_b is not None:
                result_config = result_config.insert_after(
                    result_config.agents[-1].name, agent_b
                )
                continue

            if agent_a is None:
                continue

            if agent_anc is None:
                continue

            if agent_b is None:
                continue

            instr_anc = agent_anc.instruction
            a_changed = agent_a.instruction != instr_anc
            b_changed = agent_b.instruction != instr_anc

            if not a_changed and not b_changed:
                continue
            elif a_changed and not b_changed:
                chosen = agent_a
            elif b_changed and not a_changed:
                chosen = agent_b
            else:
                merged = await self._merge(
                    name,
                    agent_a.instruction,
                    agent_b.instruction,
                    "\n".join(f"- {t}" for t in tasks[: self._max_merge_context_tasks]),
                )
                if merged:
                    chosen = _with_instruction(agent_a, merged)
                else:
                    score_a = child_a.mean_score or 0.0
                    score_b = child_b.mean_score or 0.0
                    chosen = agent_a if score_a >= score_b else agent_b

            if chosen.instruction != agent_a.instruction:
                result_config = result_config.replace_agent(
                    name, _with_instruction(agent_a, chosen.instruction)
                )

        return result_config

    async def propose_tool_mutation(
        self,
        candidate: Candidate,
        agent_name: str,
    ) -> MAWConfig:
        raise NotImplementedError(
            "Tool optimization: add/remove/replace tools per agent"
        )

    async def propose_model_mutation(
        self,
        candidate: Candidate,
        agent_name: str,
    ) -> MAWConfig:
        raise NotImplementedError(
            "Model selection optimization: choose optimal model per agent"
        )

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
            coro = run_meta_agent_call(
                agent_name=f"reflector_{agent_name}",
                instruction=REFLECTION_SYSTEM_PROMPT,
                user_message=user_message,
                output_schema=_ReflectionOutput,
                output_key="reflection_result",
                model=self._model,
                temperature=self._temperature_reflect,
            )
            if self._llm_timeout > 0:
                async with asyncio.timeout(self._llm_timeout):
                    result = await coro
            else:
                result = await coro
            self._total_prompt_tokens += result.prompt_tokens
            self._total_completion_tokens += result.completion_tokens

            output = _ReflectionOutput.model_validate(result.raw_output)
            _log.info(
                "Reflection | agent={} instruction_len={}->{}",
                agent_name,
                len(current_instruction),
                len(output.improved_instruction),
            )
            return output.improved_instruction
        except TimeoutError:
            _log.warning("Reflection timed out for {}", agent_name)
            return None
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
            coro = run_meta_agent_call(
                agent_name=f"merger_{agent_name}",
                instruction=MERGE_SYSTEM_PROMPT,
                user_message=user_message,
                output_schema=_MergeOutput,
                output_key="merge_result",
                model=self._model,
                temperature=self._temperature_merge,
            )
            if self._llm_timeout > 0:
                async with asyncio.timeout(self._llm_timeout):
                    result = await coro
            else:
                result = await coro
            self._total_prompt_tokens += result.prompt_tokens
            self._total_completion_tokens += result.completion_tokens

            output = _MergeOutput.model_validate(result.raw_output)
            _log.info(
                "Merge | agent={} merged_len={}",
                agent_name,
                len(output.merged_instruction),
            )
            return output.merged_instruction
        except TimeoutError:
            _log.warning("Merge timed out for {}", agent_name)
            return None
        except Exception as e:
            _log.warning("Merge failed for {}: {}", agent_name, e)
            return None


def _unique_agent_names(*configs: MAWConfig) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for cfg in configs:
        for a in cfg.agents:
            if a.name not in seen:
                names.append(a.name)
                seen.add(a.name)
    return names


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

        pipeline_ctx = _format_pipeline_context(ex.pipeline_output, max_output_chars)

        parts.append(
            f"### Example {i}\n"
            f"**Task:** {ex.task}\n"
            f"**Agent output:** {agent_out}\n"
            f"**Pipeline context (other agents):**\n{pipeline_ctx}\n"
            f"**Score:** {ex.score:.2f}\n"
            f"**Feedback:** {ex.feedback}"
        )
    return "\n\n".join(parts)


def _format_pipeline_context(state: dict[str, Any], max_chars: int = 3000) -> str:
    if not state:
        return "(no pipeline context)"
    parts: list[str] = []
    budget = max_chars
    for key, value in state.items():
        text = str(value)
        if len(text) > budget // max(len(state), 1):
            text = text[: budget // max(len(state), 1)] + "... (truncated)"
        parts.append(f"- **{key}:** {text}")
        budget -= len(text)
        if budget <= 0:
            break
    return "\n".join(parts)
