from __future__ import annotations

import json
from typing import Any

from google.adk.sessions import BaseSessionService
from pydantic import BaseModel

from fedotmas._settings import ModelConfig
from fedotmas.common.logging import get_logger
from fedotmas.control._run import RunError
from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas.meta._adk_runner import run_meta_agent_call
from fedotmas.meta._helpers import (
    format_server_descriptions,
    parse_llm_output,
    resolve_meta_and_workers,
)
from fedotmas.meta.maw_debug_prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    DEBUGGER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
)

_log = get_logger("fedotmas.meta.maw_debugger")


class ErrorClassification(BaseModel):
    """Result of LLM-based error classification."""

    retryable: bool
    category: str
    reasoning: str


class OutputEvaluation(BaseModel):
    """Result of LLM-based output evaluation against error_hint."""

    passed: bool
    agent_name: str
    reasoning: str


_MAX_ERROR_MESSAGE_LEN = 2000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"


async def classify_error(
    *,
    error: RunError,
    config: MAWConfig | None = None,
    error_hint: str | None = None,
    meta_model: str | ModelConfig | None = None,
    session_service: BaseSessionService | None = None,
) -> ErrorClassification:
    """Use a lightweight LLM call to classify an error as retryable or fatal."""
    resolved_meta, _, _ = resolve_meta_and_workers(
        meta_model,
        None,
        None,
    )

    agent_config_json = "{}"
    if config is not None:
        failing = next((a for a in config.agents if a.name == error.agent_name), None)
        if failing is not None:
            agent_config_json = _truncate(failing.model_dump_json(indent=2), 2000)

    error_hint_section = ""
    if error_hint:
        error_hint_section = f"\n**User hint:** {error_hint}\n"

    error_message = _truncate(error.message, _MAX_ERROR_MESSAGE_LEN)

    instruction = CLASSIFIER_SYSTEM_PROMPT.substitute(
        agent_name=error.agent_name,
        error_message=error_message,
        agent_config_json=agent_config_json,
        error_hint_section=error_hint_section,
    )

    result = await run_meta_agent_call(
        agent_name="error_classifier",
        instruction=instruction,
        user_message=f"Classify this error from agent '{error.agent_name}': {error_message}",
        output_schema=ErrorClassification,
        output_key="error_classification",
        model=resolved_meta,
        temperature=0.1,
        session_service=session_service,
        max_retries=1,
    )

    classification = parse_llm_output(result.raw_output, ErrorClassification)
    _log.info(
        "Error classified | retryable={} category={} reasoning={}",
        classification.retryable,
        classification.category,
        classification.reasoning,
    )
    return classification


async def evaluate_output(
    *,
    state: dict[str, Any],
    config: MAWConfig,
    error_hint: str,
    meta_model: str | ModelConfig | None = None,
    session_service: BaseSessionService | None = None,
) -> OutputEvaluation:
    """Use an LLM call to check pipeline output against error_hint."""
    resolved_meta, _, _ = resolve_meta_and_workers(meta_model, None, None)

    state_snapshot = _truncate(
        json.dumps(state, default=str, ensure_ascii=False),
        4000,
    )
    agent_names = ", ".join(
        f"{a.name} (output_key={a.output_key})" for a in config.agents
    )

    instruction = EVALUATOR_SYSTEM_PROMPT.substitute(
        state_snapshot=state_snapshot,
        agent_names=agent_names,
        error_hint=error_hint,
    )

    result = await run_meta_agent_call(
        agent_name="output_evaluator",
        instruction=instruction,
        user_message=f"Evaluate pipeline output against hint: {error_hint}",
        output_schema=OutputEvaluation,
        output_key="output_evaluation",
        model=resolved_meta,
        temperature=0.1,
        session_service=session_service,
        max_retries=1,
    )

    evaluation = parse_llm_output(result.raw_output, OutputEvaluation)
    _log.info(
        "Output evaluated | passed={} agent={} reasoning={}",
        evaluation.passed,
        evaluation.agent_name,
        evaluation.reasoning,
    )
    return evaluation


async def diagnose_and_fix(
    *,
    error: RunError,
    config: MAWConfig,
    task: str,
    state: dict[str, Any],
    meta_model: str | ModelConfig | None = None,
    temperature: float = 0.3,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: list[str | ModelConfig] | None = None,
    session_service: BaseSessionService | None = None,
    error_category: str | None = None,
) -> MAWAgentConfig:
    """Diagnose a failing agent and produce a fixed MAWAgentConfig."""
    resolved_meta, resolved_workers, resolved_temp = resolve_meta_and_workers(
        meta_model,
        worker_models,
        temperature,
    )

    failing_agent = next((a for a in config.agents if a.name == error.agent_name), None)
    if failing_agent is None:
        raise ValueError(f"Agent '{error.agent_name}' not found in config")

    descriptions = get_server_descriptions(mcp_registry)
    desc_text = format_server_descriptions(descriptions)
    models_text = "\n".join(f"- `{m.model}`" for m in resolved_workers)

    error_category_section = ""
    if error_category:
        error_category_section = (
            f"\n**Error category (from classifier):** {error_category}\n"
        )

    error_message = _truncate(error.message, _MAX_ERROR_MESSAGE_LEN)
    state_snapshot = _truncate(
        json.dumps(state, default=str, ensure_ascii=False),
        4000,
    )
    config_json = _truncate(config.model_dump_json(indent=2), 6000)

    instruction = DEBUGGER_SYSTEM_PROMPT.substitute(
        task=task,
        config_json=config_json,
        agent_name=error.agent_name,
        error_message=error_message,
        error_category_section=error_category_section,
        agent_config_json=failing_agent.model_dump_json(indent=2),
        state_snapshot=state_snapshot,
        mcp_servers_desc=desc_text,
        available_models=models_text,
        output_key=failing_agent.output_key,
    )

    result = await run_meta_agent_call(
        agent_name="debugger",
        instruction=instruction,
        user_message=(
            f"Fix agent '{error.agent_name}' that failed with: {error_message}"
        ),
        output_schema=MAWAgentConfig,
        output_key="fixed_agent",
        model=resolved_meta,
        temperature=resolved_temp,
        session_service=session_service,
        max_retries=2,
        allowed_models=[m.model for m in resolved_workers],
    )

    fixed = parse_llm_output(result.raw_output, MAWAgentConfig)

    # Enforce invariants: name and output_key must match the original.
    if fixed.name != error.agent_name:
        _log.warning(
            "Debugger changed agent name from '{}' to '{}', reverting",
            error.agent_name,
            fixed.name,
        )
        fixed = fixed.model_copy(update={"name": error.agent_name})
    if fixed.output_key != failing_agent.output_key:
        _log.warning(
            "Debugger changed output_key from '{}' to '{}', reverting",
            failing_agent.output_key,
            fixed.output_key,
        )
        fixed = fixed.model_copy(update={"output_key": failing_agent.output_key})

    _log.info(
        "Agent '{}' fixed | model={} tools={} instruction_len={}",
        fixed.name,
        fixed.model,
        fixed.tools,
        len(fixed.instruction),
    )
    return fixed
