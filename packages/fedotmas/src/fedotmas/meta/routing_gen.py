from __future__ import annotations

from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.mas.models import MASConfig
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas.meta._adk_runner import run_meta_agent_call
from fedotmas.meta._helpers import (
    format_server_descriptions,
    parse_llm_output,
    resolve_meta_and_workers,
)
from fedotmas.meta.agent import MetaAgentResult
from fedotmas.meta.routing_prompts import ROUTING_SYSTEM_PROMPT

_log = get_logger("fedotmas.meta.routing_gen")


async def generate_routing_config(
    task: str,
    *,
    meta_model: str | ModelConfig | None = None,
    worker_models: list[str | ModelConfig] | None = None,
    temperature: float | None = None,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    session_service: BaseSessionService | None = None,
    max_retries: int = 2,
) -> MetaAgentResult:
    """Run the meta-agent to generate an ``MASConfig`` for routing mode.

    Single-stage generation: produces a coordinator + workers configuration
    in one LLM call.
    """
    resolved_meta, resolved_workers, resolved_temp = resolve_meta_and_workers(
        meta_model,
        worker_models,
        temperature,
    )

    descriptions = get_server_descriptions(mcp_registry)
    desc_text = format_server_descriptions(descriptions)
    models_text = "\n".join(f"- `{m.model}`" for m in resolved_workers)

    instruction = ROUTING_SYSTEM_PROMPT.substitute(
        mcp_servers_desc=desc_text,
        available_models=models_text,
    )

    result = await run_meta_agent_call(
        agent_name="routing_meta_agent",
        instruction=instruction,
        user_message=f"TASK: {task}",
        output_schema=MASConfig,
        output_key="agent_system_config",
        model=resolved_meta,
        temperature=resolved_temp,
        session_service=session_service,
        max_retries=max_retries,
        allowed_models=[m.model for m in resolved_workers],
    )

    config = parse_llm_output(result.raw_output, MASConfig)

    _log.info(
        "Routing config generated | coordinator={} workers={}",
        config.coordinator.name,
        len(config.workers),
    )
    return MetaAgentResult(
        config=config,
        worker_models=resolved_workers,
        total_prompt_tokens=result.prompt_tokens,
        total_completion_tokens=result.completion_tokens,
        elapsed=result.elapsed,
    )
