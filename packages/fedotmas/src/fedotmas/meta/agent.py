from __future__ import annotations

from dataclasses import dataclass, field

from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import ModelConfig
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas.meta._adk_runner import run_meta_agent_call
from fedotmas.meta._helpers import format_server_descriptions, parse_llm_output, resolve_meta_and_workers
from fedotmas.meta.prompts import META_AGENT_SYSTEM_PROMPT
from fedotmas.pipeline.models import PipelineConfig

_log = get_logger("fedotmas.meta.agent")


@dataclass
class MetaAgentResult:
    config: PipelineConfig
    worker_models: list[ModelConfig] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed: float = 0.0


async def generate_pipeline_config(
    task: str,
    *,
    meta_model: str | ModelConfig | None = None,
    worker_models: list[str | ModelConfig] | None = None,
    temperature: float | None = None,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    session_service: BaseSessionService | None = None,
    max_retries: int = 2,
) -> MetaAgentResult:
    """Run the meta-agent and return a validated ``MetaAgentResult``.

    This is the **single-stage** generation path (``two_stage=False``).
    """
    resolved_meta, resolved_workers, resolved_temp = resolve_meta_and_workers(
        meta_model, worker_models, temperature,
    )

    descriptions = get_server_descriptions(mcp_registry)
    desc_text = format_server_descriptions(descriptions)
    models_text = "\n".join(f"- `{m.model}`" for m in resolved_workers)

    instruction = META_AGENT_SYSTEM_PROMPT.substitute(
        mcp_servers_desc=desc_text,
        available_models=models_text,
    )

    result = await run_meta_agent_call(
        agent_name="meta_agent",
        instruction=instruction,
        user_message=f"TASK: {task}",
        output_schema=PipelineConfig,
        output_key="pipeline_config",
        model=resolved_meta,
        temperature=resolved_temp,
        session_service=session_service,
        max_retries=max_retries,
        allowed_models=[m.model for m in resolved_workers],
    )

    config = parse_llm_output(result.raw_output, PipelineConfig)

    _log.info("Config extracted | agents={}", len(config.agents))
    return MetaAgentResult(
        config=config,
        worker_models=resolved_workers,
        total_prompt_tokens=result.prompt_tokens,
        total_completion_tokens=result.completion_tokens,
        elapsed=result.elapsed,
    )
