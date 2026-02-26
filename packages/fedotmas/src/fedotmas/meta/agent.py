from __future__ import annotations

from dataclasses import dataclass, field

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import (
    ModelConfig,
    get_meta_model,
    get_meta_temperature,
    get_worker_models,
    resolve_model_config,
)
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas.meta._adk_runner import run_meta_agent_call
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
) -> MetaAgentResult:
    """Run the meta-agent and return a validated ``MetaAgentResult``.

    This is the **single-stage** generation path (``two_stage=False``).
    """
    # Resolve parameters with env fallback
    resolved_meta = (
        resolve_model_config(meta_model)
        if meta_model
        else resolve_model_config(get_meta_model())
    )
    resolved_workers = (
        [resolve_model_config(m) for m in worker_models]
        if worker_models
        else [resolve_model_config(m) for m in get_worker_models()]
    )
    resolved_temp = temperature if temperature is not None else get_meta_temperature()

    # Build prompt
    descriptions = get_server_descriptions(mcp_registry)
    desc_text = _format_descriptions(descriptions)
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
    )

    raw_config = result.raw_output
    if isinstance(raw_config, dict):
        config = PipelineConfig.model_validate(raw_config)
    elif isinstance(raw_config, str):
        config = PipelineConfig.model_validate_json(raw_config)
    else:
        raise TypeError(f"Unexpected pipeline_config type: {type(raw_config)}")

    _log.info("Config extracted | agents={}", len(config.agents))
    return MetaAgentResult(
        config=config,
        worker_models=resolved_workers,
        total_prompt_tokens=result.prompt_tokens,
        total_completion_tokens=result.completion_tokens,
        elapsed=result.elapsed,
    )


def _format_descriptions(descriptions: dict[str, str]) -> str:
    if not descriptions:
        return "No MCP tools available."
    lines = []
    for name, desc in descriptions.items():
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)
