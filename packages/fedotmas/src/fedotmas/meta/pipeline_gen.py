from __future__ import annotations

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import (
    ModelConfig,
    get_meta_model,
    get_meta_temperature,
    get_worker_models,
    resolve_model_config,
)
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.meta.prompts import PIPELINE_AGENT_SYSTEM_PROMPT
from fedotmas.pipeline.models import AgentPoolConfig, PipelineConfig, StepConfig

_log = get_logger("fedotmas.meta.pipeline_gen")


class PipelineGenerator:
    """Generate a PipelineConfig with wiring given an agent pool."""

    def __init__(
        self,
        *,
        meta_model: str | ModelConfig | None = None,
        worker_models: list[str | ModelConfig] | None = None,
        temperature: float | None = None,
        mcp_registry: dict[str, MCPServerConfig] | None = None,
    ) -> None:
        self._resolved_meta = (
            resolve_model_config(meta_model)
            if meta_model
            else resolve_model_config(get_meta_model())
        )
        self._resolved_workers = (
            [resolve_model_config(m) for m in worker_models]
            if worker_models
            else [resolve_model_config(m) for m in get_worker_models()]
        )
        self._temperature = (
            temperature if temperature is not None else get_meta_temperature()
        )
        self._mcp_registry = mcp_registry
        self.result: LLMCallResult | None = None

    async def generate(self, task: str, pool: AgentPoolConfig) -> PipelineConfig:
        """Run LLM to produce ``PipelineConfig`` constrained to *pool* agents."""
        descriptions = get_server_descriptions(self._mcp_registry)
        desc_text = _format_descriptions(descriptions)
        models_text = "\n".join(f"- `{m.model}`" for m in self._resolved_workers)

        instruction = PIPELINE_AGENT_SYSTEM_PROMPT.substitute(
            mcp_servers_desc=desc_text,
            available_models=models_text,
        )

        pool_text = self._format_pool(pool)
        user_msg = f"TASK: {task}\n\nAGENT POOL:\n{pool_text}"

        self.result = await run_meta_agent_call(
            agent_name="pipeline_generator",
            instruction=instruction,
            user_message=user_msg,
            output_schema=PipelineConfig,
            output_key="pipeline_config",
            model=self._resolved_meta,
            temperature=self._temperature,
        )

        raw = self.result.raw_output
        if isinstance(raw, dict):
            config = PipelineConfig.model_validate(raw)
        elif isinstance(raw, str):
            config = PipelineConfig.model_validate_json(raw)
        else:
            raise TypeError(f"Unexpected pipeline_config type: {type(raw)}")

        self._validate_against_pool(config, pool)

        _log.info(
            "Pipeline generated | agents={} pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        _log.debug("Pipeline config:\n{}", config.model_dump_json(indent=2))
        return config

    @staticmethod
    def _validate_against_pool(config: PipelineConfig, pool: AgentPoolConfig) -> None:
        """Raise ``ValueError`` if *config* references agents not in *pool*."""
        pool_names = {a.name for a in pool.agents}
        config_names = {a.name for a in config.agents}
        extra = config_names - pool_names
        if extra:
            raise ValueError(
                f"Pipeline references agents not in pool: {extra}. "
                f"Pool agents: {pool_names}"
            )

    @staticmethod
    def _format_pool(pool: AgentPoolConfig) -> str:
        """Format pool as readable text for the stage-2 user message."""
        lines: list[str] = []
        for a in pool.agents:
            parts = [f"- **{a.name}**"]
            if a.model:
                parts.append(f"  model: {a.model}")
            parts.append(f"  instruction: {a.instruction}")
            if a.tools:
                parts.append(f"  tools: {', '.join(a.tools)}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    @staticmethod
    def _collect_agent_names(step: StepConfig) -> set[str]:
        """Recursively collect all agent names referenced in a pipeline tree."""
        names: set[str] = set()
        if step.agent_name:
            names.add(step.agent_name)
        for child in step.children:
            names.update(PipelineGenerator._collect_agent_names(child))
        return names


def _format_descriptions(descriptions: dict[str, str]) -> str:
    if not descriptions:
        return "No MCP tools available."
    return "\n".join(f"- **{name}**: {desc}" for name, desc in descriptions.items())
