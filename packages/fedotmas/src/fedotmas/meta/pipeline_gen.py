from __future__ import annotations

from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import ModelConfig
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.meta._helpers import (
    format_server_descriptions,
    parse_llm_output,
    resolve_meta_and_workers,
)
from fedotmas.meta.prompts import PIPELINE_AGENT_SYSTEM_PROMPT
from fedotmas.maw.models import AgentPoolConfig, MAWConfig

_log = get_logger("fedotmas.meta.pipeline_gen")


class PipelineGenerator:
    """Generate a MAWConfig with wiring given an agent pool."""

    def __init__(
        self,
        *,
        meta_model: str | ModelConfig | None = None,
        worker_models: list[str | ModelConfig] | None = None,
        temperature: float | None = None,
        mcp_registry: dict[str, MCPServerConfig] | None = None,
        session_service: BaseSessionService | None = None,
        max_retries: int = 2,
    ) -> None:
        self._resolved_meta, self._resolved_workers, self._temperature = (
            resolve_meta_and_workers(meta_model, worker_models, temperature)
        )
        self._mcp_registry = mcp_registry
        self._session_service = session_service
        self._max_retries = max_retries
        self.result: LLMCallResult | None = None

    async def generate(self, task: str, pool: AgentPoolConfig) -> MAWConfig:
        """Run LLM to produce ``MAWConfig`` constrained to *pool* agents."""
        descriptions = get_server_descriptions(self._mcp_registry)
        desc_text = format_server_descriptions(descriptions)
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
            output_schema=MAWConfig,
            output_key="pipeline_config",
            model=self._resolved_meta,
            temperature=self._temperature,
            session_service=self._session_service,
            max_retries=self._max_retries,
            allowed_models=[m.model for m in self._resolved_workers],
        )

        config = parse_llm_output(self.result.raw_output, MAWConfig)

        self._validate_against_pool(config, pool)

        _log.info(
            "Pipeline generated | agents={} pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        _log.debug("Pipeline config:\n{}", config.model_dump_json(indent=2))
        return config

    @staticmethod
    def _validate_against_pool(config: MAWConfig, pool: AgentPoolConfig) -> None:
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
