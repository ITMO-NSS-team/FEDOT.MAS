from __future__ import annotations

from google.adk.sessions import BaseSessionService

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
from fedotmas.meta.prompts import POOL_AGENT_SYSTEM_PROMPT
from fedotmas.pipeline.models import AgentPoolConfig

_log = get_logger("fedotmas.meta.pool_gen")


class PoolGenerator:
    """Generate an agent pool from a task description."""

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
        self._session_service = session_service
        self._max_retries = max_retries
        self.result: LLMCallResult | None = None

    async def generate(self, task: str) -> AgentPoolConfig:
        """Run LLM to produce ``AgentPoolConfig``."""
        descriptions = get_server_descriptions(self._mcp_registry)
        desc_text = _format_descriptions(descriptions)
        models_text = "\n".join(f"- `{m.model}`" for m in self._resolved_workers)

        instruction = POOL_AGENT_SYSTEM_PROMPT.substitute(
            mcp_servers_desc=desc_text,
            available_models=models_text,
        )

        self.result = await run_meta_agent_call(
            agent_name="pool_generator",
            instruction=instruction,
            user_message=f"TASK: {task}",
            output_schema=AgentPoolConfig,
            output_key="agent_pool",
            model=self._resolved_meta,
            temperature=self._temperature,
            session_service=self._session_service,
            max_retries=self._max_retries,
        )

        raw = self.result.raw_output
        if isinstance(raw, dict):
            pool = AgentPoolConfig.model_validate(raw)
        elif isinstance(raw, str):
            pool = AgentPoolConfig.model_validate_json(raw)
        else:
            raise TypeError(f"Unexpected agent_pool type: {type(raw)}")

        _log.info(
            "Pool generated | agents={}",
            len(pool.agents),
        )
        for a in pool.agents:
            _log.debug(
                "  agent={} model={} tools={} instruction={}",
                a.name,
                a.model,
                a.tools,
                a.instruction[:120],
            )
        return pool


def _format_descriptions(descriptions: dict[str, str]) -> str:
    if not descriptions:
        return "No MCP tools available."
    return "\n".join(f"- **{name}**: {desc}" for name, desc in descriptions.items())
