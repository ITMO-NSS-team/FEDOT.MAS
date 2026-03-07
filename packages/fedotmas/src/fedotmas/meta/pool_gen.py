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
from fedotmas.meta.prompts import POOL_AGENT_SYSTEM_PROMPT
from fedotmas.maw.models import AgentPoolConfig

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
        self._resolved_meta, self._resolved_workers, self._temperature = (
            resolve_meta_and_workers(meta_model, worker_models, temperature)
        )
        self._mcp_registry = mcp_registry
        self._session_service = session_service
        self._max_retries = max_retries
        self.result: LLMCallResult | None = None

    async def generate(self, task: str) -> AgentPoolConfig:
        """Run LLM to produce ``AgentPoolConfig``."""
        descriptions = get_server_descriptions(self._mcp_registry)
        desc_text = format_server_descriptions(descriptions)
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
            allowed_models=[m.model for m in self._resolved_workers],
        )

        pool = parse_llm_output(self.result.raw_output, AgentPoolConfig)

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
