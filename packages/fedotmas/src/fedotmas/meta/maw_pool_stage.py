from __future__ import annotations

from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.mcp import MCPServerConfig
from fedotmas.meta._adk_runner import LLMCallResult, run_meta_agent_call
from fedotmas.meta._helpers import (
    format_agent_pool,
    format_server_descriptions,
    parse_llm_output,
    resolve_meta_and_workers,
    resolve_tool_descriptions,
)
from fedotmas.meta.maw_prompts import POOL_AGENT_SYSTEM_PROMPT
from fedotmas.maw.models import AgentPoolConfig

_log = get_logger("fedotmas.meta.maw_pool_stage")


class PoolGenerator:
    """Generate an agent pool from a task description."""

    def __init__(
        self,
        *,
        meta_model: str | ModelConfig | None = None,
        worker_models: list[str | ModelConfig] | None = None,
        temperature: float | None = None,
        mcp_registry: dict[str, MCPServerConfig] | None = None,
        tool_catalog: dict[str, str] | None = None,
        session_service: BaseSessionService | None = None,
        max_retries: int = 2,
        plugins: list[BasePlugin] | None = None,
    ) -> None:
        self._resolved_meta, self._resolved_workers, self._temperature = (
            resolve_meta_and_workers(meta_model, worker_models, temperature)
        )
        self._mcp_registry = mcp_registry
        self._tool_catalog = tool_catalog
        self._session_service = session_service
        self._max_retries = max_retries
        self._plugins = plugins
        self.result: LLMCallResult | None = None

    async def generate(
        self, task: str, existing: AgentPoolConfig | None = None
    ) -> AgentPoolConfig:
        """Run LLM to produce ``AgentPoolConfig``.

        *existing* lists agents the caller already has. They are appended to the
        user message rather than the system prompt, so a run without them is
        byte-identical to before.
        """
        descriptions = resolve_tool_descriptions(self._mcp_registry, self._tool_catalog)
        desc_text = format_server_descriptions(descriptions)
        models_text = "\n".join(f"- `{m.model}`" for m in self._resolved_workers)

        instruction = POOL_AGENT_SYSTEM_PROMPT.substitute(
            mcp_servers_desc=desc_text,
            available_models=models_text,
        )

        user_message = f"TASK: {task}"
        if existing is not None and existing.agents:
            user_message += (
                "\n\nEXISTING AGENTS — reuse the ones that fit this task, keeping "
                "their names and instructions exactly as given, and add new agents "
                "only for roles none of them covers:\n" + format_agent_pool(existing)
            )

        self.result = await run_meta_agent_call(
            agent_name="pool_generator",
            instruction=instruction,
            user_message=user_message,
            output_schema=AgentPoolConfig,
            output_key="agent_pool",
            model=self._resolved_meta,
            temperature=self._temperature,
            session_service=self._session_service,
            max_retries=self._max_retries,
            allowed_models=[m.model for m in self._resolved_workers],
            plugins=self._plugins,
        )

        pool = parse_llm_output(self.result.raw_output, AgentPoolConfig)
        # ``AgentPoolConfig`` doubles as the output schema, so ``id`` is a field
        # the meta-agent can fill in.  An invented one would send an export at
        # somebody's existing record; only a caller may set it.
        for entry in pool.agents:
            entry.id = None

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
