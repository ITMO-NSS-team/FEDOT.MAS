from __future__ import annotations

from google.adk.agents.base_agent import BaseAgent

from fedotmas.common.logging import get_logger
from fedotmas.core.base import BaseMAS
from fedotmas.mas.builder import build_routing_system
from fedotmas.mas.models import MASConfig
from fedotmas.meta.mas_gen import generate_routing_config

_log = get_logger("fedotmas.mas")


class MAS(BaseMAS[MASConfig]):
    """Multi-Agent System are dynamic LLM-driven routing.

    Generates and executes agent systems where a coordinator agent
    dynamically routes tasks to specialized workers using ADK AutoFlow's
    ``transfer_to_agent`` mechanism.

    Usage::

        mas = MAS()
        result = await mas.run("Handle customer support request")

        # Two-step with review:
        config = await mas.generate_config("Handle customer support request")
        result = await mas.build_and_run(config, "Handle customer support request")
    """

    async def generate_config(self, task: str) -> MASConfig:
        """Ask the meta-agent to design a routing system for *task*.

        Returns an ``MASConfig`` with a coordinator and workers.
        """
        _log.info("Generating routing config for task: {}", task)

        meta_result = await generate_routing_config(
            task,
            meta_model=self._meta_model,
            worker_models=self._worker_models,
            temperature=self._temperature,
            mcp_registry=self._mcp_registry,
            session_service=self._session_service,
            max_retries=self._max_retries,
        )

        self._last_meta_result = meta_result
        self._resolved_workers = meta_result.worker_models
        config = meta_result.config
        assert isinstance(config, MASConfig)
        _log.info(
            "Routing config generated | coordinator={} workers={}",
            config.coordinator.name,
            len(config.workers),
        )
        return config

    def build(self, config: MASConfig) -> BaseAgent:
        """Build an ADK agent tree with routing from *config*."""
        _log.info("Building routing system")
        agent = build_routing_system(
            config,
            mcp_registry=self._mcp_registry,
            worker_models=self._worker_map(),
        )
        _log.info("Config:\n{}", config)
        return agent
