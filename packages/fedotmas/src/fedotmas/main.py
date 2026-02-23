from __future__ import annotations

from typing import Any

from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger
from fedotmas.mcp import MCP_SERVERS, MCPServerConfig
from fedotmas.meta.agent import generate_pipeline_config
from fedotmas.pipeline._ppline_utils import print_tree
from fedotmas.pipeline.builder import AgentCallback, build
from fedotmas.pipeline.models import PipelineConfig
from fedotmas.pipeline.runner import EventCallback, PipelineResult, run_pipeline

_log = get_logger("fedotmas.main")


class MAS:
    """High-level API for automatic multi-agent pipeline generation and execution.

    Usage::

        mas = MAS()

        # Full auto: generate + run
        result = await mas.run("Research quantum computing trends")

        # Two-step with human review:
        config = await mas.generate_config("Research quantum computing trends")
        # ... inspect / edit config ...
        result = await mas.build_and_run(config, "Research quantum computing trends")
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        mcp_registry: dict[str, MCPServerConfig] | None = None,
        session_service: BaseSessionService | None = None,
        event_callback: EventCallback | None = None,
        before_agent_callbacks: list[AgentCallback] | None = None,
        after_agent_callbacks: list[AgentCallback] | None = None,
    ) -> None:
        self._model = model
        self._mcp_registry = mcp_registry or MCP_SERVERS
        self._session_service = session_service
        self._event_callback = event_callback
        self._before_agent_callbacks = before_agent_callbacks
        self._after_agent_callbacks = after_agent_callbacks
        self._last_result: PipelineResult | None = None

    @property
    def last_result(self) -> PipelineResult | None:
        """The result of the most recent pipeline execution, or ``None``."""
        return self._last_result

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens from the last run."""
        return self._last_result.total_prompt_tokens if self._last_result else 0

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens from the last run."""
        return self._last_result.total_completion_tokens if self._last_result else 0

    @property
    def elapsed(self) -> float:
        """Elapsed seconds for the last run."""
        return self._last_result.elapsed if self._last_result else 0.0

    async def generate_config(self, task: str) -> PipelineConfig:
        """Ask the meta-agent to design a pipeline for *task*.

        Returns a ``PipelineConfig`` that can be inspected, serialised to
        JSON for human review, and optionally edited before execution.
        """
        _log.info("Generating pipeline config for task: {}", task)
        config = await generate_pipeline_config(
            task,
            model=self._model,
            mcp_registry=self._mcp_registry,
        )
        _log.info(
            "Config generated | agents={} pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        return config

    async def build_and_run(
        self,
        config: PipelineConfig,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the ADK agent tree from *config* and execute it.

        Returns the final ``session.state`` dict.
        """
        _log.info("Building agent tree")
        agent = build(
            config,
            mcp_registry=self._mcp_registry,
            before_agent_callbacks=self._before_agent_callbacks,
            after_agent_callbacks=self._after_agent_callbacks,
        )
        print_tree(config)
        _log.info("Running pipeline")
        self._last_result = await run_pipeline(
            agent,
            user_query,
            session_service=self._session_service,
            event_callback=self._event_callback,
            initial_state=initial_state,
        )
        return self._last_result.state

    async def run(
        self,
        task: str,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a pipeline config and immediately execute it.

        Equivalent to ``generate_config`` followed by ``build_and_run``.
        """
        _log.info("Full-auto run for task: {}", task)
        config = await self.generate_config(task)
        return await self.build_and_run(config, task, initial_state=initial_state)
