from __future__ import annotations

from typing import Any

from fedotmas.mcp.registry import MCP_SERVERS, MCPServerConfig
from fedotmas.meta.agent import generate_pipeline_config
from fedotmas.pipeline.builder import build
from fedotmas.pipeline.models import PipelineConfig
from fedotmas.pipeline.runner import run_pipeline


class MASOrchestrator:
    """High-level API for automatic multi-agent pipeline generation and execution.

    Usage::

        mas = MASOrchestrator()

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
    ) -> None:
        self._model = model
        self._mcp_registry = mcp_registry or MCP_SERVERS

    async def generate_config(self, task: str) -> PipelineConfig:
        """Ask the meta-agent to design a pipeline for *task*.

        Returns a ``PipelineConfig`` that can be inspected, serialised to
        JSON for human review, and optionally edited before execution.
        """
        return await generate_pipeline_config(
            task,
            model=self._model,
            mcp_registry=self._mcp_registry,
        )

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
        agent = build(config, mcp_registry=self._mcp_registry)
        return await run_pipeline(
            agent,
            user_query,
            initial_state=initial_state,
        )

    async def run(
        self,
        task: str,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a pipeline config and immediately execute it.

        Equivalent to ``generate_config`` followed by ``build_and_run``.
        """
        config = await self.generate_config(task)
        return await self.build_and_run(config, task, initial_state=initial_state)
