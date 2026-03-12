from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar

from fastapi import FastAPI
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.memory import BaseMemoryService
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger, setup_logging
from fedotmas._settings import ModelConfig, resolve_model_config
from fedotmas.core.runner import PipelineResult, run_pipeline
from fedotmas.mcp import MCPServerConfig, resolve_mcp_registry
from fedotmas.meta._result import MetaAgentResult
from fedotmas.plugins import LoggingPlugin

_log = get_logger("fedotmas.core.base")

ConfigT = TypeVar("ConfigT")


class BaseMAS(ABC, Generic[ConfigT]):
    """Abstract base class for multi-agent system orchestration.

    Provides shared infrastructure: logging, MCP registry, model resolution,
    token tracking, and common execution methods (build_app, build_and_run,
    serve, run).

    Subclasses implement ``generate_config()`` and ``build()`` for their
    specific orchestration mode.
    """

    def __init__(
        self,
        *,
        meta_model: str | ModelConfig | None = None,
        worker_models: list[str | ModelConfig] | None = None,
        temperature: float | None = None,
        mcp_servers: list[str]
        | dict[str, MCPServerConfig]
        | Literal["all"]
        | None = None,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
        plugins: list[BasePlugin] | None = None,
        max_retries: int = 3,
    ) -> None:
        setup_logging()
        self._meta_model = meta_model
        self._worker_models = worker_models
        self._temperature = temperature
        self._mcp_registry = resolve_mcp_registry(mcp_servers)
        self._session_service = session_service
        self._memory_service = memory_service
        if plugins is not None:
            self._plugins = list(plugins)
        else:
            self._plugins = [LoggingPlugin()]
        self._max_retries = max_retries
        self._last_result: PipelineResult | None = None
        self._last_meta_result: MetaAgentResult | None = None
        self._resolved_workers: list[ModelConfig] | None = None

    @property
    def mcp_servers(self) -> dict[str, MCPServerConfig]:
        """Registry of MCP servers available to this instance."""
        return dict(self._mcp_registry)

    @property
    def last_result(self) -> PipelineResult | None:
        """The result of the most recent pipeline execution, or ``None``."""
        return self._last_result

    @property
    def meta_prompt_tokens(self) -> int:
        return (
            self._last_meta_result.total_prompt_tokens if self._last_meta_result else 0
        )

    @property
    def meta_completion_tokens(self) -> int:
        return (
            self._last_meta_result.total_completion_tokens
            if self._last_meta_result
            else 0
        )

    @property
    def meta_elapsed(self) -> float:
        return self._last_meta_result.elapsed if self._last_meta_result else 0.0

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens from the last run (meta-agent + pipeline)."""
        pipeline = self._last_result.total_prompt_tokens if self._last_result else 0
        meta = (
            self._last_meta_result.total_prompt_tokens if self._last_meta_result else 0
        )
        return pipeline + meta

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens from the last run (meta-agent + pipeline)."""
        pipeline = self._last_result.total_completion_tokens if self._last_result else 0
        meta = (
            self._last_meta_result.total_completion_tokens
            if self._last_meta_result
            else 0
        )
        return pipeline + meta

    @property
    def elapsed(self) -> float:
        """Elapsed seconds for the last run (meta-agent + pipeline)."""
        pipeline = self._last_result.elapsed if self._last_result else 0.0
        meta = self._last_meta_result.elapsed if self._last_meta_result else 0.0
        return pipeline + meta

    def _ensure_resolved_workers(self) -> None:
        if self._resolved_workers is not None:
            return
        if self._worker_models:
            self._resolved_workers = [
                resolve_model_config(m) for m in self._worker_models
            ]

    def _worker_map(self) -> dict[str, ModelConfig] | None:
        self._ensure_resolved_workers()
        if self._resolved_workers:
            return {m.model: m for m in self._resolved_workers}
        return None

    @abstractmethod
    async def generate_config(self, task: str) -> ConfigT: ...

    @abstractmethod
    def build(self, config: ConfigT) -> BaseAgent: ...

    def build_app(self, config: ConfigT, *, name: str = "fedotmas") -> App:
        """Build an ADK ``App`` (agent tree + plugins) from *config*."""
        agent = self.build(config)
        return App(name=name, root_agent=agent, plugins=list(self._plugins))

    async def build_and_run(
        self,
        config: ConfigT,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the ADK agent tree from *config* and execute it.

        Returns the final ``session.state`` dict.
        """
        app = self.build_app(config)
        _log.info("Running pipeline")
        self._last_result = await run_pipeline(
            app,
            user_query,
            session_service=self._session_service,
            memory_service=self._memory_service,
            initial_state=initial_state,
        )
        return self._last_result.state

    def serve(
        self,
        config: ConfigT,
        *,
        name: str = "fedotmas",
        session_service_uri: str | None = None,
        web: bool = False,
        host: str = "127.0.0.1",
        port: int = 8000,
        allow_origins: list[str] | None = None,
        auto_create_session: bool = False,
    ) -> FastAPI:
        """Build an ``App`` from *config* and create a FastAPI server."""
        from fedotmas._serving import serve as _serve

        app = self.build_app(config, name=name)
        return _serve(
            {name: app},
            session_service=self._session_service,
            session_service_uri=session_service_uri,
            web=web,
            host=host,
            port=port,
            allow_origins=allow_origins,
            auto_create_session=auto_create_session,
        )

    async def run(
        self,
        task: str,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a config and immediately execute it.

        Equivalent to ``generate_config`` followed by ``build_and_run``.
        """
        _log.info("Full-auto run for task: {}", task)
        config = await self.generate_config(task)
        return await self.build_and_run(config, task, initial_state=initial_state)
