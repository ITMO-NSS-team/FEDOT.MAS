from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from fastapi import FastAPI
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.memory import BaseMemoryService
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService

from fedotmas._settings import ModelConfig, resolve_model_config
from fedotmas.common.logging import get_logger, setup_logging
from fedotmas.core.runner import PipelineExecutionError, PipelineResult, run_pipeline
from fedotmas.mcp import MCPServerConfig, describe_servers, resolve_mcp_registry
from fedotmas.meta._result import MetaAgentResult
from fedotmas.plugins import (
    DEFAULT_WEB_SEARCH_LIMIT,
    LoggingPlugin,
    UnknownToolRecoveryPlugin,
    WebSearchLimitPlugin,
)

_log = get_logger("fedotmas.core.base")


class _Unset:
    """Tells "left at its default" apart from an explicit value, ``None`` included."""


_UNSET = _Unset()

ConfigT = TypeVar("ConfigT")


class _TraceFinalizer(Protocol):
    def end_trace(self) -> None: ...


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
        tool_catalog: dict[str, str] | None = None,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
        plugins: list[BasePlugin] | None = None,
        max_retries: int = 3,
        web_search_limit: int | None | _Unset = _UNSET,
    ) -> None:
        setup_logging()
        self._meta_model = meta_model
        self._worker_models = worker_models
        self._temperature = temperature
        self._mcp_registry = resolve_mcp_registry(mcp_servers)
        # A caller who built the configs itself named addresses of its own; one
        # that does not answer is a mistake in the request, not in the machine,
        # so it is reported rather than skipped.
        self._mcp_supplied = isinstance(mcp_servers, dict)
        self._mcp_prepared = False
        self._tool_catalog = dict(tool_catalog) if tool_catalog is not None else None
        # An explicit [] asks for no tools, and a catalogue brings its own; the
        # rest is worth reporting, "all" over an empty catalogue included.
        asked_for_none = mcp_servers is not None and not mcp_servers
        if not self._mcp_registry and tool_catalog is None and not asked_for_none:
            _log.warning(
                "No MCP servers given: agents will have no tools and may invent "
                "what they cannot look up. Pass mcp_servers='all' or a list of "
                "server names."
            )
        self._session_service = session_service
        self._memory_service = memory_service
        if plugins is not None:
            if not isinstance(web_search_limit, _Unset):
                raise ValueError(
                    "web_search_limit configures the default plugin set, and a "
                    "plugins= list replaces that set; put a WebSearchLimitPlugin "
                    "in the list instead."
                )
            self._plugins: list[BasePlugin] = list(plugins)
        else:
            limit = (
                DEFAULT_WEB_SEARCH_LIMIT
                if isinstance(web_search_limit, _Unset)
                else web_search_limit
            )
            self._plugins = [LoggingPlugin(), UnknownToolRecoveryPlugin()]
            if limit is not None:
                self._plugins.append(WebSearchLimitPlugin(max_calls_per_agent=limit))
        self._max_retries = max_retries
        self._last_result: PipelineResult | None = None
        self._last_meta_result: MetaAgentResult | None = None
        self._resolved_workers: list[ModelConfig] | None = None

    @property
    def meta_model(self) -> str | ModelConfig | None:
        """The meta-model used for pipeline generation and debugging."""
        return self._meta_model

    @property
    def worker_models(self) -> list[str | ModelConfig] | None:
        """Worker models available for pipeline agents."""
        return self._worker_models

    @property
    def temperature(self) -> float | None:
        """Temperature setting for meta-agent LLM calls."""
        return self._temperature

    @property
    def mcp_registry(self) -> dict[str, MCPServerConfig]:
        """Registry of MCP servers available to this instance."""
        return self._mcp_registry

    async def _prepare_mcp_registry(self) -> None:
        """Describe the servers that carry no description of their own.

        A server handed in by URL is advertised to the meta-agent as ``"MCP
        server: <name>"`` until it is asked for its own tool list, which names
        it without saying what it does; and an address that answers nothing
        should surface before generation is paid for.  Only such servers are
        contacted: one that declares a description keeps it, and its
        reachability is left to ``just doctor`` rather than checked on every
        run.  Skipped entirely when a ``tool_catalog`` is set, which replaces
        the registry for generation anyway.

        Runs once per instance.  The flag is set only after a successful pass,
        so a caller that starts the server and retries is described then; two
        concurrent calls both describe, which costs a duplicate connection and
        converges on the same registry.
        """
        if self._mcp_prepared or self._tool_catalog is not None:
            return
        if not self._mcp_registry:
            self._mcp_prepared = True
            return

        described, unreachable = await describe_servers(self._mcp_registry)
        if unreachable:
            listed = ", ".join(f"{u.name} ({u.reason})" for u in unreachable)
            if self._mcp_supplied:
                raise ValueError(
                    f"These MCP servers did not answer: {listed}. Check the "
                    "address, whether the server is running, and any headers "
                    "it needs."
                )
            _log.warning(
                "Generating without undescribed MCP servers that did not answer: {}",
                listed,
            )
            for dead in unreachable:
                described.pop(dead.name, None)
        self._mcp_registry = described
        self._mcp_prepared = True

    def _reject_external_build(self) -> None:
        """Fail a build on an instance that generates for another runtime.

        Provenance is the catalogue, not the tool names: a name this workspace
        also happens to have is a different tool over there, and a config that
        came out toolless was still designed against their catalogue.
        """
        if self._tool_catalog is None:
            return
        raise ValueError(
            "Cannot build locally: this instance was given a tool_catalog, so the "
            "configs it generates describe another runtime's tools and are meant "
            "to be exported and run there. Build from an instance without one."
        )

    @property
    def tool_catalog(self) -> dict[str, str] | None:
        """Tool catalogue advertised to the meta-agent instead of the MCP registry.

        Set it to generate configs for a runtime other than this one; such a
        config names that runtime's tools and cannot be built locally.
        """
        return dict(self._tool_catalog) if self._tool_catalog is not None else None

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
    def build(self, config: ConfigT, *, autonomous: bool = True) -> BaseAgent: ...

    def build_app(
        self, config: ConfigT, *, name: str = "fedotmas", autonomous: bool = True
    ) -> App:
        """Build an ADK ``App`` (agent tree + plugins) from *config*."""
        agent = self.build(config, autonomous=autonomous)
        return App(name=name, root_agent=agent, plugins=list(self._plugins))

    async def build_and_run(
        self,
        config: ConfigT,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Build the ADK agent tree from *config* and execute it.

        Returns the final ``session.state`` dict. When *timeout* is set and
        execution exceeds it, the partial state accumulated so far is returned
        instead of raising (see :func:`run_pipeline`).
        """
        app = self.build_app(config)
        _log.info("Running pipeline")
        try:
            self._last_result = await run_pipeline(
                app,
                user_query,
                session_service=self._session_service,
                memory_service=self._memory_service,
                initial_state=initial_state,
                timeout=timeout,
            )
        except PipelineExecutionError as error:
            self._last_result = error.result
            raise
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
        autonomous: bool = True,
    ) -> FastAPI:
        """Build an ``App`` from *config* and create a FastAPI server.

        Serving covers both an unattended HTTP consumer and a person chatting
        through a UI.  Only the caller knows which, so pass ``autonomous=False``
        for the latter to let agents ask their clarifying questions.
        """
        from fedotmas._serving import serve as _serve

        app = self.build_app(config, name=name, autonomous=autonomous)
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

    def _finalize_langfuse(self) -> None:
        """End Langfuse trace if a LangfusePlugin is among the plugins."""
        for plugin in self._plugins:
            if hasattr(plugin, "end_trace"):
                try:
                    cast(_TraceFinalizer, plugin).end_trace()
                except Exception as exc:
                    _log.warning("Plugin finalization failed: {}", exc)

    async def run(
        self,
        task: str,
        *,
        initial_state: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Generate a config and immediately execute it.

        Equivalent to ``generate_config`` followed by ``build_and_run``. When
        *timeout* is set it bounds pipeline *execution*; on expiry the partial
        state gathered so far is returned rather than raising.
        """
        # The build at the far end would refuse this instance anyway; refusing
        # here keeps the generation call from being paid for first.
        self._reject_external_build()
        _log.info("Full-auto run for task: {}", task)
        try:
            config = await self.generate_config(task)
            return await self.build_and_run(
                config, task, initial_state=initial_state, timeout=timeout
            )
        finally:
            self._finalize_langfuse()
