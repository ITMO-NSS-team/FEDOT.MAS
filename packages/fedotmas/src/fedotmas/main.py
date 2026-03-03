from __future__ import annotations

from typing import Any, Literal

from google.adk.memory import BaseMemoryService
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService

from fedotmas.common.logging import get_logger, setup_logging
from fedotmas.config.settings import ModelConfig
from fedotmas.mcp import MCPServerConfig, resolve_mcp_registry
from fedotmas.meta.agent import MetaAgentResult, generate_pipeline_config
from fedotmas.meta.pipeline_gen import PipelineGenerator
from fedotmas.meta.pool_gen import PoolGenerator
from fedotmas.pipeline._ppline_utils import print_tree
from fedotmas.pipeline.builder import AgentCallback, build
from fedotmas.pipeline.models import PipelineConfig
from fedotmas.pipeline.runner import EventCallback, PipelineResult, run_pipeline

_log = get_logger("fedotmas.main")


class MAS:
    """High-level API for automatic multi-agent pipeline generation and execution.

    Args:
        two_stage: When ``True`` (default), pipeline generation is split into
            two LLM calls — first an agent pool is generated, then the
            pipeline tree is designed around that pool. This improves accuracy
            by letting each call focus on a single concern.  Set to ``False``
            for the legacy single-call behaviour.
        meta_model: LLM for the meta-agent that designs the pipeline.
            A model name string (e.g. ``"openai/gpt-4o"``) or a
            ``ModelConfig`` with custom endpoint/key. Defaults to the
            built-in model pool. Examples::

                meta_model="openai/gpt-4o"
                meta_model=ModelConfig(model="openai/gpt-4o", api_base="https://my-proxy.example.com/v1")

        worker_models: Pool of LLMs for worker agents. Each element is
            a model name string or ``ModelConfig``. When ``None``, the
            meta-agent picks from the built-in pool. Examples::

                worker_models=["openai/gpt-4o-mini", "google/gemini-2.0-flash"]
                worker_models=[ModelConfig(model="openai/gpt-4o", api_key="sk-...")]
        temperature: Sampling temperature for LLM calls. ``None`` keeps
            the provider default.
        mcp_servers: MCP servers to use. Defaults to ``None`` (no MCP tools).
            - ``["server1", "server2"]`` - filter: use only these registered servers.
            - ``{name: MCPServerConfig, ...}`` - explicit server configurations.
            - ``"all"`` - enable every registered server.
        session_service: ADK ``BaseSessionService`` for session persistence.
            Defaults to an in-memory service.
        memory_service: ADK ``BaseMemoryService`` for long-term memory.
            Defaults to ``None`` (no memory).
        plugins: ADK ``BasePlugin`` instances registered on the ``Runner``.
            Plugins intercept all agent, model, and tool lifecycle events.
        event_callback: Async or sync callable invoked on every ``Event``
            emitted during pipeline execution.
        before_agent_callbacks: Callbacks invoked before each agent step.
        after_agent_callbacks: Callbacks invoked after each agent step.
        max_retries: Max retry attempts for meta-agent LLM calls on failure
            (e.g. invalid JSON). Defaults to 3.

    Usage::

        mas = MAS()  # two_stage=True by default
        mas_with_mcp = MAS(mcp_servers=["browser", "filesystem"])

        # Full auto: generate + run
        result = await mas.run("Research quantum computing trends")

        # Two-step with human review:
        config = await mas.generate_config("Research quantum computing trends")
        # ... inspect / edit config ...
        result = await mas.build_and_run(config, "Research quantum computing trends")

        # Legacy single-stage generation:
        mas_legacy = MAS(two_stage=False)
    """

    def __init__(
        self,
        *,
        two_stage: bool = True,
        meta_model: str | ModelConfig | None = None,
        worker_models: list[str | ModelConfig] | None = None,
        temperature: float | None = None,
        mcp_servers: list[str] | dict[str, MCPServerConfig] | Literal["all"] | None = None,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
        plugins: list[BasePlugin] | None = None,
        event_callback: EventCallback | None = None,
        before_agent_callbacks: list[AgentCallback] | None = None,
        after_agent_callbacks: list[AgentCallback] | None = None,
        max_retries: int = 3,
    ) -> None:
        setup_logging()
        self._two_stage = two_stage
        self._meta_model = meta_model
        self._worker_models = worker_models
        self._temperature = temperature
        self._mcp_registry = resolve_mcp_registry(mcp_servers or [])
        self._session_service = session_service
        self._memory_service = memory_service
        self._plugins = plugins
        self._event_callback = event_callback
        self._before_agent_callbacks = before_agent_callbacks
        self._after_agent_callbacks = after_agent_callbacks
        self._max_retries = max_retries
        self._last_result: PipelineResult | None = None
        self._last_meta_result: MetaAgentResult | None = None
        self._resolved_workers: list[ModelConfig] | None = None

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

    async def generate_config(self, task: str) -> PipelineConfig:
        """Ask the meta-agent to design a pipeline for *task*.

        Returns a ``PipelineConfig`` that can be inspected, serialised to
        JSON for human review, and optionally edited before execution.

        When ``two_stage=True`` the generation is split into two LLM calls:
        1. **Pool generation** — produces a list of agents (no wiring).
        2. **Pipeline generation** — wires the pool into a StepConfig tree.
        """
        _log.info(
            "Generating pipeline config for task (two_stage={}): {}",
            self._two_stage,
            task,
        )

        if self._two_stage:
            meta_result = await self._generate_two_stage(task)
        else:
            meta_result = await generate_pipeline_config(
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
        _log.info(
            "Config generated | agents={} pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        return config

    async def _generate_two_stage(self, task: str) -> MetaAgentResult:
        """Run pool generationthen pipeline generation."""
        from fedotmas.config.settings import (
            get_worker_models,
            resolve_model_config,
        )

        _log.info("Two-stage | stage 1/2: generating agent pool")
        pool_gen = PoolGenerator(
            meta_model=self._meta_model,
            worker_models=self._worker_models,
            temperature=self._temperature,
            mcp_registry=self._mcp_registry,
            session_service=self._session_service,
            max_retries=self._max_retries,
        )
        pool = await pool_gen.generate(task)

        _log.info(
            "Two-stage | stage 2/2: generating pipeline from {} agents",
            len(pool.agents),
        )
        pipeline_gen = PipelineGenerator(
            meta_model=self._meta_model,
            worker_models=self._worker_models,
            temperature=self._temperature,
            mcp_registry=self._mcp_registry,
            session_service=self._session_service,
            max_retries=self._max_retries,
        )
        config = await pipeline_gen.generate(task, pool)

        # Resolve workers for downstream use
        resolved_workers = (
            [resolve_model_config(m) for m in self._worker_models]
            if self._worker_models
            else [resolve_model_config(m) for m in get_worker_models()]
        )

        # Combine metrics from both stages
        pool_r = pool_gen.result
        pipe_r = pipeline_gen.result
        return MetaAgentResult(
            config=config,
            worker_models=resolved_workers,
            total_prompt_tokens=(pool_r.prompt_tokens if pool_r else 0)
            + (pipe_r.prompt_tokens if pipe_r else 0),
            total_completion_tokens=(pool_r.completion_tokens if pool_r else 0)
            + (pipe_r.completion_tokens if pipe_r else 0),
            elapsed=(pool_r.elapsed if pool_r else 0.0)
            + (pipe_r.elapsed if pipe_r else 0.0),
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
        _log.info("Building agent tree")
        worker_map = (
            {m.model: m for m in self._resolved_workers}
            if self._resolved_workers
            else None
        )
        agent = build(
            config,
            mcp_registry=self._mcp_registry,
            worker_models=worker_map,
            before_agent_callbacks=self._before_agent_callbacks,
            after_agent_callbacks=self._after_agent_callbacks,
        )
        print_tree(config)
        _log.info("Running pipeline")
        self._last_result = await run_pipeline(
            agent,
            user_query,
            session_service=self._session_service,
            memory_service=self._memory_service,
            plugins=self._plugins,
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
