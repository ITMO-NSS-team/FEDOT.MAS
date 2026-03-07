from __future__ import annotations

from google.adk.agents.base_agent import BaseAgent

from fedotmas.common.logging import get_logger
from fedotmas._settings import resolve_model_config
from fedotmas.core.base import BaseMAS
from fedotmas.maw._ppline_utils import print_tree
from fedotmas.maw.builder import build
from fedotmas.maw.models import MAWConfig
from fedotmas.meta.agent import MetaAgentResult, generate_pipeline_config
from fedotmas.meta.pipeline_gen import PipelineGenerator
from fedotmas.meta.pool_gen import PoolGenerator

_log = get_logger("fedotmas.maw")


class MAW(BaseMAS[MAWConfig]):
    """Multi-Agent Workflow are fixed pipeline orchestration.

    Generates and executes workflow pipelines using Sequential, Parallel,
    and Loop structures for deterministic agent orchestration.

    Args:
        two_stage: When ``True`` (default), pipeline generation is split into
            two LLM calls — first an agent pool is generated, then the
            pipeline tree is designed around that pool.
        **kwargs: Passed to ``BaseMAS.__init__``.

    Usage::

        maw = MAW()
        result = await maw.run("Research quantum computing trends")

        # Two-step with review:
        config = await maw.generate_config("Research quantum computing trends")
        result = await maw.build_and_run(config, "Research quantum computing trends")
    """

    def __init__(self, *, two_stage: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._two_stage = two_stage

    async def generate_config(self, task: str) -> MAWConfig:
        """Ask the meta-agent to design a pipeline for *task*.

        Returns a ``MAWConfig`` that can be inspected, serialised to
        JSON for human review, and optionally edited before execution.
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
        assert isinstance(config, MAWConfig)
        _log.info(
            "Config generated | agents={} pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        return config

    async def _generate_two_stage(self, task: str) -> MetaAgentResult:
        """Run pool generation then pipeline generation."""
        from fedotmas._settings import get_worker_models

        _log.info("Stage 1/2: generating agent pool")
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
            "Stage 2/2: generating pipeline from {} agents",
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

        sources = self._worker_models or get_worker_models()
        resolved_workers = [resolve_model_config(m) for m in sources]

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

    def build(self, config: MAWConfig) -> BaseAgent:
        """Build an ADK agent tree from *config*."""
        _log.info("Building agent tree")
        agent = build(
            config,
            mcp_registry=self._mcp_registry,
            worker_models=self._worker_map(),
        )
        print_tree(config)
        return agent
