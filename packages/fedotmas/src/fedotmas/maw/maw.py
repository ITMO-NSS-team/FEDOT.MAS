from __future__ import annotations

from google.adk.agents.base_agent import BaseAgent

from fedotmas.common.logging import get_logger
from fedotmas._settings import resolve_model_config
from fedotmas.core.base import BaseMAS
from fedotmas.maw.builder import build
from fedotmas.maw.models import MAWConfig
from fedotmas.meta._result import MetaAgentResult
from fedotmas.meta.maw_single_stage import generate_pipeline_config
from fedotmas.meta.maw_pipeline_stage import PipelineGenerator
from fedotmas.meta.maw_pool_stage import PoolGenerator

_log = get_logger("fedotmas.maw")

#: Floor applied to a *generated* worker's ``max_output_tokens``.  Reasoning
#: models spend this budget thinking before they emit anything, so too small a
#: cap yields a turn with no content at all -- reported as MAX_TOKENS, which
#: ADK surfaces as an error rather than a short answer.  The meta-agent picks
#: the number itself and has picked 2000, below what the default workers need.
#: A hand-written config is left alone: an explicit cap means what it says.
MIN_GENERATED_MAX_OUTPUT_TOKENS = 4000


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
                plugins=self._plugins,
            )

        self._last_meta_result = meta_result
        self._resolved_workers = meta_result.worker_models
        config = meta_result.config
        assert isinstance(config, MAWConfig)
        _raise_thin_token_budgets(config)
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
            plugins=self._plugins,
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
            plugins=self._plugins,
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
        _log.info("Config:\n{}", config)
        return agent


def _raise_thin_token_budgets(config: MAWConfig) -> None:
    """Lift generated ``max_output_tokens`` values that cannot produce output."""
    for agent in config.agents:
        if agent.max_output_tokens is None:
            continue
        if agent.max_output_tokens >= MIN_GENERATED_MAX_OUTPUT_TOKENS:
            continue
        _log.warning(
            "Raising generated max_output_tokens for '{}' from {} to {}",
            agent.name,
            agent.max_output_tokens,
            MIN_GENERATED_MAX_OUTPUT_TOKENS,
        )
        agent.max_output_tokens = MIN_GENERATED_MAX_OUTPUT_TOKENS
