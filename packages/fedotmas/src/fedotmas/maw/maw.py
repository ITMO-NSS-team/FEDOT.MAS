from __future__ import annotations

from typing import Literal

from google.adk.agents.base_agent import BaseAgent

from fedotmas.common.logging import get_logger
from fedotmas._settings import resolve_model_config, validate_model_name
from fedotmas.core.base import BaseMAS
from fedotmas.maw.builder import _STATE_REF_RE, build
from fedotmas.maw.models import AgentPoolConfig, MAWAgentConfig, MAWConfig
from fedotmas.meta._result import MetaAgentResult
from fedotmas.meta.maw_single_stage import generate_pipeline_config
from fedotmas.meta.maw_pipeline_stage import PipelineGenerator
from fedotmas.meta.maw_pool_stage import PoolGenerator

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

    async def generate_config(
        self,
        task: str,
        *,
        existing_agents: AgentPoolConfig | None = None,
        reuse: Literal["prefer", "only"] = "prefer",
    ) -> MAWConfig:
        """Ask the meta-agent to design a pipeline for *task*.

        Returns a ``MAWConfig`` that can be inspected, serialised to
        JSON for human review, and optionally edited before execution.

        Args:
            existing_agents: Agents the caller already has. ``reuse="prefer"``
                lets the meta-agent add roles they do not cover; ``"only"``
                confines the pipeline to them. Either way their instructions,
                models and tools survive generation unchanged — they are records
                in someone else's system, to be wired rather than rewritten.
            reuse: Ignored when *existing_agents* is ``None``.
        """
        if reuse not in ("prefer", "only"):
            raise ValueError(f"reuse must be 'prefer' or 'only', got {reuse!r}")
        if existing_agents is not None and not existing_agents.agents:
            if reuse == "only":
                raise ValueError(
                    "reuse='only' confines the pipeline to existing_agents, but "
                    "the pool is empty. Pass agents, or use reuse='prefer'."
                )
            existing_agents = None  # nothing to reuse; an ordinary generation
        if existing_agents is not None:
            # MAWAgentConfig rejects model names AgentPoolEntry accepts, and
            # would only do so after both LLM calls.
            for entry in existing_agents.agents:
                validate_model_name(entry.model)

        _log.info(
            "Generating pipeline config for task (two_stage={}, existing={}): {}",
            self._two_stage,
            len(existing_agents.agents) if existing_agents else 0,
            task,
        )

        if existing_agents is not None and reuse == "only":
            meta_result = await self._generate_from_pool(task, existing_agents)
        # A caller-supplied pool has no place in the single-stage prompt, so it
        # forces the staged path regardless of how this instance was built.
        elif existing_agents is not None or self._two_stage:
            meta_result = await self._generate_two_stage(task, existing_agents)
        else:
            meta_result = await generate_pipeline_config(
                task,
                meta_model=self._meta_model,
                worker_models=self._worker_models,
                temperature=self._temperature,
                mcp_registry=self._mcp_registry,
                tool_catalog=self._tool_catalog,
                session_service=self._session_service,
                max_retries=self._max_retries,
                plugins=self._plugins,
            )

        self._last_meta_result = meta_result
        self._resolved_workers = meta_result.worker_models
        config = meta_result.config
        assert isinstance(config, MAWConfig)
        _drop_generated_token_budgets(config)
        if existing_agents is not None:
            config = _restore_external_agents(config, existing_agents)
        _log.info(
            "Config generated | agents={} pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        return config

    async def _generate_two_stage(
        self, task: str, existing: AgentPoolConfig | None = None
    ) -> MetaAgentResult:
        """Run pool generation then pipeline generation."""
        from fedotmas._settings import get_worker_models

        _log.info("Stage 1/2: generating agent pool")
        pool_gen = PoolGenerator(
            meta_model=self._meta_model,
            worker_models=self._worker_models,
            temperature=self._temperature,
            mcp_registry=self._mcp_registry,
            tool_catalog=self._tool_catalog,
            session_service=self._session_service,
            max_retries=self._max_retries,
            plugins=self._plugins,
        )
        pool = await pool_gen.generate(
            task, _pool_for_prompt(existing) if existing else None
        )
        if existing is not None:
            pool = _restore_pool_agents(pool, existing)

        _log.info(
            "Stage 2/2: generating pipeline from {} agents",
            len(pool.agents),
        )
        pipeline_gen = self._pipeline_generator()
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

    def _pipeline_generator(self) -> PipelineGenerator:
        return PipelineGenerator(
            meta_model=self._meta_model,
            worker_models=self._worker_models,
            temperature=self._temperature,
            mcp_registry=self._mcp_registry,
            tool_catalog=self._tool_catalog,
            session_service=self._session_service,
            max_retries=self._max_retries,
            plugins=self._plugins,
        )

    async def _generate_from_pool(
        self, task: str, pool: AgentPoolConfig
    ) -> MetaAgentResult:
        """Design a pipeline over *pool* alone, skipping pool generation."""
        from fedotmas._settings import get_worker_models

        _log.info(
            "Generating pipeline from {} caller-supplied agents", len(pool.agents)
        )
        pipeline_gen = self._pipeline_generator()
        config = await pipeline_gen.generate(task, _pool_for_prompt(pool))

        sources = self._worker_models or get_worker_models()
        result = pipeline_gen.result
        return MetaAgentResult(
            config=config,
            worker_models=[resolve_model_config(m) for m in sources],
            total_prompt_tokens=result.prompt_tokens if result else 0,
            total_completion_tokens=result.completion_tokens if result else 0,
            elapsed=result.elapsed if result else 0.0,
        )

    def build(self, config: MAWConfig, *, autonomous: bool = True) -> BaseAgent:
        """Build an ADK agent tree from *config*."""
        self._reject_external_build()
        _log.info("Building agent tree")
        agent = build(
            config,
            mcp_registry=self._mcp_registry,
            worker_models=self._worker_map(),
            autonomous=autonomous,
        )
        _log.info("Config:\n{}", config)
        return agent


def _pool_for_prompt(pool: AgentPoolConfig) -> AgentPoolConfig:
    """Drop models before the pool is shown to the meta-agent.

    The prompt demands a model from this instance's worker list and the call
    rejects anything else, so a caller's own model names would fail generation
    outright.  ``_restore_external_agents`` puts them back afterwards.
    """
    return AgentPoolConfig(
        agents=[a.model_copy(update={"model": None}) for a in pool.agents]
    )


def _restore_pool_agents(
    pool: AgentPoolConfig, existing: AgentPoolConfig
) -> AgentPoolConfig:
    """Undo stage 1's edits to a supplied agent before stage 2 sees the pool.

    Stage 2 designs the topology around what it is shown, and that outlives the
    final restore: rewritten here, an agent comes back as the caller wrote it
    inside a pipeline built for somebody else.  Models are left as stage 1 has
    them — ``_pool_for_prompt`` drops the caller's on purpose, and the pipeline
    prompt requires one this instance owns.
    """
    originals = {a.name: a for a in existing.agents}
    return AgentPoolConfig(
        agents=[
            a.model_copy(
                update={
                    "instruction": originals[a.name].instruction,
                    "tools": list(originals[a.name].tools),
                }
            )
            if a.name in originals
            else a
            for a in pool.agents
        ]
    )


def _keep_added_state_refs(original: str, generated: str) -> str:
    """Return *original* carrying whatever state references *generated* added."""
    present = set(_STATE_REF_RE.findall(original))
    added = [
        ref
        for ref in dict.fromkeys(_STATE_REF_RE.findall(generated))
        if ref not in present
    ]
    if not added:
        return original
    refs = "\n".join(f"{{{ref}?}}" for ref in added)
    return f"{original}\n\nInput from earlier steps:\n{refs}"


def _restore_external_agents(config: MAWConfig, pool: AgentPoolConfig) -> MAWConfig:
    """Undo edits the meta-agent made to an agent it was handed.

    A caller-supplied agent is a record in someone else's system: it may be
    selected and wired, not rewritten.  What survives generation is wiring, not
    content: ``output_key``, which the pool entry does not carry; the state
    references stage 2 wrote into the instruction, without which a reused agent
    reads nothing (``builder`` installs the state provider only for an
    instruction that carries one); and an assigned model where the caller named
    none, the alternative being no model at all.
    """
    originals = {a.name: a for a in pool.agents}
    reused = [a.name for a in config.agents if a.name in originals]
    unmatched = sorted(originals.keys() - {a.name for a in config.agents})
    if unmatched:
        # Either the task had no use for them or the meta-agent renamed them;
        # the two are indistinguishable here, and only the second is a problem.
        _log.info("Supplied agents absent from the config: {}", unmatched)
    if not reused:
        return config

    agents = [
        MAWAgentConfig.model_validate(
            {
                **a.model_dump(),
                "instruction": _keep_added_state_refs(
                    originals[a.name].instruction, a.instruction
                ),
                "model": originals[a.name].model or a.model,
                "tools": list(originals[a.name].tools),
            }
        )
        if a.name in originals
        else a
        for a in config.agents
    ]
    _log.info(
        "Agents reused as given: {} | synthesized: {}",
        len(reused),
        len(config.agents) - len(reused),
    )
    return config.model_copy(update={"agents": agents})


def _drop_generated_token_budgets(config: MAWConfig) -> None:
    """Clear ``max_output_tokens`` on a freshly generated config.

    ``maw_prompts.py`` never asks for the field, so a value arriving in it is
    the meta-agent filling in the JSON schema, not sizing the step.  Treating
    small values as too small and raising them to a floor made that floor the
    cap every agent landed on, and a whole run came back cut off mid-sentence.
    No threshold fixes that, because the number was never an estimate at any
    magnitude: 4000 deserves no more trust than 2000 does.  Drop it and let the
    provider default apply.  A hand-written config never reaches here -- an
    explicit cap means what it says.
    """
    for agent in config.agents:
        if agent.max_output_tokens is None:
            continue
        _log.warning(
            "Dropping generated max_output_tokens for '{}' ({}): nothing asked "
            "the meta-agent for this number; using the provider default",
            agent.name,
            agent.max_output_tokens,
        )
        agent.max_output_tokens = None
