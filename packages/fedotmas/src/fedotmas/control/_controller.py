from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.plugins import BasePlugin

from fedotmas.common.logging import get_logger
from fedotmas.control._iterable import IterableRun
from fedotmas.control._run import ControlledRun, RunError, extract_failed_agent_name
from fedotmas.control._strategy import Strategy, resolve_initial_state
from fedotmas.core.runner import run_pipeline
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import CheckpointPlugin
from fedotmas.plugins._skip_completed import SkipCompletedPlugin

_log = get_logger("fedotmas.control")


class Controller:
    """Manages controlled execution and resumption of MAW pipelines.

    Uses ``MAW.build`` to get the agent tree, then calls ``run_pipeline``
    directly with custom plugins for full control over execution.
    """

    def __init__(self, maw: MAW) -> None:
        self._maw = maw
        self._task: str | None = None
        self._last_run: ControlledRun | None = None

    async def run(self, task: str, config: MAWConfig | None = None) -> ControlledRun:
        """Generate (or use provided) config and execute the pipeline.

        Args:
            task: The user's task description.
            config: Optional pre-built config. If ``None``, generates one
                via ``maw.generate_config(task)``.
        """
        self._task = task

        if config is None:
            config = await self._maw.generate_config(task)

        agent = self._maw.build(config)
        self._last_run = await self._execute(agent, task, config, plugins=[])
        return self._last_run

    @asynccontextmanager
    async def iter(
        self, task: str, config: MAWConfig | None = None
    ) -> AsyncIterator[IterableRun]:
        """Iterate over pipeline steps with pause before each.

        Args:
            task: The user's task description.
            config: Optional pre-built config. If ``None``, generates one
                via ``maw.generate_config(task)``.

        Usage::

            async with ctrl.iter("Analyse market") as run:
                async for step in run:
                    print(step.name, step.state)
                result = run.result
        """
        self._task = task
        if config is None:
            config = await self._maw.generate_config(task)
        run = IterableRun(self._maw, config, task)
        try:
            yield run
        finally:
            await run._cleanup()
            if run._result is not None:
                self._last_run = run._result

    async def run_with_recovery(
        self,
        task: str,
        *,
        max_retries: int = 2,
        config: MAWConfig | None = None,
        llm_error_detection: bool = False,
        error_hint: str | None = None,
    ) -> ControlledRun:
        """Run a pipeline with automatic recovery on agent failures.

        On error, diagnoses the failing agent via an LLM meta-call, patches
        its config, and resumes the pipeline.  Repeats up to *max_retries*
        times.

        Args:
            task: The user's task description.
            max_retries: Maximum number of recovery attempts.
            config: Optional pre-built config.
            llm_error_detection: When ``True``, use an LLM call to classify
                errors as retryable/fatal before attempting recovery.
                When ``False`` (default), use regex-based heuristics.
            error_hint: Free-text hint passed to the LLM error classifier
                (only used when ``llm_error_detection=True``).
        """
        from fedotmas.meta.maw_debugger import classify_error, diagnose_and_fix

        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        if error_hint and not llm_error_detection:
            _log.warning("error_hint is ignored when llm_error_detection=False")

        run = await self.run(task, config=config)

        for attempt in range(max_retries):
            if run.status == "success":
                return run

            error = run.error
            if error is None:
                return run

            # Cannot recover if we don't know which agent failed
            if not _has_known_agent(error, run.config):
                _log.warning(
                    "Cannot identify failed agent from error, skipping recovery: {}",
                    error.message,
                )
                return run

            error_category: str | None = None

            if llm_error_detection:
                classification = await classify_error(
                    error=error,
                    error_hint=error_hint,
                    meta_model=self._maw.meta_model,
                    session_service=self._maw._session_service,
                )
                if not classification.retryable:
                    _log.warning(
                        "LLM classified as non-retryable: {} ({})",
                        classification.category,
                        classification.reasoning,
                    )
                    return run
                error_category = classification.category
            else:
                if not _is_retryable(error):
                    _log.warning("Non-retryable error: {}", error.message)
                    return run

            _log.info(
                "Recovery attempt {}/{}: fixing '{}'",
                attempt + 1,
                max_retries,
                error.agent_name,
            )

            temperature = self._maw.temperature
            fixed_agent = await diagnose_and_fix(
                error=error,
                config=run.config,
                task=task,
                state=run.state,
                meta_model=self._maw.meta_model,
                temperature=temperature if temperature is not None else 0.3,
                mcp_registry=self._maw.mcp_registry,
                worker_models=self._maw.worker_models,
                session_service=self._maw._session_service,
                error_category=error_category,
            )

            new_config = run.config.replace_agent(error.agent_name, fixed_agent)
            run = await self.resume(new_config, strategy=Strategy.RETRY_FAILED)

        return run

    async def resume(
        self,
        new_config: MAWConfig,
        *,
        strategy: Strategy | None = None,
    ) -> ControlledRun:
        """Resume execution with a modified config.

        Args:
            new_config: The modified pipeline configuration.
            strategy: How to seed state from previous checkpoints.
                Defaults to ``RESTART_AFTER``.
        """
        if self._last_run is None or self._task is None:
            raise RuntimeError("No previous run to resume from. Call run() first")

        strategy = strategy or Strategy.RESTART_AFTER

        initial_state, completed = resolve_initial_state(
            strategy, self._last_run.checkpoints, self._last_run.config, new_config
        )

        extra_plugins: list[BasePlugin] = []
        if completed:
            extra_plugins.append(SkipCompletedPlugin(completed))

        agent = self._maw.build(new_config)
        self._last_run = await self._execute(
            agent,
            self._task,
            new_config,
            plugins=extra_plugins,
            initial_state=initial_state,
        )
        return self._last_run

    async def _execute(
        self,
        agent: BaseAgent,
        task: str,
        config: MAWConfig,
        *,
        plugins: list[BasePlugin],
        initial_state: dict[str, Any] | None = None,
    ) -> ControlledRun:
        checkpoint = CheckpointPlugin()
        all_plugins = [checkpoint, *plugins]

        try:
            result = await run_pipeline(
                agent,
                task,
                plugins=all_plugins,
                session_service=self._maw._session_service,
                memory_service=self._maw._memory_service,
                initial_state=initial_state,
            )
        except Exception as exc:
            msg = str(exc)
            agent_name = extract_failed_agent_name(msg)
            error_state = (
                dict(checkpoint.checkpoints[-1].state) if checkpoint.checkpoints else {}
            )
            return ControlledRun(
                config=config,
                status="error",
                state=error_state,
                checkpoints=checkpoint.checkpoints,
                error=RunError(agent_name=agent_name, message=msg),
            )

        return ControlledRun(
            config=config,
            status="success",
            state=result.state,
            checkpoints=checkpoint.checkpoints,
        )


_FATAL_ERROR_RE = re.compile(
    r"\bconnection\b|\btimeout\b|\bauth\b|\bunauthorized\b|\brate[.\s_]?limit\b|\bquota\b",
    re.IGNORECASE,
)


def _is_retryable(error: RunError | None) -> bool:
    """Check whether an error is retryable using regex heuristics.

    Returns ``False`` for infrastructure/auth errors (fatal),
    ``True`` for everything else (retryable).
    """
    if error is None:
        return False
    return not bool(_FATAL_ERROR_RE.search(error.message))


def _has_known_agent(error: RunError, config: MAWConfig) -> bool:
    """Check that the failed agent actually exists in the config."""
    if error.agent_name == "unknown":
        return False
    return any(a.name == error.agent_name for a in config.agents)
