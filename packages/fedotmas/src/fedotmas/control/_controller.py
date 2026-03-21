from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.plugins import BasePlugin

from fedotmas.common.logging import get_logger
from fedotmas.control._iterable import IterableRun
from fedotmas.control._run import ControlledRun, RunError
from fedotmas.control._strategy import Strategy, resolve_initial_state
from fedotmas.core.runner import run_pipeline
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import CheckpointPlugin
from fedotmas.plugins._skip_completed import SkipCompletedPlugin

_log = get_logger("fedotmas.control")

_AGENT_ERROR_RE = re.compile(r"Agent '(.+?)' failed")


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
    ) -> ControlledRun:
        """Auto-recovery with meta-debugger. Not yet implemented."""
        raise NotImplementedError(
            "run_with_recovery requires meta-debugger (coming soon)"
        )

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
        except RuntimeError as exc:
            msg = str(exc)
            match = _AGENT_ERROR_RE.search(msg)
            agent_name = match.group(1) if match else "unknown"
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
