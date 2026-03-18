from __future__ import annotations

import re

from fedotmas.common.logging import get_logger
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

        checkpoint = CheckpointPlugin()
        agent = self._maw.build(config)

        try:
            result = await run_pipeline(
                agent,
                task,
                plugins=[checkpoint],
                session_service=self._maw._session_service,
                memory_service=self._maw._memory_service,
            )
            self._last_run = ControlledRun(
                config=config,
                status="success",
                state=result.state,
                checkpoints=checkpoint.checkpoints,
            )
        except RuntimeError as exc:
            msg = str(exc)
            match = _AGENT_ERROR_RE.search(msg)
            agent_name = match.group(1) if match else "unknown"
            error_state = (
                dict(checkpoint.checkpoints[-1].state) if checkpoint.checkpoints else {}
            )
            self._last_run = ControlledRun(
                config=config,
                status="error",
                state=error_state,
                checkpoints=checkpoint.checkpoints,
                error=RunError(agent_name=agent_name, message=msg),
            )

        return self._last_run

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
                Defaults to ``RETRY_FAILED``.
        """
        if self._last_run is None or self._task is None:
            raise RuntimeError("No previous run to resume from — call run() first")

        strategy = strategy or Strategy.RETRY_FAILED
        old_config = self._last_run.config
        checkpoints = self._last_run.checkpoints

        initial_state, completed = resolve_initial_state(
            strategy, checkpoints, old_config, new_config
        )

        checkpoint = CheckpointPlugin()
        plugins = [checkpoint]
        if completed:
            plugins.append(SkipCompletedPlugin(completed))

        agent = self._maw.build(new_config)

        try:
            result = await run_pipeline(
                agent,
                self._task,
                plugins=plugins,
                session_service=self._maw._session_service,
                memory_service=self._maw._memory_service,
                initial_state=initial_state,
            )
            self._last_run = ControlledRun(
                config=new_config,
                status="success",
                state=result.state,
                checkpoints=checkpoint.checkpoints,
            )
        except RuntimeError as exc:
            msg = str(exc)
            match = _AGENT_ERROR_RE.search(msg)
            agent_name = match.group(1) if match else "unknown"
            error_state = (
                dict(checkpoint.checkpoints[-1].state) if checkpoint.checkpoints else {}
            )
            self._last_run = ControlledRun(
                config=new_config,
                status="error",
                state=error_state,
                checkpoints=checkpoint.checkpoints,
                error=RunError(agent_name=agent_name, message=msg),
            )

        return self._last_run
