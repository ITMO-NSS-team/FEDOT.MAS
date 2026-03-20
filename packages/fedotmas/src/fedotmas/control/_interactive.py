from __future__ import annotations

import asyncio
import re
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin
from google.genai import types

from fedotmas.common.logging import get_logger
from fedotmas.control._run import ControlledRun, RunError
from fedotmas.core.runner import run_pipeline
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import Checkpoint, CheckpointPlugin

_log = get_logger("fedotmas.control.interactive")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")
_AGENT_ERROR_RE = re.compile(r"Agent '(.+?)' failed")


class _PausePlugin(BasePlugin):
    """Pauses pipeline execution before a target agent."""

    def __init__(self) -> None:
        super().__init__(name="fedotmas_pause")
        self._target: str | None = None
        self._reached = asyncio.Event()
        self._resume = asyncio.Event()

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        if agent.name.startswith(_WORKFLOW_PREFIXES):
            return None

        if self._target and agent.name == self._target:
            _log.debug("Pausing before agent: {}", agent.name)
            self._reached.set()
            await self._resume.wait()
            self._resume.clear()

        return None


class InteractiveRun:
    """Handle for an interactive pipeline execution with pause/resume.

    Created by `Controller.run_interactive`. Provides
    `wait_until` to pause before a named agent and
    `continue_` to resume to completion.
    """

    def __init__(self, maw: MAW, config: MAWConfig, task: str) -> None:
        self._maw = maw
        self._config = config
        self._task = task
        self._pause = _PausePlugin()
        self._checkpoint = CheckpointPlugin()
        self._exec_task: asyncio.Task[ControlledRun] | None = None
        self._result: ControlledRun | None = None

    @property
    def state(self) -> dict[str, Any]:
        """State from the last completed checkpoint."""
        if self._checkpoint.checkpoints:
            return dict(self._checkpoint.checkpoints[-1].state)
        return {}

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return self._checkpoint.checkpoints

    async def wait_until(self, agent_name: str) -> None:
        """Run pipeline until just before *agent_name* and pause.

        After this returns, ``self.state`` reflects all agents
        that completed before the pause point.
        """
        self._pause._target = agent_name
        self._pause._reached.clear()

        if self._exec_task is None:
            self._exec_task = asyncio.create_task(self._run())
        else:
            self._pause._resume.set()

        await self._pause._reached.wait()

    async def continue_(self) -> ControlledRun:
        """Resume execution to completion and return the result."""
        if self._exec_task is None:
            raise RuntimeError("No execution in progress. Call wait_until() first")

        self._pause._resume.set()
        self._result = await self._exec_task
        return self._result

    async def _cleanup(self) -> None:
        """Release any pending pause and wait for pipeline to finish."""
        if self._exec_task is None or self._exec_task.done():
            return
        self._pause._target = None
        self._pause._resume.set()
        try:
            self._result = await self._exec_task
        except Exception:
            pass

    async def _run(self) -> ControlledRun:
        agent = self._maw.build(self._config)
        plugins = [self._checkpoint, self._pause]

        try:
            result = await run_pipeline(
                agent,
                self._task,
                plugins=plugins,
                session_service=self._maw._session_service,
                memory_service=self._maw._memory_service,
            )
        except RuntimeError as exc:
            msg = str(exc)
            match = _AGENT_ERROR_RE.search(msg)
            agent_name = match.group(1) if match else "unknown"
            error_state = (
                dict(self._checkpoint.checkpoints[-1].state)
                if self._checkpoint.checkpoints
                else {}
            )
            return ControlledRun(
                config=self._config,
                status="error",
                state=error_state,
                checkpoints=self._checkpoint.checkpoints,
                error=RunError(agent_name=agent_name, message=msg),
            )

        return ControlledRun(
            config=self._config,
            status="success",
            state=result.state,
            checkpoints=self._checkpoint.checkpoints,
        )
