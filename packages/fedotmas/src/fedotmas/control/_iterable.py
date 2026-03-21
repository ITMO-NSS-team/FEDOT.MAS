from __future__ import annotations

import asyncio
import re
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin
from google.genai import types

from fedotmas.common.logging import get_logger
from fedotmas.control._run import ControlledRun, PipelineStep, RunError
from fedotmas.core.runner import run_pipeline
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import Checkpoint, CheckpointPlugin

_log = get_logger("fedotmas.control.iterable")

_AGENT_ERROR_RE = re.compile(r"Agent '(.+?)' failed")


class _StepPlugin(BasePlugin):
    """Pauses pipeline execution before each top-level step."""

    def __init__(self, pause_names: set[str]) -> None:
        super().__init__(name="fedotmas_step")
        self._pause_names = pause_names
        self._step_queue: asyncio.Queue[str] = asyncio.Queue()
        self._resume = asyncio.Event()
        self._pausing = True

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        if not self._pausing:
            return None
        if agent.name in self._pause_names:
            await self._step_queue.put(agent.name)
            await self._resume.wait()
            self._resume.clear()
        return None


class IterableRun:
    """Handle for iterating over pipeline steps.

    Created by ``Controller.iter``. Yields a ``PipelineStep``
    before each top-level step executes.
    """

    def __init__(self, maw: MAW, config: MAWConfig, task: str) -> None:
        self._maw = maw
        self._config = config
        self._task = task
        self._checkpoint = CheckpointPlugin()
        self._plugin: _StepPlugin | None = None
        self._plugin_ready = asyncio.Event()
        self._exec_task: asyncio.Task[ControlledRun] | None = None
        self._result: ControlledRun | None = None
        self._step_index: int = 0
        self._done: bool = False
        self._agents_by_name = {a.name: a for a in config.agents}

    @property
    def state(self) -> dict[str, Any]:
        """State from the last completed checkpoint."""
        if self._checkpoint.checkpoints:
            return dict(self._checkpoint.checkpoints[-1].state)
        return {}

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return self._checkpoint.checkpoints

    @property
    def result(self) -> ControlledRun:
        """Final result. Available after iteration completes."""
        if self._result is None:
            raise RuntimeError("Pipeline has not completed yet")
        return self._result

    def __aiter__(self) -> IterableRun:
        return self

    async def __anext__(self) -> PipelineStep:
        """Yield the next step (before it executes)."""
        if self._done:
            raise StopAsyncIteration

        # Resume previous step (if any)
        if self._step_index > 0 and self._plugin is not None:
            self._plugin._resume.set()

        # Start pipeline on first call
        if self._exec_task is None:
            self._exec_task = asyncio.create_task(self._run())

        # Wait for _run() to create the plugin before using it
        await self._plugin_ready.wait()
        if self._plugin is None:
            # _run() failed during setup (e.g. build() error)
            self._done = True
            exc = self._exec_task.exception()
            self._result = ControlledRun(
                config=self._config,
                status="error",
                state={},
                error=RunError(agent_name="build", message=str(exc)),
            )
            raise StopAsyncIteration

        # Race: next step arrives vs pipeline completes
        queue_task = asyncio.ensure_future(self._plugin._step_queue.get())
        done, pending = await asyncio.wait(
            [queue_task, self._exec_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if queue_task in done:
            step_name = queue_task.result()
            state = (
                dict(self._checkpoint.checkpoints[-1].state)
                if self._checkpoint.checkpoints
                else {}
            )
            agent_cfg = self._agents_by_name.get(step_name)
            step = PipelineStep(
                name=step_name, index=self._step_index, state=state, agent=agent_cfg
            )
            self._step_index += 1
            return step

        # Pipeline completed before next pause
        for t in pending:
            t.cancel()
        self._done = True
        try:
            self._result = self._exec_task.result()
        except Exception as exc:
            self._result = ControlledRun(
                config=self._config,
                status="error",
                state=self.state,
                error=RunError(agent_name="unknown", message=str(exc)),
            )
        raise StopAsyncIteration

    async def finish(self) -> ControlledRun:
        """Execute remaining steps without pausing."""
        if self._exec_task is None:
            raise RuntimeError("No execution in progress")
        if self._plugin is not None:
            self._plugin._pausing = False
            self._plugin._resume.set()
        try:
            self._result = await self._exec_task
        except Exception as exc:
            _log.warning("Pipeline failed during finish: {}", exc)
            self._result = ControlledRun(
                config=self._config,
                status="error",
                state=self.state,
                error=RunError(agent_name="unknown", message=str(exc)),
            )
        self._done = True
        return self._result

    async def _cleanup(self) -> None:
        """Release pause and wait for pipeline completion."""
        if self._exec_task is None or self._exec_task.done():
            return
        if self._plugin is not None:
            self._plugin._pausing = False
            self._plugin._resume.set()
        try:
            self._result = await self._exec_task
        except Exception as exc:
            _log.warning("Pipeline failed during cleanup: {}", exc)
            self._result = ControlledRun(
                config=self._config,
                status="error",
                state=self.state,
                error=RunError(agent_name="unknown", message=str(exc)),
            )

    async def _run(self) -> ControlledRun:
        try:
            agent = self._maw.build(self._config)
            if agent.sub_agents:
                pause_names = {child.name for child in agent.sub_agents}
            else:
                # Single-agent pipeline: the root IS the only step
                pause_names = {agent.name}
            self._plugin = _StepPlugin(pause_names)
        finally:
            self._plugin_ready.set()
        plugins = [self._checkpoint, self._plugin]

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
