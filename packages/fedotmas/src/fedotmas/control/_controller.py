from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.plugins import BasePlugin
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from fedotmas.common.llm import make_llm
from fedotmas.common.logging import get_logger
from fedotmas.control._iterable import IterableRun
from fedotmas.control._run import ControlledRun, RunError, extract_failed_agent_name
from fedotmas.control._strategy import Strategy, resolve_initial_state
from fedotmas.core.runner import run_pipeline
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import CheckpointPlugin
from fedotmas.plugins._eval import CheckFn, EvalPlugin
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

    async def run(
        self,
        task: str,
        config: MAWConfig | None = None,
        plugins: list[BasePlugin] | None = None,
    ) -> ControlledRun:
        """Generate (or use provided) config and execute the pipeline.

        Args:
            task: The user's task description.
            config: Optional pre-built config. If ``None``, generates one
                via ``maw.generate_config(task)``.
            plugins: Extra ADK plugins injected into the pipeline execution.
        """
        self._task = task

        if config is None:
            config = await self._maw.generate_config(task)

        agent = self._maw.build(config)
        self._last_run = await self._execute(
            agent, task, config, plugins=list(plugins or [])
        )
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
        fix_tools: list[Callable] | None = None,
        checks: dict[str, CheckFn] | None = None,
        strategy: Strategy = Strategy.RESTART_AFTER,
        llm_error_detection: bool = False,
        error_hint: str | None = None,
    ) -> ControlledRun:
        """Run a pipeline with automatic recovery on agent failures.

        On error, launches a debugger LLM agent that uses *fix_tools* to
        patch the config, then resumes the pipeline.  Repeats up to
        *max_retries* times.

        Args:
            task: The user's task description.
            max_retries: Maximum number of recovery attempts.
            config: Optional pre-built config.
            fix_tools: ADK-compatible tool functions the debugger agent can
                call.  Each tool reads/writes config via
                ``tool_context.state["config"]``.  Defaults to
                ``[fix_instruction]``.
            checks: Per-agent check functions.  Each function receives the
                pipeline state dict and returns an error message (str) if
                the output is wrong, or ``None`` if OK.  The error message
                is passed to the debugger as the problem description.
            strategy: Resume strategy after debugger fixes the config.
                Defaults to ``RESTART_AFTER`` which re-runs only the
                modified agent and everything after it.
            llm_error_detection: When ``True``, use an LLM call to classify
                errors as retryable/fatal before attempting recovery.
                When ``False`` (default), use regex-based heuristics.
            error_hint: Free-text hint passed to the debugger agent to
                provide additional context about expected behavior.
        """
        from fedotmas.meta.maw_debugger import classify_error, evaluate_output

        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        plugins: list[BasePlugin] = []
        if checks:
            plugins.append(EvalPlugin(checks))

        run = await self.run(task, config=config, plugins=plugins)

        for attempt in range(max_retries + 1):
            if run.status == "success":
                if error_hint:
                    eval_result = await evaluate_output(
                        state=run.state,
                        config=run.config,
                        error_hint=error_hint,
                        meta_model=self._maw.meta_model,
                        session_service=self._maw._session_service,
                    )
                    if not eval_result.passed:
                        _log.info(
                            "LLM eval failed: {} (agent={})",
                            eval_result.reasoning,
                            eval_result.agent_name,
                        )
                        run = ControlledRun(
                            config=run.config,
                            status="error",
                            state=run.state,
                            checkpoints=run.checkpoints,
                            error=RunError(
                                agent_name=eval_result.agent_name or "unknown",
                                message=eval_result.reasoning,
                            ),
                        )
                    else:
                        return run
                else:
                    return run

            if attempt >= max_retries:
                return run

            error = run.error
            if error is None:
                return run

            if not _has_known_agent(error, run.config):
                _log.warning(
                    "Cannot identify failed agent from error, skipping recovery: {}",
                    error.message,
                )
                return run

            if llm_error_detection:
                classification = await classify_error(
                    error=error,
                    config=run.config,
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

            new_config = await self._run_debugger(
                task=task,
                error=error,
                current_config=run.config,
                state=run.state,
                fix_tools=fix_tools,
                error_hint=error_hint,
            )

            run = await self.resume(new_config, strategy=strategy)

        return run

    async def _run_debugger(
        self,
        *,
        task: str,
        error: RunError,
        current_config: MAWConfig,
        state: dict[str, Any],
        fix_tools: list[Callable] | None = None,
        error_hint: str | None = None,
    ) -> MAWConfig:
        """Launch a debugger LLM agent with fix tools to patch the config."""
        from fedotmas.control.fixes import fix_instruction, guardrail_validate_config
        from fedotmas.meta._helpers import resolve_meta_and_workers
        from fedotmas.meta.maw_debug_prompts import DEBUGGER_TOOL_PROMPT

        tools = fix_tools or [fix_instruction]
        resolved_meta, _, _ = resolve_meta_and_workers(
            self._maw.meta_model,
            None,
            None,
        )

        config_json = current_config.model_dump_json(indent=2)
        if len(config_json) > 6000:
            config_json = config_json[:6000] + "... (truncated)"

        state_str = json.dumps(state, default=str, ensure_ascii=False)
        if len(state_str) > 4000:
            state_str = state_str[:4000] + "... (truncated)"

        error_message = error.message
        if len(error_message) > 2000:
            error_message = error_message[:2000] + "... (truncated)"

        instruction = DEBUGGER_TOOL_PROMPT.substitute(
            task=task,
            agent_name=error.agent_name,
            error_message=error_message,
            config_json=config_json,
            state_snapshot=state_str,
        )
        if error_hint:
            instruction += f"\n\n**User hint:** {error_hint}\n"

        llm = make_llm(resolved_meta)

        debugger_agent = LlmAgent(
            name="debugger",
            model=llm,
            instruction=instruction,
            tools=[FunctionTool(func=f) for f in tools],
            after_tool_callback=guardrail_validate_config,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.3,
            ),
        )

        session_service = self._maw._session_service or InMemorySessionService()
        session_id = uuid.uuid4().hex
        app_name = "fedotmas_debugger"

        session = await session_service.create_session(
            app_name=app_name,
            user_id="system",
            session_id=session_id,
            state={"config": current_config.model_dump_json()},
        )

        message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Fix agent '{error.agent_name}' that failed with: {error_message}",
                )
            ],
        )

        start = time.monotonic()

        async with Runner(
            app_name=app_name,
            agent=debugger_agent,
            session_service=session_service,
        ) as runner:
            async for event in runner.run_async(
                user_id="system",
                session_id=session.id,
                new_message=message,
            ):
                if event.partial:
                    continue
                if event.error_code:
                    _log.error(
                        "Debugger LLM error | code={} msg={}",
                        event.error_code,
                        event.error_message,
                    )
                    raise RuntimeError(
                        f"Debugger LLM error {event.error_code}: {event.error_message}"
                    )

        elapsed = time.monotonic() - start
        _log.info("Debugger complete | elapsed={:.1f}s", elapsed)

        final_session = await session_service.get_session(
            app_name=app_name,
            user_id="system",
            session_id=session.id,
        )
        if final_session is None:
            raise RuntimeError("Debugger session lost after execution")

        config_raw = final_session.state.get("config")
        if config_raw is None:
            raise RuntimeError("Debugger did not produce config in session state")

        if isinstance(config_raw, str):
            new_config = MAWConfig.model_validate_json(config_raw)
        else:
            new_config = MAWConfig.model_validate(config_raw)
        _log.info("Debugger produced new config | agents={}", len(new_config.agents))
        return new_config

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
