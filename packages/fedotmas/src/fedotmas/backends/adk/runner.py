from __future__ import annotations

import time
import uuid
from typing import Any

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.plugins import BasePlugin
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.adk.memory import BaseMemoryService
from google.genai import types
from pydantic import BaseModel

from fedotmas._settings import ModelConfig
from fedotmas.backends.adk.builder import build_adk_tree, _resolve_llm, _build_tools
from fedotmas.backends.adk.middleware import MiddlewareAdapter
from fedotmas.common.llm import make_llm
from fedotmas.common.logging import get_logger
from fedotmas.interfaces.agent import AgentTree
from fedotmas.interfaces.middleware import MiddlewareProtocol
from fedotmas.interfaces.runner import PipelineResult, SingleAgentResult
from fedotmas.interfaces.tools import ToolDescriptor

_log = get_logger("fedotmas.backends.adk.runner")


class ADKRunner:
    """ADK implementation of :class:`RunnerProtocol`."""

    def __init__(
        self,
        *,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
    ) -> None:
        self._session_service = session_service
        self._memory_service = memory_service

    async def run_pipeline(
        self,
        agent_tree: AgentTree,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        middlewares: list[MiddlewareProtocol] | None = None,
        backend_plugins: list[Any] | None = None,
    ) -> PipelineResult:
        """Execute a full agent tree via ADK Runner."""
        adk_agent = build_adk_tree(agent_tree)

        plugins: list[BasePlugin] = []
        if middlewares:
            plugins.append(MiddlewareAdapter(middlewares))
        if backend_plugins:
            plugins.extend(backend_plugins)

        app = App(
            name="fedotmas",
            root_agent=adk_agent,
            plugins=plugins,
        )

        session_service = self._session_service or InMemorySessionService()
        session_id = uuid.uuid4().hex
        user_id = "user"

        state: dict[str, Any] = {"user_query": user_query}
        if initial_state:
            state.update(initial_state)

        session = await session_service.create_session(
            app_name="fedotmas",
            user_id=user_id,
            session_id=session_id,
            state=state,
        )

        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )

        _log.info("Pipeline run started | pipeline={}", adk_agent.name)
        total_prompt = 0
        total_completion = 0
        pipeline_start = time.monotonic()

        async with Runner(
            app=app,
            session_service=session_service,
            memory_service=self._memory_service,
        ) as runner:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=message,
            ):
                if event.partial:
                    continue

                if event.usage_metadata:
                    um = event.usage_metadata
                    total_prompt += um.prompt_token_count or 0
                    total_completion += um.candidates_token_count or 0

                if event.error_code:
                    _log.error(
                        "LLM error | agent={} code={} msg={}",
                        event.author,
                        event.error_code,
                        event.error_message,
                    )
                    raise RuntimeError(
                        f"Agent '{event.author}' failed with error {event.error_code}: "
                        f"{event.error_message}"
                    )

        total_elapsed = time.monotonic() - pipeline_start
        _log.info(
            "Pipeline complete | total_elapsed={:.1f}s total_prompt={} total_completion={}",
            total_elapsed,
            total_prompt,
            total_completion,
        )

        final_session = await session_service.get_session(
            app_name="fedotmas",
            user_id=user_id,
            session_id=session.id,
        )
        if final_session is None:
            raise RuntimeError(
                f"Session '{session.id}' lost after pipeline execution — results unavailable"
            )
        return PipelineResult(
            state=dict(final_session.state),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            elapsed=total_elapsed,
        )

    async def run_single_agent(
        self,
        *,
        agent_name: str,
        instruction: str,
        user_message: str,
        model: str | ModelConfig,
        temperature: float,
        output_schema: type[BaseModel] | None = None,
        output_key: str,
        tools: list[ToolDescriptor] | None = None,
        after_tool_callback: Any | None = None,
        initial_state: dict[str, Any] | None = None,
        backend_plugins: list[Any] | None = None,
    ) -> SingleAgentResult:
        """Execute a single LLM agent call (meta-agent, debugger, etc.)."""
        llm = _resolve_llm(model)

        agent_kwargs: dict[str, Any] = {
            "name": agent_name,
            "model": llm,
            "instruction": instruction,
            "output_key": output_key,
            "generate_content_config": types.GenerateContentConfig(
                temperature=temperature,
            ),
        }
        if output_schema is not None:
            agent_kwargs["output_schema"] = output_schema
        if tools:
            from google.adk.tools import FunctionTool

            agent_kwargs["tools"] = _build_tools(tools)
        if after_tool_callback is not None:
            agent_kwargs["after_tool_callback"] = after_tool_callback

        agent = LlmAgent(**agent_kwargs)

        session_service = self._session_service or InMemorySessionService()
        session_id = uuid.uuid4().hex
        app_name = f"fedotmas_{agent_name}"

        session = await session_service.create_session(
            app_name=app_name,
            user_id="system",
            session_id=session_id,
            state=dict(initial_state or {}),
        )

        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )

        total_prompt = 0
        total_completion = 0
        start = time.monotonic()

        if backend_plugins:
            runner_kwargs: dict[str, Any] = {
                "app": App(
                    name=app_name,
                    root_agent=agent,
                    plugins=list(backend_plugins),
                ),
                "session_service": session_service,
            }
        else:
            runner_kwargs = {
                "app_name": app_name,
                "agent": agent,
                "session_service": session_service,
            }

        async with Runner(**runner_kwargs) as runner:
            async for event in runner.run_async(
                user_id="system",
                session_id=session.id,
                new_message=message,
            ):
                if event.partial:
                    continue

                if event.usage_metadata:
                    um = event.usage_metadata
                    prompt = um.prompt_token_count or 0
                    completion = um.candidates_token_count or 0
                    total_prompt += prompt
                    total_completion += completion
                    if prompt or completion:
                        _log.info("Tokens | prompt={} completion={}", prompt, completion)

                if event.content and event.content.parts:
                    texts = [p.text for p in event.content.parts if p.text]
                    if texts:
                        _log.debug("Response preview | text={}", texts[0][:200])

                if event.error_code:
                    _log.error(
                        "LLM error | agent={} code={} msg={}",
                        agent_name,
                        event.error_code,
                        event.error_message,
                    )
                    raise RuntimeError(
                        f"{agent_name} LLM error {event.error_code}: {event.error_message}"
                    )

        elapsed = time.monotonic() - start
        _log.info(
            "{} complete | elapsed={:.1f}s prompt={} completion={}",
            agent_name,
            elapsed,
            total_prompt,
            total_completion,
        )

        final_session = await session_service.get_session(
            app_name=app_name,
            user_id="system",
            session_id=session.id,
        )
        if final_session is None:
            raise RuntimeError(
                f"{agent_name}: session lost after execution — results unavailable"
            )

        raw_output = final_session.state.get(output_key)
        _log.debug(
            "Raw output | key={} type={} preview={}",
            output_key,
            type(raw_output).__name__,
            str(raw_output)[:500],
        )

        return SingleAgentResult(
            raw_output=raw_output,
            state=dict(final_session.state),
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            elapsed=elapsed,
        )
