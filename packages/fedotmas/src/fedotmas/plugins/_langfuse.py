from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from langfuse import Langfuse

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.langfuse")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


def _is_workflow_node(name: str) -> bool:
    return name.startswith(_WORKFLOW_PREFIXES)


def _extract_text(content: types.Content | None) -> str | None:
    """Extract text from ADK Content object."""
    if not content or not content.parts:
        return None
    texts = [p.text for p in content.parts if p.text]
    return "\n".join(texts) if texts else None


def _contents_to_messages(contents: list[types.Content]) -> list[dict[str, str]]:
    """Convert ADK Contents to a simple list of role/content dicts for Langfuse input."""
    messages = []
    for c in contents:
        text = _extract_text(c)
        if text:
            messages.append({"role": c.role or "user", "content": text})
    return messages


class LangfusePlugin(BasePlugin):
    """FEDOT.MAS plugin that sends traces to Langfuse.

    Maps ADK agent lifecycle callbacks to Langfuse observations:

    - Pipeline/meta-agent runs → top-level trace
    - Individual agents → spans (``as_type="agent"``)
    - LLM calls → generations (``as_type="generation"``)
    - Tool calls → tool spans (``as_type="tool"``)

    The plugin persists its trace across multiple ADK ``Runner`` invocations
    so that meta-agent config generation and pipeline execution appear
    under a single Langfuse trace.

    Configuration is via standard Langfuse env vars (``LANGFUSE_PUBLIC_KEY``,
    ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST``) or constructor parameters.

    Usage::

        from fedotmas.plugins import LangfusePlugin, LoggingPlugin

        maw = MAW(plugins=[LoggingPlugin(), LangfusePlugin()])
        result = await maw.run("Research quantum computing")
    """

    def __init__(
        self,
        *,
        trace_name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        flush_timeout_s: float | None = None,
        shutdown_timeout_s: float | None = None,
    ) -> None:
        super().__init__(name="fedotmas_langfuse")
        self._trace_name = trace_name
        self._user_id = user_id
        self._session_id = session_id
        self._tags = tags or []
        self._metadata = metadata or {}
        self._flush_timeout_s = (
            flush_timeout_s
            if flush_timeout_s is not None
            else _float_env("LANGFUSE_FLUSH_TIMEOUT_S", 5.0)
        )
        self._shutdown_timeout_s = (
            shutdown_timeout_s
            if shutdown_timeout_s is not None
            else _float_env("LANGFUSE_SHUTDOWN_TIMEOUT_S", 2.0)
        )

        # Langfuse client kwargs — None values are resolved from env by the SDK.
        self._lf_kwargs: dict[str, Any] = {}
        if public_key is not None:
            self._lf_kwargs["public_key"] = public_key
        if secret_key is not None:
            self._lf_kwargs["secret_key"] = secret_key
        if host is not None:
            self._lf_kwargs["host"] = host

        # Lazily initialised in the first before_run_callback.
        self._langfuse: Langfuse | None = None

        # One trace per full MAS/MAW run, reused across meta + pipeline runners.
        self._trace: Any | None = None
        # Per-runner "run" span (child of the trace).
        self._run_span: Any | None = None
        # agent_name → Langfuse observation
        self._spans: dict[str, Any] = {}
        # agent_name → current generation observation
        self._generations: dict[str, Any] = {}
        # function_call_id → tool observation
        self._tool_spans: dict[str, Any] = {}
        # agent_name → monotonic start time
        self._agent_start: dict[str, float] = {}

    def _ensure_client(self) -> Langfuse:
        if self._langfuse is None:
            self._langfuse = Langfuse(**self._lf_kwargs)
            _log.debug("Langfuse client initialised")
        return self._langfuse

    def _ensure_trace(self, run_name: str) -> Any:
        """Create a trace if one doesn't exist yet, or reuse the existing one."""
        lf = self._ensure_client()
        if self._trace is None:
            name = self._trace_name or f"fedotmas:{run_name}"
            self._trace = lf.start_observation(
                name=name,
                as_type="span",
                metadata={**self._metadata, "framework": "fedotmas"},
            )
            _log.debug("Created Langfuse trace | name={}", name)
        return self._trace

    def _get_agent_name(self, callback_context: CallbackContext) -> str:
        return callback_context._invocation_context.agent.name

    def _get_parent_span(self, callback_context: CallbackContext) -> Any:
        """Find the parent span for the current agent using the branch hierarchy."""
        branch = callback_context._invocation_context.branch
        if branch:
            parts = branch.split(".")
            # Walk up the branch to find the nearest tracked parent.
            for i in range(len(parts) - 1, -1, -1):
                parent_name = parts[i]
                if parent_name in self._spans:
                    return self._spans[parent_name]
        return self._run_span or self._trace

    # ── Run lifecycle ──────────────────────────────────────────────

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        root_name = invocation_context.agent.name
        trace = self._ensure_trace(root_name)
        self._run_span = trace.start_observation(
            name=f"run:{root_name}",
            as_type="span",
            metadata={"app_name": invocation_context.app_name},
        )
        _log.debug("Langfuse run span started | name=run:{}", root_name)
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        if self._run_span is not None:
            self._run_span.end()
            self._run_span = None
        _log.debug("Langfuse run span ended")

    # ── Agent lifecycle ────────────────────────────────────────────

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if _is_workflow_node(agent.name):
            return None

        parent = self._get_parent_span(callback_context)
        if parent is None:
            parent = self._ensure_trace(agent.name)

        span = parent.start_observation(
            name=agent.name,
            as_type="agent",
            input={"instruction": getattr(agent, "instruction", None)},
        )
        self._spans[agent.name] = span
        self._agent_start[agent.name] = time.monotonic()
        _log.debug("Langfuse agent span started | name={}", agent.name)
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if _is_workflow_node(agent.name):
            return None

        span = self._spans.pop(agent.name, None)
        t0 = self._agent_start.pop(agent.name, None)
        if span is not None:
            elapsed = time.monotonic() - t0 if t0 is not None else None
            span.update(
                metadata={"elapsed_s": round(elapsed, 2)} if elapsed else None,
            )
            span.end()
            _log.debug("Langfuse agent span ended | name={}", agent.name)
        return None

    # ── Model (LLM) lifecycle ──────────────────────────────────────

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        agent_name = self._get_agent_name(callback_context)
        parent = self._spans.get(agent_name) or self._run_span or self._trace
        if parent is None:
            return None

        input_messages = _contents_to_messages(llm_request.contents)
        model_params: Dict[str, Any] = {}
        if llm_request.config and llm_request.config.temperature is not None:
            model_params["temperature"] = llm_request.config.temperature
        if llm_request.config and llm_request.config.max_output_tokens is not None:
            model_params["max_tokens"] = llm_request.config.max_output_tokens

        gen = parent.start_observation(
            name=f"{agent_name}:llm",
            as_type="generation",
            model=llm_request.model or "unknown",
            input=input_messages,
            model_parameters=model_params if model_params else None,
        )
        self._generations[agent_name] = gen
        _log.debug(
            "Langfuse generation started | agent={} model={}",
            agent_name,
            llm_request.model,
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        agent_name = self._get_agent_name(callback_context)
        gen = self._generations.pop(agent_name, None)
        if gen is None:
            return None

        output_text = _extract_text(llm_response.content)
        usage: Dict[str, int] = {}
        if llm_response.usage_metadata:
            um = llm_response.usage_metadata
            if um.prompt_token_count:
                usage["input"] = um.prompt_token_count
            if um.candidates_token_count:
                usage["output"] = um.candidates_token_count
            if um.total_token_count:
                usage["total"] = um.total_token_count

        gen.update(
            output=output_text,
            usage_details=usage if usage else None,
            version=llm_response.model_version,
        )
        gen.end()
        _log.debug(
            "Langfuse generation ended | agent={} usage={}",
            agent_name,
            usage,
        )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        agent_name = self._get_agent_name(callback_context)
        gen = self._generations.pop(agent_name, None)
        if gen is not None:
            gen.update(
                level="ERROR",
                status_message=str(error),
            )
            gen.end()
            _log.debug(
                "Langfuse generation error | agent={} error={}", agent_name, error
            )
        return None

    # ── Tool lifecycle ─────────────────────────────────────────────

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        agent_name = self._get_agent_name(tool_context)
        parent = self._spans.get(agent_name) or self._run_span or self._trace
        if parent is None:
            return None

        call_id = tool_context.function_call_id or tool.name
        tool_span = parent.start_observation(
            name=f"tool:{tool.name}",
            as_type="tool",
            input=tool_args,
        )
        self._tool_spans[call_id] = tool_span
        _log.debug(
            "Langfuse tool span started | agent={} tool={}", agent_name, tool.name
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        call_id = tool_context.function_call_id or tool.name
        tool_span = self._tool_spans.pop(call_id, None)
        if tool_span is not None:
            # Truncate large results to avoid bloating Langfuse.
            output = str(result)[:2000] if result else None
            tool_span.update(output=output)
            tool_span.end()
            _log.debug("Langfuse tool span ended | tool={}", tool.name)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        call_id = tool_context.function_call_id or tool.name
        tool_span = self._tool_spans.pop(call_id, None)
        if tool_span is not None:
            tool_span.update(level="ERROR", status_message=str(error))
            tool_span.end()
            _log.debug("Langfuse tool span error | tool={} error={}", tool.name, error)
        return None

    # ── Event callback (state tracking) ────────────────────────────

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        # We don't need to do much here since before/after model/tool callbacks
        # handle the heavy lifting. But capture error events that aren't
        # model/tool errors (e.g. agent-level errors from ADK).
        if event.error_code and self._run_span is not None:
            self._run_span.update(
                level="ERROR",
                status_message=f"{event.error_code}: {event.error_message}",
            )
        return None

    # ── Cleanup ────────────────────────────────────────────────────

    def _call_with_timeout(
        self,
        name: str,
        func: Any,
        timeout_s: float,
    ) -> None:
        if timeout_s <= 0:
            func()
            return

        errors: list[BaseException] = []
        done = threading.Event()

        def _target() -> None:
            try:
                func()
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(
            target=_target,
            name=f"fedotmas-langfuse-{name}",
            daemon=True,
        )
        thread.start()

        if not done.wait(timeout_s):
            _log.warning(
                "Langfuse {} timed out after {:.1f}s; continuing without blocking run",
                name,
                timeout_s,
            )
            return

        if errors:
            raise errors[0]

    def end_trace(self) -> None:
        """End the current trace and flush. Call after a full MAS/MAW run."""
        if self._trace is not None:
            self._trace.end()
            self._trace = None
        if self._langfuse is not None:
            try:
                self._call_with_timeout(
                    "flush",
                    self._langfuse.flush,
                    self._flush_timeout_s,
                )
                self._call_with_timeout(
                    "shutdown",
                    self._langfuse.shutdown,
                    self._shutdown_timeout_s,
                )
            except Exception as exc:
                _log.warning("Langfuse finalization failed: {}", exc)
            self._langfuse = None
        _log.debug("Langfuse trace ended and flushed")

    async def close(self) -> None:
        # Do NOT end the trace here — close() is called by each Runner's
        # __aexit__, but we want the trace to persist across multiple runners
        # (meta-agent + pipeline).  The trace is finalised by end_trace()
        # which BaseMAS calls after the full run completes.
        pass


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        _log.warning("Invalid {}={!r}; using {}", name, value, default)
        return default
