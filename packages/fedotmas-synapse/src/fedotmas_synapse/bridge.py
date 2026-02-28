from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from events.emitter import EventEmitter
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InvocationContext
    from google.adk.tools import BaseTool, ToolContext

_PREVIEW_MAX_LEN = 200
_log = get_logger("fedotmas_synapse.bridge")


class MASEventBridge:
    """Translate ADK lifecycle hooks into CodeSynapse SSE events.

    Args:
        event_emitter: CodeSynapse ``EventEmitter`` instance (injected via DI).
        project_id: Scoping identifier for all emitted events.
    """

    def __init__(self, event_emitter: EventEmitter, project_id: str) -> None:
        self._emitter = event_emitter
        self._project_id = project_id
        self._run_start_time: float = 0.0

    # --- Runner lifecycle ---

    async def pipeline_started(self, invocation_context: InvocationContext) -> None:
        """Emit ``mas_pipeline_started``.  Called from ``before_run_callback``."""
        raise NotImplementedError("MASEventBridge.pipeline_started")

    async def pipeline_completed(self, invocation_context: InvocationContext) -> None:
        """Emit ``mas_pipeline_completed``.  Called from ``after_run_callback``.

        Payload includes ``elapsed_seconds`` computed from
        ``_run_start_time``.
        """
        raise NotImplementedError("MASEventBridge.pipeline_completed")

    # --- Agent lifecycle ---

    async def agent_started(
        self, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Emit ``mas_agent_started``.  Called from ``before_agent_callback``."""
        raise NotImplementedError("MASEventBridge.agent_started")

    async def agent_completed(
        self, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Emit ``mas_agent_completed``.  Called from ``after_agent_callback``."""
        raise NotImplementedError("MASEventBridge.agent_completed")

    # --- Model lifecycle ---

    async def model_call(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """Emit ``mas_model_call``.  Called from ``after_model_callback``.

        Payload includes ``usage_metadata`` extracted from *llm_response*.
        """
        raise NotImplementedError("MASEventBridge.model_call")

    async def model_error(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> None:
        """Emit ``mas_model_error``.  Called from ``on_model_error_callback``."""
        raise NotImplementedError("MASEventBridge.model_error")

    # --- Tool lifecycle ---

    async def tool_started(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        """Emit ``mas_tool_call``.  Called from ``before_tool_callback``.

        ``args_preview`` is truncated to :data:`_PREVIEW_MAX_LEN` chars.
        """
        raise NotImplementedError("MASEventBridge.tool_started")

    async def tool_completed(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        """Emit ``mas_tool_result``.  Called from ``after_tool_callback``.

        ``result_preview`` is truncated to :data:`_PREVIEW_MAX_LEN` chars.
        """
        raise NotImplementedError("MASEventBridge.tool_completed")

    async def tool_error(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> None:
        """Emit ``mas_tool_error``.  Called from ``on_tool_error_callback``."""
        raise NotImplementedError("MASEventBridge.tool_error")
