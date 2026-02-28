"""Unified ADK plugin for CodeSynapse integration.

``SynapsePlugin`` is the single entry-point that intercepts all ADK
lifecycle hooks and delegates to sub-components:

- :class:`~fedotmas_synapse.checkpoint.CheckpointCallback`
  (before/after agent → MongoDB snapshots)
- :class:`~fedotmas_synapse.otel.OtelEventCallback`
  (on_event → OTEL spans)
- :class:`~fedotmas_synapse.bridge.MASEventBridge`
  (all hooks → SSE events)
- :class:`~fedotmas_synapse.model_gates.BifrostModelGates`
  (before_model → temperature enforcement)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from google.adk.events import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase

from fedotmas.common.logging import get_logger
from fedotmas_synapse.bridge import MASEventBridge
from fedotmas_synapse.checkpoint import CheckpointCallback
from fedotmas_synapse.memory import SynapseMemoryServiceAdapter
from fedotmas_synapse.model_gates import BifrostModelGates
from fedotmas_synapse.otel import OtelEventCallback
from fedotmas_synapse.session import MongoSessionService

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.runners import InvocationContext
    from google.adk.tools import BaseTool, ToolContext

    from events.emitter import EventEmitter
    from telemetry.tracer import SynapseTracer

_log = get_logger("fedotmas_synapse.plugin")


class SynapsePlugin(BasePlugin):
    """Unified ADK plugin for CodeSynapse integration.

    Args:
        db: Motor async database instance (shared ``synaps`` DB from
            CodeSynapse's ``MongoStorageBackend.db``).
        project_id: Scoping identifier for memory service and events.
        event_emitter: CodeSynapse ``EventEmitter`` for SSE bridge.
            Optional — when ``None``, bridge hooks are no-ops.
        tracer: CodeSynapse ``SynapseTracer`` for OTEL integration.
            Preferred over *otel_endpoint*.
        otel_endpoint: **Deprecated.** Use *tracer* instead.  Kept for
            backward compatibility; triggers a deprecation warning.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        *,
        project_id: str = "",
        event_emitter: EventEmitter | None = None,
        tracer: SynapseTracer | None = None,
        otel_endpoint: str | None = None,
    ) -> None:
        super().__init__(name="synapse")
        self._db = db

        if otel_endpoint is not None:
            _log.warning(
                "otel_endpoint is deprecated — pass tracer= instead"
            )

        # --- public services (exposed via mas_kwargs) ---
        self.session_service = MongoSessionService(db)
        self.memory_service = SynapseMemoryServiceAdapter(
            db, project_id=project_id
        )

        # --- internal sub-components ---
        self._checkpoint = CheckpointCallback(db=db, project_id=project_id)
        self._bridge: MASEventBridge | None = (
            MASEventBridge(event_emitter, project_id)
            if event_emitter is not None
            else None
        )
        self._model_gates = BifrostModelGates()
        self._otel: OtelEventCallback | None = (
            OtelEventCallback(tracer=tracer)
            if (tracer or otel_endpoint)
            else None
        )

    def mas_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``MAS()`` constructor.

        Includes session_service, memory_service, and plugins=[self].
        Agent-level callbacks (before/after_agent_callbacks) removed --
        checkpoint logic now handled via plugin hooks.
        """
        return {
            "session_service": self.session_service,
            "memory_service": self.memory_service,
            "plugins": [self],
        }

    # --- BasePlugin lifecycle hooks ---

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        if self._bridge:
            try:
                await self._bridge.pipeline_started(invocation_context)
            except Exception:
                _log.warning("bridge.pipeline_started failed", exc_info=True)
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        if self._bridge:
            try:
                await self._bridge.pipeline_completed(invocation_context)
            except Exception:
                _log.warning(
                    "bridge.pipeline_completed failed", exc_info=True
                )

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        try:
            await self._checkpoint.before_agent(agent, callback_context)
        except Exception:
            _log.warning("checkpoint.before_agent failed", exc_info=True)
        if self._bridge:
            try:
                await self._bridge.agent_started(agent, callback_context)
            except Exception:
                _log.warning("bridge.agent_started failed", exc_info=True)
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        try:
            await self._checkpoint.after_agent(agent, callback_context)
        except Exception:
            _log.warning("checkpoint.after_agent failed", exc_info=True)
        if self._bridge:
            try:
                await self._bridge.agent_completed(agent, callback_context)
            except Exception:
                _log.warning("bridge.agent_completed failed", exc_info=True)
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        try:
            self._model_gates.enforce(llm_request)
        except Exception:
            _log.warning("model_gates.enforce failed", exc_info=True)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if self._bridge:
            try:
                await self._bridge.model_call(callback_context, llm_response)
            except Exception:
                _log.warning("bridge.model_call failed", exc_info=True)
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        if self._bridge:
            try:
                await self._bridge.model_error(
                    callback_context, llm_request, error
                )
            except Exception:
                _log.warning("bridge.model_error failed", exc_info=True)
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        if self._bridge:
            try:
                await self._bridge.tool_started(tool, tool_args, tool_context)
            except Exception:
                _log.warning("bridge.tool_started failed", exc_info=True)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        if self._bridge:
            try:
                await self._bridge.tool_completed(
                    tool, tool_args, tool_context, result
                )
            except Exception:
                _log.warning("bridge.tool_completed failed", exc_info=True)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        if self._bridge:
            try:
                await self._bridge.tool_error(
                    tool, tool_args, tool_context, error
                )
            except Exception:
                _log.warning("bridge.tool_error failed", exc_info=True)
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        if self._otel:
            try:
                self._otel(event)
            except Exception:
                _log.warning("otel callback failed", exc_info=True)
        return None

    # NOTE: on_user_message_callback is defined by BasePlugin but not
    # overridden here.  Evaluate whether bridge/checkpoint should observe
    # user messages when those components are implemented.  Base returns
    # None (no-op), which is correct for now.

    async def close(self) -> None:
        """Release resources held by sub-components."""
        if self._otel and hasattr(self._otel, "close"):
            try:
                self._otel.close()
            except Exception:
                _log.warning("otel close failed", exc_info=True)
        if hasattr(self.session_service, "close"):
            try:
                await self.session_service.close()
            except Exception:
                _log.warning("session_service close failed", exc_info=True)
