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

import warnings
from typing import TYPE_CHECKING, Any

from google.adk.plugins import BasePlugin
from motor.motor_asyncio import AsyncIOMotorDatabase

from fedotmas_synapse.bridge import MASEventBridge
from fedotmas_synapse.checkpoint import CheckpointCallback
from fedotmas_synapse.memory import SynapseMemoryServiceAdapter
from fedotmas_synapse.model_gates import BifrostModelGates
from fedotmas_synapse.otel import OtelEventCallback
from fedotmas_synapse.session import MongoSessionService

if TYPE_CHECKING:
    from events.emitter import EventEmitter
    from telemetry.tracer import SynapseTracer


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
            warnings.warn(
                "otel_endpoint is deprecated, pass tracer instead",
                DeprecationWarning,
                stacklevel=2,
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

    async def before_run_callback(self, *, invocation_context):
        if self._bridge:
            await self._bridge.pipeline_started(invocation_context)

    async def after_run_callback(self, *, invocation_context):
        if self._bridge:
            await self._bridge.pipeline_completed(invocation_context)

    async def before_agent_callback(self, *, agent, callback_context):
        await self._checkpoint.before_agent(agent, callback_context)
        if self._bridge:
            await self._bridge.agent_started(agent, callback_context)

    async def after_agent_callback(self, *, agent, callback_context):
        await self._checkpoint.after_agent(agent, callback_context)
        if self._bridge:
            await self._bridge.agent_completed(agent, callback_context)

    async def before_model_callback(self, *, callback_context, llm_request):
        self._model_gates.enforce(llm_request)
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        if self._bridge:
            await self._bridge.model_call(callback_context, llm_response)
        return None

    async def on_model_error_callback(
        self, *, callback_context, llm_request, error
    ):
        if self._bridge:
            await self._bridge.model_error(
                callback_context, llm_request, error
            )
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        if self._bridge:
            await self._bridge.tool_started(tool, tool_args, tool_context)
        return None

    async def after_tool_callback(
        self, *, tool, tool_args, tool_context, result
    ):
        if self._bridge:
            await self._bridge.tool_completed(
                tool, tool_args, tool_context, result
            )
        return None

    async def on_tool_error_callback(
        self, *, tool, tool_args, tool_context, error
    ):
        if self._bridge:
            await self._bridge.tool_error(tool, tool_args, tool_context, error)
        return None

    async def on_event_callback(self, *, invocation_context, event):
        if self._otel:
            try:
                self._otel(event)
            except NotImplementedError:
                pass
        return None

    async def close(self):
        pass
