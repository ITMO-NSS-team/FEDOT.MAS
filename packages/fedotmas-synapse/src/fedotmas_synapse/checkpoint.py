"""Checkpoint callback — snapshots session state before/after agent execution.

Persists snapshots to the ``fedotmas_checkpoints`` collection in MongoDB.
Used by ``SynapsePlugin`` for agent lifecycle hooks.

See INCOMPATIBILITIES.md §1 for the ``invocation_id`` private-API tension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorDatabase

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext

_COLLECTION = "fedotmas_checkpoints"


class CheckpointCallback:
    """Before/after agent callbacks that snapshot session state to MongoDB.

    Document schema::

        {
            "session_id": str,
            "invocation_id": str,
            "agent_name": str,
            "phase": "before" | "after",
            "state_snapshot": dict,
            "project_id": str,
            "timestamp": datetime,
        }

    Args:
        db: Motor async database instance.
        project_id: Scoping identifier written into each checkpoint doc.
    """

    def __init__(
        self, *, db: AsyncIOMotorDatabase, project_id: str = ""
    ) -> None:
        self._db = db
        self._project_id = project_id
        self._collection = db[_COLLECTION]

    async def before_agent(
        self,
        agent: BaseAgent,
        callback_context: CallbackContext,
        invocation_id: str = "",
    ) -> None:
        """Snapshot session state **before** agent execution.

        Args:
            agent: The ADK agent about to execute.
            callback_context: ADK callback context (carries session state).
            invocation_id: Pipeline invocation ID for ADK Rewind scoping.
                Falls back to extracting from
                ``callback_context._invocation_context`` (private API).
        """
        raise NotImplementedError("CheckpointCallback.before_agent")

    async def after_agent(
        self,
        agent: BaseAgent,
        callback_context: CallbackContext,
        invocation_id: str = "",
    ) -> None:
        """Snapshot session state **after** agent execution.

        Args:
            agent: The ADK agent that just executed.
            callback_context: ADK callback context (carries session state).
            invocation_id: Pipeline invocation ID for ADK Rewind scoping.
                Falls back to extracting from
                ``callback_context._invocation_context`` (private API).
        """
        raise NotImplementedError("CheckpointCallback.after_agent")
