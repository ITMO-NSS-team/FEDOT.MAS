"""Checkpoint callback — snapshots session state before/after agent execution.

Persists snapshots to the ``fedotmas_checkpoints`` collection in MongoDB.
Used by ``SynapsePlugin`` for agent lifecycle hooks.

See INCOMPATIBILITIES.md §1 for the ``invocation_id`` resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorDatabase

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext

_COLLECTION = "fedotmas_checkpoints"
_log = get_logger("fedotmas_synapse.checkpoint")


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
    ) -> None:
        """Snapshot session state **before** agent execution.

        Args:
            agent: The ADK agent about to execute.
            callback_context: ADK callback context (carries session state).
                Use ``callback_context.invocation_id`` for pipeline scoping.
        """
        raise NotImplementedError("CheckpointCallback.before_agent")

    async def after_agent(
        self,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        """Snapshot session state **after** agent execution.

        Args:
            agent: The ADK agent that just executed.
            callback_context: ADK callback context (carries session state).
                Use ``callback_context.invocation_id`` for pipeline scoping.
        """
        raise NotImplementedError("CheckpointCallback.after_agent")
