from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fedotmas.common.logging import get_logger
from motor.motor_asyncio import AsyncIOMotorDatabase

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

    def __init__(self, *, db: AsyncIOMotorDatabase, project_id: str = "") -> None:
        self._db = db
        self._project_id = project_id
        self._collection = db[_COLLECTION]

    async def _insert_checkpoint(
        self,
        agent: BaseAgent,
        callback_context: CallbackContext,
        phase: str,
    ) -> None:
        state = (
            callback_context.state.to_dict()
            if hasattr(callback_context.state, "to_dict")
            else dict(callback_context.state.items())
        )
        state_snapshot = {k: v for k, v in state.items() if not k.startswith("temp:")}
        doc = {
            "session_id": callback_context.session.id,
            "invocation_id": callback_context.invocation_id,
            "agent_name": agent.name,
            "phase": phase,
            "state_snapshot": state_snapshot,
            "project_id": self._project_id,
            "timestamp": datetime.now(timezone.utc),
        }
        await self._collection.insert_one(doc)
        _log.debug(
            "checkpoint {}/{} phase={} keys={}",
            callback_context.session.id,
            agent.name,
            phase,
            list(state_snapshot.keys()),
        )

    async def before_agent(
        self,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        """Snapshot session state **before** agent execution."""
        await self._insert_checkpoint(agent, callback_context, "before")

    async def after_agent(
        self,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        """Snapshot session state **after** agent execution."""
        await self._insert_checkpoint(agent, callback_context, "after")
