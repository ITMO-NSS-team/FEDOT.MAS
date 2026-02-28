from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import (
    GetSessionConfig,
    ListSessionsResponse,
)
from motor.motor_asyncio import AsyncIOMotorDatabase

_COLLECTION = "fedotmas_sessions"


class MongoSessionService(BaseSessionService):
    """ADK session service backed by MongoDB (via Motor).

    Stores sessions in the ``fedotmas_sessions`` collection with a compound
    unique index on ``(app_name, user_id, session_id)``.

    State prefix conventions for FEDOT.MAS agents:
    - No prefix: per-pipeline-run state (output keys, intermediate results)
    - ``user:``: per-project persistent state (survives retries)
    - ``app:``: global FEDOT.MAS config (shared across all runs)
    - ``temp:``: intra-invocation scratch (discarded before persisting)
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[_COLLECTION]
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        if self._indexes_created:
            return
        await self._collection.create_index(
            [("app_name", 1), ("user_id", 1), ("session_id", 1)],
            unique=True,
        )
        self._indexes_created = True

    @staticmethod
    def _strip_temp_keys(state: dict[str, Any]) -> dict[str, Any]:
        """Remove ``temp:`` prefixed keys before persisting."""
        return {k: v for k, v in state.items() if not k.startswith("temp:")}

    def _doc_to_session(self, doc: dict[str, Any]) -> Session:
        """Convert a MongoDB document to an ADK ``Session``."""
        return Session(
            app_name=doc["app_name"],
            user_id=doc["user_id"],
            id=doc["session_id"],
            state=doc.get("state", {}),
            events=[],
        )

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        await self._ensure_indexes()
        session_id = session_id or uuid.uuid4().hex
        clean_state = self._strip_temp_keys(state or {})
        now = datetime.now(timezone.utc)

        doc = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "state": clean_state,
            "events": [],
            "created_at": now,
            "updated_at": now,
        }
        await self._collection.insert_one(doc)

        return Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=dict(clean_state),
            events=[],
        )

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        await self._ensure_indexes()
        doc = await self._collection.find_one({
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
        })
        if doc is None:
            return None
        return self._doc_to_session(doc)

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        await self._ensure_indexes()
        query: dict[str, Any] = {"app_name": app_name}
        if user_id is not None:
            query["user_id"] = user_id

        sessions: list[Session] = []
        async for doc in self._collection.find(query):
            sessions.append(self._doc_to_session(doc))
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        await self._ensure_indexes()
        await self._collection.delete_one({
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
        })

    async def append_event(
        self,
        *,
        session: Session,
        event: Event,
    ) -> Session:
        """Persist an event and apply its ``state_delta`` to the stored session.

        ``temp:`` prefixed keys in the delta are stripped before persisting.
        """
        await self._ensure_indexes()

        update: dict[str, Any] = {
            "$set": {"updated_at": datetime.now(timezone.utc)},
            "$push": {"events": {"event_id": getattr(event, "id", None)}},
        }

        if event.actions and event.actions.state_delta:
            clean_delta = self._strip_temp_keys(event.actions.state_delta)
            for key, value in clean_delta.items():
                update["$set"][f"state.{key}"] = value

        await self._collection.update_one(
            {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
            },
            update,
        )

        # Update in-memory session object
        if event.actions and event.actions.state_delta:
            clean_delta = self._strip_temp_keys(event.actions.state_delta)
            session.state.update(clean_delta)

        return session
