from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fedotmas.common.logging import get_logger
from google.adk.memory import BaseMemoryService
from google.adk.memory.base_memory_service import MemoryEntry, SearchMemoryResponse
from google.adk.sessions import Session
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import OperationFailure

_COLLECTION = "fedotmas_memory"
_log = get_logger("fedotmas_synapse.memory")


class SynapseMemoryServiceAdapter(BaseMemoryService):
    """ADK memory service backed by MongoDB ``$text`` search.

    Stores ingested session data in the ``fedotmas_memory`` collection and
    supports full-text search via a MongoDB text index on the ``content``
    field.

    Args:
        db: Motor async database instance (shared with CodeSynapse).
        project_id: Scoping identifier — used as ``user_id`` context.
    """

    def __init__(self, db: AsyncIOMotorDatabase, project_id: str) -> None:
        self._db = db
        self._collection = db[_COLLECTION]
        self._project_id = project_id
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        if self._indexes_created:
            return
        await self._collection.create_index(
            [("content", "text")],
            name="content_text",
        )
        self._indexes_created = True

    async def add_session_to_memory(self, session: Session) -> None:
        """Extract session events/state and write to ``fedotmas_memory``."""
        await self._ensure_indexes()

        content_parts: list[str] = []

        # Include state values as searchable text
        for key, value in (session.state or {}).items():
            if key.startswith("temp:"):
                continue
            if isinstance(value, str):
                content_parts.append(f"{key}: {value}")
            else:
                content_parts.append(f"{key}: {value!r}")

        # Include text from events
        for event in session.events or []:
            if hasattr(event, "content") and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        content_parts.append(part.text)

        content = "\n".join(content_parts)
        if not content.strip():
            return

        now = datetime.now(timezone.utc)
        doc: dict[str, Any] = {
            "app_name": session.app_name,
            "user_id": session.user_id,
            "session_id": session.id,
            "source": "session_ingest",
            "content": content,
            "metadata": {
                "project_id": self._project_id,
                "state_keys": list((session.state or {}).keys()),
            },
            "created_at": now,
        }

        await self._collection.update_one(
            {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
            },
            {"$set": doc},
            upsert=True,
        )
        _log.debug(
            "ingested session {} ({} content parts)",
            session.id,
            len(content_parts),
        )

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """MongoDB ``$text`` search on ingested session data.

        Falls back to case-insensitive ``$regex`` if ``$text`` is not
        available (e.g. in mongomock test environments).
        """
        await self._ensure_indexes()

        base_filter: dict[str, Any] = {
            "app_name": app_name,
            "user_id": user_id,
        }

        try:
            text_filter = {**base_filter, "$text": {"$search": query}}
            cursor = (
                self._collection.find(
                    text_filter,
                    {"score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(10)
            )
            docs = [doc async for doc in cursor]
        except (OperationFailure, NotImplementedError):
            # Fallback for environments without $text support (e.g. mongomock)
            _log.debug("$text search unavailable, falling back to $regex")
            regex_filter = {
                **base_filter,
                "content": {"$regex": query, "$options": "i"},
            }
            cursor = self._collection.find(regex_filter).limit(10)
            docs = [doc async for doc in cursor]

        memories: list[MemoryEntry] = []
        for doc in docs:
            entry = MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=doc["content"])],
                ),
                id=doc.get("session_id"),
                author="memory_service",
                timestamp=doc.get("created_at", "").isoformat()
                if isinstance(doc.get("created_at"), datetime)
                else str(doc.get("created_at", "")),
                custom_metadata=doc.get("metadata", {}),
            )
            memories.append(entry)

        return SearchMemoryResponse(memories=memories)
