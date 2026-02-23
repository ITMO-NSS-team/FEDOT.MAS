from __future__ import annotations

from typing import Any, Optional

from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import GetSessionConfig, ListSessionsResponse


class MongoSessionService(BaseSessionService):
    """ADK session service backed by MongoDB (via Motor)."""

    def __init__(self, mongo_uri: str, db_name: str = "fedotmas") -> None:
        self.mongo_uri = mongo_uri
        self.db_name = db_name

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        raise NotImplementedError("MongoSessionService.create_session")

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        raise NotImplementedError("MongoSessionService.get_session")

    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        raise NotImplementedError("MongoSessionService.list_sessions")

    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str,
    ) -> None:
        raise NotImplementedError("MongoSessionService.delete_session")
