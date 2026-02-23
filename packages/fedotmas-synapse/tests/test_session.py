from __future__ import annotations

from google.adk.sessions import BaseSessionService, Session

from fedotmas_synapse.session import MongoSessionService


APP = "test-app"
USER = "user-1"


async def test_create_session(session_service: MongoSessionService) -> None:
    session = await session_service.create_session(app_name=APP, user_id=USER)
    assert isinstance(session, Session)
    assert session.app_name == APP
    assert session.user_id == USER
    assert session.id is not None


async def test_get_session(session_service: MongoSessionService) -> None:
    created = await session_service.create_session(app_name=APP, user_id=USER)
    fetched = await session_service.get_session(
        app_name=APP, user_id=USER, session_id=created.id,
    )
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.app_name == APP
    assert fetched.user_id == USER


async def test_get_session_not_found(session_service: MongoSessionService) -> None:
    result = await session_service.get_session(
        app_name=APP, user_id=USER, session_id="nonexistent-id",
    )
    assert result is None


async def test_list_sessions(session_service: MongoSessionService) -> None:
    await session_service.create_session(app_name=APP, user_id=USER)
    await session_service.create_session(app_name=APP, user_id=USER)
    response = await session_service.list_sessions(app_name=APP, user_id=USER)
    assert len(response.sessions) >= 2


async def test_delete_session(session_service: MongoSessionService) -> None:
    created = await session_service.create_session(app_name=APP, user_id=USER)
    await session_service.delete_session(
        app_name=APP, user_id=USER, session_id=created.id,
    )
    fetched = await session_service.get_session(
        app_name=APP, user_id=USER, session_id=created.id,
    )
    assert fetched is None


async def test_create_with_initial_state(session_service: MongoSessionService) -> None:
    session = await session_service.create_session(
        app_name=APP, user_id=USER, state={"key": "val"},
    )
    assert session.state.get("key") == "val"


def test_is_base_session_service(session_service: MongoSessionService) -> None:
    assert isinstance(session_service, BaseSessionService)
