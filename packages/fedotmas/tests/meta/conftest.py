"""ADK mock fixtures for meta-agent tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int = 10
    candidates_token_count: int = 20


@dataclass
class FakePart:
    text: str | None = None


@dataclass
class FakeContent:
    parts: list[FakePart] = field(default_factory=list)
    role: str = "model"


@dataclass
class FakeEvent:
    partial: bool = False
    usage_metadata: FakeUsageMetadata | None = None
    content: FakeContent | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class FakeSession:
    id: str = "test-session"
    state: dict[str, Any] = field(default_factory=dict)


@pytest.fixture()
def mock_session_service():
    svc = AsyncMock()
    session = FakeSession(id="test-session", state={})
    svc.create_session = AsyncMock(return_value=session)
    svc.get_session = AsyncMock(return_value=session)
    return svc
