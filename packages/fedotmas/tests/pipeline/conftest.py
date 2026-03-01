"""Shared fixtures for pipeline builder & runner tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fedotmas.pipeline.models import PipelineConfig


# ---------------------------------------------------------------------------
# Fake event helpers (mimic google.adk.events.Event)
# ---------------------------------------------------------------------------


@dataclass
class FakeActions:
    state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeUsageMetadata:
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None


@dataclass
class FakeEvent:
    partial: bool = False
    author: str = "agent"
    content: Any = None
    error_code: str | None = None
    error_message: str | None = None
    usage_metadata: FakeUsageMetadata | None = None
    actions: FakeActions = field(default_factory=FakeActions)

    def get_function_calls(self) -> list:
        return []

    def get_function_responses(self) -> list:
        return []


@dataclass
class FakeSession:
    id: str = "sess-1"
    state: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_pipeline_config() -> PipelineConfig:
    """Two-agent sequential pipeline config."""
    return PipelineConfig.model_validate({
        "agents": [
            {
                "name": "alpha",
                "instruction": "Do alpha work on {topic}",
                "model": "openai/gpt-4o",
                "output_key": "alpha_out",
            },
            {
                "name": "beta",
                "instruction": "Do beta work",
                "model": "openai/gpt-4o-mini",
                "output_key": "beta_out",
            },
        ],
        "pipeline": {
            "type": "sequential",
            "children": [
                {"type": "agent", "agent_name": "alpha"},
                {"type": "agent", "agent_name": "beta"},
            ],
        },
    })


@pytest.fixture()
def mock_session_service():
    """AsyncMock session service with configurable session."""
    svc = AsyncMock()
    session = FakeSession()
    svc.create_session = AsyncMock(return_value=session)
    svc.get_session = AsyncMock(return_value=session)
    return svc
