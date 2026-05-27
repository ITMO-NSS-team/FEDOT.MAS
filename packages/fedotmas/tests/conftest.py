"""Shared fixtures for fedotmas tests."""

from __future__ import annotations

import pytest
from loguru import logger as _loguru_logger

from fedotmas._settings import ModelConfig


@pytest.fixture(autouse=True)
def _bridge_loguru_to_caplog(caplog: pytest.LogCaptureFixture):
    """Make pytest's ``caplog`` capture loguru records.

    The codebase logs through loguru, which writes to its own
    handlers and bypasses stdlib ``logging`` — so ``caplog`` sees
    nothing by default. Bridging here means tests can do the natural
    ``assert "..." in caplog.text`` instead of monkeypatching
    private ``_log`` objects per module.

    Removal is guarded because production code (e.g. logging setup
    inside a Runner) may call ``logger.remove()`` globally during the
    test, dropping our handler before we get to it.
    """
    handler_id = _loguru_logger.add(
        caplog.handler,
        level=0,
        format="{message}",
    )
    try:
        yield
    finally:
        try:
            _loguru_logger.remove(handler_id)
        except ValueError:
            # Handler already removed by code under test (e.g. a
            # logging reconfiguration). Nothing to clean up.
            pass


@pytest.fixture()
def openai_model() -> str:
    return "openai/gpt-4o"


@pytest.fixture()
def gemini_model() -> str:
    return "gemini/gemini-2.0-flash"


@pytest.fixture()
def allowed_models() -> list[str]:
    return ["openai/gpt-4o", "openai/gpt-4o-mini", "gemini/gemini-2.0-flash"]


@pytest.fixture()
def model_config() -> ModelConfig:
    return ModelConfig(model="openai/gpt-4o")


@pytest.fixture()
def two_agent_data() -> dict:
    return {
        "agents": [
            {
                "name": "researcher",
                "instruction": "Research the topic",
                "model": "openai/gpt-4o",
                "output_key": "research_result",
            },
            {
                "name": "writer",
                "instruction": "Write a report",
                "model": "openai/gpt-4o-mini",
                "output_key": "report",
            },
        ],
        "pipeline": {
            "type": "sequential",
            "children": [
                {"type": "agent", "agent_name": "researcher"},
                {"type": "agent", "agent_name": "writer"},
            ],
        },
    }


@pytest.fixture()
def single_agent_data() -> dict:
    return {
        "agents": [
            {
                "name": "solver",
                "instruction": "Solve the task",
                "model": "openai/gpt-4o",
                "output_key": "solution",
            },
        ],
        "pipeline": {
            "type": "agent",
        },
    }
