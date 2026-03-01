"""Shared fixtures for fedotmas tests."""

from __future__ import annotations

import pytest

from fedotmas.config.settings import ModelConfig


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
