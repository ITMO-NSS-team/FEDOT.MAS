"""Shared fixtures for MAS (routing-based) tests."""

from __future__ import annotations

import pytest

from fedotmas.mas.models import MASConfig


@pytest.fixture()
def simple_routing_config() -> MASConfig:
    """Coordinator + 2 workers routing config."""
    return MASConfig(
        coordinator={
            "name": "coordinator",
            "description": "Routes tasks to workers",
            "instruction": "Route the user request to the appropriate worker.",
        },
        workers=[
            {
                "name": "billing",
                "description": "Handles billing inquiries",
                "instruction": "Handle billing questions.",
                "model": "openai/gpt-4o",
                "output_key": "billing_out",
            },
            {
                "name": "support",
                "description": "Handles support tickets",
                "instruction": "Handle support questions.",
                "model": "openai/gpt-4o-mini",
                "output_key": "support_out",
            },
        ],
    )
