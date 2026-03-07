"""Pydantic validation rules for MAS routing models — no mocks needed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedotmas.mas.models import MASConfig, MASAgentConfig


class TestDuplicateAgentNames:
    """Rule 1: Coordinator + worker sharing a name → ValueError."""

    def test_coordinator_worker_same_name(self):
        with pytest.raises(ValidationError, match="Duplicate agent name"):
            MASConfig(
                coordinator={
                    "name": "agent",
                    "description": "coord",
                    "instruction": "route",
                },
                workers=[
                    {
                        "name": "agent",
                        "description": "worker",
                        "instruction": "do work",
                    },
                ],
            )


class TestDuplicateWorkerNames:
    """Rule 2: Two workers with the same name → ValueError."""

    def test_duplicate_workers(self):
        with pytest.raises(ValidationError, match="Duplicate agent name"):
            MASConfig(
                coordinator={
                    "name": "coord",
                    "description": "coord",
                    "instruction": "route",
                },
                workers=[
                    {
                        "name": "worker",
                        "description": "first",
                        "instruction": "do A",
                    },
                    {
                        "name": "worker",
                        "description": "second",
                        "instruction": "do B",
                    },
                ],
            )


class TestEmptyWorkersList:
    """Rule 3: Empty workers list → ValueError."""

    def test_no_workers(self):
        with pytest.raises(ValidationError, match="At least one worker"):
            MASConfig(
                coordinator={
                    "name": "coord",
                    "description": "coord",
                    "instruction": "route",
                },
                workers=[],
            )


class TestModelNormalization:
    """Rule 4: Bare model name without provider prefix → ValueError."""

    def test_bare_model_rejected(self):
        with pytest.raises(ValidationError, match="must include a provider prefix"):
            MASAgentConfig(name="a", description="d", instruction="i", model="gpt-4o")

    def test_prefixed_model_stays(self):
        cfg = MASAgentConfig(
            name="a", description="d", instruction="i", model="gemini/flash"
        )
        assert cfg.model == "gemini/flash"

    def test_none_model_stays(self):
        cfg = MASAgentConfig(name="a", description="d", instruction="i", model=None)
        assert cfg.model is None


class TestCoordinatorCanBeWorkerName:
    """Rule 5: Coordinator name colliding with worker name → rejected."""

    def test_coord_name_equals_worker(self):
        with pytest.raises(ValidationError, match="Duplicate agent name"):
            MASConfig(
                coordinator={
                    "name": "coord",
                    "description": "coord",
                    "instruction": "route",
                },
                workers=[
                    {
                        "name": "coord",
                        "description": "worker",
                        "instruction": "work",
                    },
                ],
            )


class TestDescriptionRequired:
    """Rule 6: Omitting description → ValidationError."""

    def test_missing_description(self):
        with pytest.raises(ValidationError):
            MASAgentConfig(name="a", instruction="i")


class TestManyWorkers:
    """Rule 7: 10+ workers with unique names → valid."""

    def test_many_unique_workers(self):
        workers = [
            {
                "name": f"worker_{i}",
                "description": f"Worker {i}",
                "instruction": f"Do task {i}",
            }
            for i in range(12)
        ]
        config = MASConfig(
            coordinator={
                "name": "coord",
                "description": "coord",
                "instruction": "route",
            },
            workers=workers,
        )
        assert len(config.workers) == 12
