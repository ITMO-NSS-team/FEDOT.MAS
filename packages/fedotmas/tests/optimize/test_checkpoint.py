"""Tests for checkpoint save/load."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._state import OptimizationState, TaskResult


def _config(instruction: str = "Do a") -> MAWConfig:
    agents = [MAWAgentConfig(name="a", instruction=instruction, output_key="a")]
    return MAWConfig(agents=agents, pipeline=MAWStepConfig(agent_name="a"))


class TestCheckpointSaveLoad:
    def test_roundtrip(self):
        state = OptimizationState()
        c1 = state.add_candidate(_config("v1"), origin="seed")
        state.record_task_result(
            c1,
            TaskResult(task="t1", state={"a": "out1"}, score=0.7, feedback="good"),
        )
        c2 = state.add_candidate(
            _config("v2"), parent_index=0, origin="mutation"
        )
        state.record_task_result(
            c2,
            TaskResult(task="t1", state={"a": "out2"}, score=0.9, feedback="great"),
        )
        state.update_pareto_front()
        state.iteration = 5

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            state.save(path)
            loaded = OptimizationState.load(path)

            assert len(loaded.candidates) == 2
            assert loaded.candidates[0].origin == "seed"
            assert loaded.candidates[1].origin == "mutation"
            assert loaded.candidates[0].scores["t1"] == 0.7
            assert loaded.candidates[1].scores["t1"] == 0.9
            assert loaded.candidates[1].parent_index == 0

            # total_evaluations and iteration should survive roundtrip
            assert loaded.total_evaluations == 2
            assert loaded.iteration == 5

            # Cache should be restored
            cached = loaded.cache.get(loaded.candidates[0].config_hash, "t1")
            assert cached is not None
            assert cached.score == 0.7
        finally:
            path.unlink(missing_ok=True)

    def test_empty_state_roundtrip(self):
        state = OptimizationState()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            state.save(path)
            loaded = OptimizationState.load(path)
            assert len(loaded.candidates) == 0
            assert loaded.total_evaluations == 0
            assert loaded.iteration == 0
        finally:
            path.unlink(missing_ok=True)
