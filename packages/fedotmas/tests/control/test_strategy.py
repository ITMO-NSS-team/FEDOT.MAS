"""Tests for Strategy enum and resolve_initial_state."""

from __future__ import annotations

from fedotmas.control._strategy import Strategy, resolve_initial_state
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.plugins._checkpoint import Checkpoint


def _agent(name: str) -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=f"Do {name}", output_key=name)


def _config(*names: str) -> MAWConfig:
    agents = [_agent(n) for n in names]
    if len(names) == 1:
        pipeline = MAWStepConfig(agent_name=names[0])
    else:
        pipeline = MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(agent_name=n) for n in names],
        )
    return MAWConfig(agents=agents, pipeline=pipeline)


def _checkpoints(*agent_names: str) -> list[Checkpoint]:
    state: dict = {}
    result = []
    for i, name in enumerate(agent_names):
        state = {**state, name: f"output_{name}"}
        result.append(Checkpoint(agent_name=name, state=dict(state), index=i))
    return result


class TestRetryFailed:
    def test_returns_last_checkpoint_state(self):
        cps = _checkpoints("a", "b")
        old = _config("a", "b", "c")
        new = _config("a", "b", "c")

        state, completed = resolve_initial_state(
            Strategy.RETRY_FAILED, cps, old, new
        )

        assert state == {"a": "output_a", "b": "output_b"}
        assert completed == {"a", "b"}

    def test_empty_checkpoints(self):
        old = _config("a", "b")
        new = _config("a", "b")
        state, completed = resolve_initial_state(
            Strategy.RETRY_FAILED, [], old, new
        )
        assert state is None
        assert completed == set()


class TestRestartAfter:
    def test_cuts_at_modified_agent(self):
        cps = _checkpoints("a", "b")
        old = _config("a", "b", "c")
        new_b = MAWAgentConfig(name="b", instruction="New B", output_key="b")
        new = MAWConfig(
            agents=[_agent("a"), new_b, _agent("c")],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(agent_name="a"),
                    MAWStepConfig(agent_name="b"),
                    MAWStepConfig(agent_name="c"),
                ],
            ),
        )

        state, completed = resolve_initial_state(
            Strategy.RESTART_AFTER, cps, old, new
        )

        assert state == {"a": "output_a"}
        assert completed == {"a"}

    def test_cuts_at_removed_agent(self):
        cps = _checkpoints("a", "b")
        old = _config("a", "b", "c")
        new = _config("a", "c")

        state, completed = resolve_initial_state(
            Strategy.RESTART_AFTER, cps, old, new
        )

        assert state == {"a": "output_a"}
        assert completed == {"a"}

    def test_all_unchanged(self):
        cps = _checkpoints("a", "b")
        cfg = _config("a", "b", "c")

        state, completed = resolve_initial_state(
            Strategy.RESTART_AFTER, cps, cfg, cfg
        )

        assert state == {"a": "output_a", "b": "output_b"}
        assert completed == {"a", "b"}

    def test_first_agent_modified(self):
        cps = _checkpoints("a", "b")
        old = _config("a", "b", "c")
        new_a = MAWAgentConfig(name="a", instruction="New A", output_key="a")
        new = MAWConfig(
            agents=[new_a, _agent("b"), _agent("c")],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(agent_name="a"),
                    MAWStepConfig(agent_name="b"),
                    MAWStepConfig(agent_name="c"),
                ],
            ),
        )

        state, completed = resolve_initial_state(
            Strategy.RESTART_AFTER, cps, old, new
        )

        assert state is None
        assert completed == set()


class TestRestartAll:
    def test_returns_none(self):
        cps = _checkpoints("a", "b")
        cfg = _config("a", "b", "c")

        state, completed = resolve_initial_state(
            Strategy.RESTART_ALL, cps, cfg, cfg
        )

        assert state is None
        assert completed == set()
