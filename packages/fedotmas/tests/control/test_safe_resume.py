from types import SimpleNamespace

import pytest
from google.adk.events import Event, EventActions

from fedotmas.control._strategy import resolve_initial_state, Strategy
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import Checkpoint, CheckpointPlugin


def config():
    return MAWConfig.model_validate(
        {
            "agents": [
                {"name": "a", "instruction": "Read A", "output_key": "a"},
                {"name": "b", "instruction": "Read B", "output_key": "b"},
                {
                    "name": "answer",
                    "instruction": "Use {a} and {b}",
                    "output_key": "answer",
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [
                    {
                        "type": "parallel",
                        "children": [{"agent_name": "a"}, {"agent_name": "b"}],
                    },
                    {"agent_name": "answer"},
                ],
            },
        }
    )


def test_changed_parallel_branch_does_not_discard_independent_verified_work():
    old = config()
    new = old.model_copy(deep=True)
    new.agents[0].instruction = "Repair A"
    cps = [
        Checkpoint("a", {"a": "old A"}, 0),
        Checkpoint("b", {"a": "old A", "b": "B", "untracked": "stale"}, 1),
        Checkpoint("answer", {"a": "old A", "b": "B", "answer": "old answer"}, 2),
    ]
    for strategy in (Strategy.RESTART_AFTER, Strategy.RETRY_FAILED):
        state, done = resolve_initial_state(strategy, cps, old, new)
        assert state == {"b": "B"}
        assert done == {"b"}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        {},
        [],
        "<abstain>missing</abstain>",
        '{"status":"unresolved","reason":"missing"}',
    ],
)
def test_invalid_output_and_dependents_cannot_be_reused(bad):
    cfg = config()
    cps = [
        Checkpoint("a", {"a": bad}, 0),
        Checkpoint("b", {"b": "B"}, 1),
        Checkpoint("answer", {"answer": "stale"}, 2),
    ]
    state, done = resolve_initial_state(Strategy.RESTART_AFTER, cps, cfg, cfg)
    assert state == {"b": "B"} and done == {"b"}


def test_unfinished_callback_checkpoint_is_not_reusable():
    cfg = config()
    state, done = resolve_initial_state(
        Strategy.RESTART_AFTER,
        [Checkpoint("a", {"a": "unchecked"}, 0, validated=False)],
        cfg,
        cfg,
    )
    assert state is None and not done


@pytest.mark.asyncio
async def test_checkpoint_finalizes_only_after_agent_validation_and_is_deep_copy():
    from google.adk.sessions.state import State

    plugin = CheckpointPlugin()
    state = State({"a": {"records": [1]}}, {})
    ctx = SimpleNamespace(state=state)
    await plugin.after_agent_callback(
        agent=SimpleNamespace(name="a"), callback_context=ctx
    )
    assert not plugin.checkpoints[0].validated
    state["a"] = {"status": "unresolved", "reason": "Rejected by after_agent"}
    event = Event(author="a", actions=EventActions(state_delta=state._delta))
    await plugin.on_event_callback(
        invocation_context=SimpleNamespace(session=SimpleNamespace(state={})),
        event=event,
    )
    assert plugin.checkpoints[0].validated
    assert plugin.checkpoints[0].state["a"]["status"] == "unresolved"
    state["a"]["status"] = "mutated"
    assert plugin.checkpoints[0].state["a"]["status"] == "unresolved"
