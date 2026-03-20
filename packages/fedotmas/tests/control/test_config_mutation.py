"""Tests for MAWConfig mutation methods."""

from __future__ import annotations

import pytest

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig


def _agent(name: str, output_key: str | None = None) -> MAWAgentConfig:
    return MAWAgentConfig(
        name=name,
        instruction=f"Do {name}",
        output_key=output_key or name,
    )


def _seq_pipeline(*names: str) -> MAWStepConfig:
    return MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name=n) for n in names],
    )


def _three_agent_config() -> MAWConfig:
    return MAWConfig(
        agents=[_agent("a"), _agent("b"), _agent("c")],
        pipeline=_seq_pipeline("a", "b", "c"),
    )


class TestReplaceAgent:
    def test_same_name_replaces_config(self):
        cfg = _three_agent_config()
        new_a = MAWAgentConfig(
            name="b", instruction="New instruction", output_key="b"
        )
        result = cfg.replace_agent("b", new_a)

        agents_by_name = {a.name: a for a in result.agents}
        assert agents_by_name["b"].instruction == "New instruction"
        assert len(result.agents) == 3

    def test_different_name_updates_pipeline(self):
        cfg = _three_agent_config()
        new_agent = _agent("b_v2", output_key="b")
        result = cfg.replace_agent("b", new_agent)

        agents_by_name = {a.name: a for a in result.agents}
        assert "b" not in agents_by_name
        assert "b_v2" in agents_by_name

        refs = _collect_agent_names(result.pipeline)
        assert "b_v2" in refs
        assert "b" not in refs

    def test_unknown_agent_raises(self):
        cfg = _three_agent_config()
        with pytest.raises(ValueError, match="not found"):
            cfg.replace_agent("nonexistent", _agent("x"))

    def test_preserves_other_agents(self):
        cfg = _three_agent_config()
        result = cfg.replace_agent("b", _agent("b_v2", output_key="b"))
        names = {a.name for a in result.agents}
        assert names == {"a", "b_v2", "c"}


class TestReplaceStep:
    def test_replace_with_parallel(self):
        cfg = _three_agent_config()
        result = cfg.replace_step(
            "b",
            step=MAWStepConfig(
                type="parallel",
                children=[
                    MAWStepConfig(agent_name="b1"),
                    MAWStepConfig(agent_name="b2"),
                ],
            ),
            agents=[_agent("b1"), _agent("b2")],
        )

        agent_names = {a.name for a in result.agents}
        assert agent_names == {"a", "c", "b1", "b2"}
        assert "b" not in agent_names

        refs = _collect_agent_names(result.pipeline)
        assert refs == {"a", "c", "b1", "b2"}

    def test_replace_with_loop(self):
        cfg = _three_agent_config()
        result = cfg.replace_step(
            "b",
            step=MAWStepConfig(
                type="loop",
                children=[
                    MAWStepConfig(agent_name="drafter"),
                    MAWStepConfig(agent_name="reviewer"),
                ],
                max_iterations=3,
            ),
            agents=[_agent("drafter"), _agent("reviewer")],
        )

        agent_names = {a.name for a in result.agents}
        assert agent_names == {"a", "c", "drafter", "reviewer"}

    def test_unknown_agent_raises(self):
        cfg = _three_agent_config()
        with pytest.raises(ValueError, match="not found"):
            cfg.replace_step(
                "nonexistent",
                step=MAWStepConfig(agent_name="x"),
                agents=[_agent("x")],
            )


class TestInsertAfter:
    def test_insert_in_sequential(self):
        cfg = _three_agent_config()
        result = cfg.insert_after("b", _agent("d"))

        refs = _collect_agent_names_ordered(result.pipeline)
        assert refs == ["a", "b", "d", "c"]
        assert len(result.agents) == 4

    def test_insert_after_root_agent(self):
        cfg = MAWConfig(
            agents=[_agent("solo")],
            pipeline=MAWStepConfig(agent_name="solo"),
        )
        result = cfg.insert_after("solo", _agent("extra"))

        assert result.pipeline.type == "sequential"
        assert len(result.pipeline.children) == 2
        assert len(result.agents) == 2

    def test_unknown_agent_raises(self):
        cfg = _three_agent_config()
        with pytest.raises(ValueError, match="not found"):
            cfg.insert_after("nonexistent", _agent("x"))


class TestRemoveAgent:
    def test_remove_from_sequential(self):
        cfg = _three_agent_config()
        result = cfg.remove_agent("b")

        agent_names = {a.name for a in result.agents}
        assert agent_names == {"a", "c"}
        refs = _collect_agent_names(result.pipeline)
        assert refs == {"a", "c"}

    def test_remove_unwraps_singleton(self):
        cfg = MAWConfig(
            agents=[_agent("a"), _agent("b")],
            pipeline=_seq_pipeline("a", "b"),
        )
        result = cfg.remove_agent("b")
        assert result.pipeline.type == "agent"
        assert result.pipeline.agent_name == "a"

    def test_remove_only_agent_raises(self):
        cfg = MAWConfig(
            agents=[_agent("solo")],
            pipeline=MAWStepConfig(agent_name="solo"),
        )
        with pytest.raises(ValueError, match="Cannot remove"):
            cfg.remove_agent("solo")

    def test_unknown_agent_raises(self):
        cfg = _three_agent_config()
        with pytest.raises(ValueError, match="not found"):
            cfg.remove_agent("nonexistent")


def _collect_agent_names(node: MAWStepConfig) -> set[str]:
    if node.type == "agent":
        return {node.agent_name} if node.agent_name else set()
    result: set[str] = set()
    for child in node.children:
        result |= _collect_agent_names(child)
    return result


def _collect_agent_names_ordered(node: MAWStepConfig) -> list[str]:
    if node.type == "agent":
        return [node.agent_name] if node.agent_name else []
    result: list[str] = []
    for child in node.children:
        result.extend(_collect_agent_names_ordered(child))
    return result
