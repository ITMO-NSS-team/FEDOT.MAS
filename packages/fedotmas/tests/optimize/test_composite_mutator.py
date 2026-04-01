"""Tests for CompositeMutator and WeightedMutator."""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._mutators._composite import CompositeMutator, WeightedMutator
from fedotmas.optimize._state import Candidate, Task


def _agent(name: str, instruction: str = "Do stuff") -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=instruction, output_key=name)


def _config(*names: str) -> MAWConfig:
    agents = [_agent(n) for n in names]
    pipeline = MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name=n) for n in names],
    )
    return MAWConfig(agents=agents, pipeline=pipeline)


def _mock_mutator(name: str = "mock") -> MagicMock:
    m = MagicMock()
    cfg = _config("a")
    m.mutate = AsyncMock(return_value=cfg)
    m.merge = AsyncMock(return_value=cfg)
    m.genealogy_merge = AsyncMock(return_value=cfg)
    m.token_usage = (0, 0)
    return m


def test_weighted_mutator_frozen():
    m = _mock_mutator()
    wm = WeightedMutator(mutator=m, weight=0.5)
    assert wm.weight == 0.5
    assert wm.mutator is m


def test_weighted_mutator_default_weight():
    m = _mock_mutator()
    wm = WeightedMutator(mutator=m)
    assert wm.weight == 1.0


def test_composite_requires_at_least_one():
    with pytest.raises(ValueError, match="At least one"):
        CompositeMutator([])


@pytest.mark.asyncio
async def test_mutate_delegates_to_one():
    """mutate() should call exactly one mutator."""
    m1 = _mock_mutator()
    m2 = _mock_mutator()
    candidate = Candidate(index=0, config=_config("a"), config_hash="h")

    composite = CompositeMutator(
        [WeightedMutator(m1, 1.0), WeightedMutator(m2, 0.0)],
        rng=random.Random(42),
    )
    await composite.mutate(candidate, ["a"], [Task("t1")])

    # With weight 0.0 for m2, only m1 should be called
    m1.mutate.assert_called_once()
    m2.mutate.assert_not_called()


@pytest.mark.asyncio
async def test_mutate_weighted_selection():
    """With extreme weights, selection should be deterministic."""
    m1 = _mock_mutator()
    m2 = _mock_mutator()
    candidate = Candidate(index=0, config=_config("a"), config_hash="h")

    composite = CompositeMutator(
        [WeightedMutator(m1, 0.0), WeightedMutator(m2, 1.0)],
        rng=random.Random(42),
    )
    await composite.mutate(candidate, ["a"], [Task("t1")])

    m1.mutate.assert_not_called()
    m2.mutate.assert_called_once()


@pytest.mark.asyncio
async def test_merge_calls_all_mutators():
    """merge() should call ALL mutators sequentially."""
    m1 = _mock_mutator()
    m2 = _mock_mutator()
    ca = Candidate(index=0, config=_config("a"), config_hash="h0")
    cb = Candidate(index=1, config=_config("a"), config_hash="h1")

    composite = CompositeMutator(
        [WeightedMutator(m1), WeightedMutator(m2)]
    )
    await composite.merge(ca, cb, [Task("t1")])

    m1.merge.assert_called_once()
    m2.merge.assert_called_once()


@pytest.mark.asyncio
async def test_genealogy_merge_calls_all_mutators():
    """genealogy_merge() should call ALL mutators sequentially."""
    m1 = _mock_mutator()
    m2 = _mock_mutator()
    anc = Candidate(index=0, config=_config("a"), config_hash="h0")
    ca = Candidate(index=1, config=_config("a"), config_hash="h1")
    cb = Candidate(index=2, config=_config("a"), config_hash="h2")

    composite = CompositeMutator(
        [WeightedMutator(m1), WeightedMutator(m2)]
    )
    await composite.genealogy_merge(anc, ca, cb, [Task("t1")])

    m1.genealogy_merge.assert_called_once()
    m2.genealogy_merge.assert_called_once()


def test_token_usage_aggregated():
    m1 = _mock_mutator()
    m1.token_usage = (100, 50)
    m2 = _mock_mutator()
    m2.token_usage = (200, 75)

    composite = CompositeMutator(
        [WeightedMutator(m1), WeightedMutator(m2)]
    )
    assert composite.token_usage == (300, 125)


@pytest.mark.asyncio
async def test_single_mutator_composite():
    """CompositeMutator with a single mutator should just delegate."""
    m = _mock_mutator()
    candidate = Candidate(index=0, config=_config("a"), config_hash="h")

    composite = CompositeMutator([WeightedMutator(m)])
    await composite.mutate(candidate, ["a"], [Task("t1")])

    m.mutate.assert_called_once()
    assert composite.token_usage == m.token_usage
