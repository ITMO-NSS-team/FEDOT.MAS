"""Tests for fedotmas.routing.embeddings.Embedder."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fedotmas.routing.embeddings import Embedder


class _FakeEmbedFn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, model: str, text: str) -> np.ndarray:
        self.calls.append((model, text))
        h = abs(hash(text)) % 1000
        return np.array([h / 1000.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.mark.asyncio
async def test_embed_calls_underlying_function_once_per_text() -> None:
    fn = _FakeEmbedFn()
    e = Embedder(model="m", embed_fn=fn)

    v1 = await e.embed("hello")
    v2 = await e.embed("hello")

    assert len(fn.calls) == 1
    np.testing.assert_array_equal(v1, v2)


@pytest.mark.asyncio
async def test_different_texts_produce_different_calls() -> None:
    fn = _FakeEmbedFn()
    e = Embedder(model="m", embed_fn=fn)

    await e.embed("hello")
    await e.embed("world")

    assert len(fn.calls) == 2


@pytest.mark.asyncio
async def test_concurrent_calls_for_same_text_dedupe() -> None:
    fn = _SlowEmbedFn()
    e = Embedder(model="m", embed_fn=fn)

    # 10 concurrent embedders of the same text result in exactly 1
    # underlying call — the per-key inflight Event coordinates waiters.
    results = await asyncio.gather(*[e.embed("same") for _ in range(10)])

    assert len(fn.calls) == 1
    for r in results:
        np.testing.assert_array_equal(r, results[0])


class _SlowEmbedFn:
    """Embed function that yields at least one event-loop tick. Lets us
    construct realistic interleavings under ``asyncio.gather``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.concurrent_peak = 0
        self._inflight = 0

    async def __call__(self, model: str, text: str) -> np.ndarray:
        self._inflight += 1
        self.concurrent_peak = max(self.concurrent_peak, self._inflight)
        self.calls.append((model, text))
        try:
            await asyncio.sleep(0)
            h = abs(hash(text)) % 1000
            return np.array([h / 1000.0, 0.0, 0.0, 0.0], dtype=np.float32)
        finally:
            self._inflight -= 1


@pytest.mark.asyncio
async def test_concurrent_calls_for_different_texts_run_in_parallel() -> None:
    fn = _SlowEmbedFn()
    e = Embedder(model="m", embed_fn=fn)

    await asyncio.gather(*[e.embed(f"q{i}") for i in range(10)])

    # All 10 distinct keys progressed past the first await concurrently;
    # a single shared lock would have driven this to 1.
    assert fn.concurrent_peak == 10
    assert len(fn.calls) == 10


@pytest.mark.asyncio
async def test_inflight_failure_does_not_poison_subsequent_calls() -> None:
    """A failed embed call must release its inflight slot so retries
    can run, instead of waiters spinning on a stuck Event."""

    attempts = {"n": 0}

    async def flaky(model: str, text: str) -> np.ndarray:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    e = Embedder(model="m", embed_fn=flaky)

    with pytest.raises(RuntimeError):
        await e.embed("same")
    vec = await e.embed("same")
    assert vec[0] == 1.0
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_model_propagates_to_embed_fn() -> None:
    fn = _FakeEmbedFn()
    e = Embedder(model="custom-model", embed_fn=fn)

    await e.embed("hello")

    assert fn.calls[0][0] == "custom-model"
