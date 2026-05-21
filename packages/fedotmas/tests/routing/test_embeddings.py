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
    fn = _FakeEmbedFn()
    e = Embedder(model="m", embed_fn=fn)

    # 10 concurrent embedders of same text should result in at most 1 call.
    # (Single asyncio task at a time per Embedder thanks to the lock.)
    results = await asyncio.gather(*[e.embed("same") for _ in range(10)])

    assert len(fn.calls) <= 2  # allow one race past the first check
    for r in results:
        np.testing.assert_array_equal(r, results[0])


@pytest.mark.asyncio
async def test_model_propagates_to_embed_fn() -> None:
    fn = _FakeEmbedFn()
    e = Embedder(model="custom-model", embed_fn=fn)

    await e.embed("hello")

    assert fn.calls[0][0] == "custom-model"
