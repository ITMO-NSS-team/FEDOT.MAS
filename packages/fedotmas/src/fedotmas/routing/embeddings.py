"""Query-embedding helper.

Default provider is ``litellm.aembedding`` against an OpenAI-compatible
endpoint — the same transport the rest of the pipeline already uses, so
no extra API key is needed. The embed function is injectable: tests pass
a fake to avoid network I/O.

Embeddings are cached in-process by ``(model, sha256(text))`` so we don't
pay twice for the same query during a single run or when back-filling
historical data.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable

import numpy as np

EmbedFn = Callable[[str, str], Awaitable[np.ndarray]]


async def _default_litellm_embed(model: str, text: str) -> np.ndarray:
    import litellm

    resp = await litellm.aembedding(model=model, input=text)
    vec = resp["data"][0]["embedding"]
    return np.asarray(vec, dtype=np.float32)


class Embedder:
    """Async caching wrapper around a text-to-vector function."""

    DEFAULT_MODEL = "openai/text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self._model = model
        self._embed_fn: EmbedFn = embed_fn or _default_litellm_embed
        self._cache: dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        key = self._key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            vec = await self._embed_fn(self._model, text)
            self._cache[key] = vec
        return vec

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._model}:{h}"
