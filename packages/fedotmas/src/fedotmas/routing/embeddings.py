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
    """Async caching wrapper around a text-to-vector function.

    Concurrency model:

    - **Cache hits** return without acquiring any lock or yielding.
    - **Misses** coordinate through a per-key :class:`asyncio.Event`:
      the first caller for a key starts the embed; subsequent callers
      for the same key await that event instead of issuing a duplicate
      request. Different keys never block each other — they all run in
      parallel.

    This replaces a single ``asyncio.Lock`` covering the whole method,
    which serialised every embed call (even cache hits, even unrelated
    texts) and turned parallel routing into a sequential bottleneck.
    """

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
        self._inflight: dict[str, asyncio.Event] = {}

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        key = self._key(text)

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        event = self._inflight.get(key)
        if event is not None:
            # Another coroutine is already fetching this key. Wait,
            # then read the cache. If the inflight raised, the event
            # will still be set (via finally) but the cache stays
            # empty — fall through and retry ourselves.
            await event.wait()
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # Be the first/retrying fetcher. Safe to set without locking:
        # between the get above and this assignment there is no await
        # (asyncio is cooperative), so two coroutines can't both reach
        # this branch on the same key concurrently.
        event = asyncio.Event()
        self._inflight[key] = event
        try:
            vec = await self._embed_fn(self._model, text)
            self._cache[key] = vec
            return vec
        finally:
            # Pop before set so a waiter that wakes and re-enters sees
            # a clean slate (either cache hit or no inflight).
            self._inflight.pop(key, None)
            event.set()

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._model}:{h}"
