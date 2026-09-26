from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated

import httpx
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

load_dotenv()

_log = logging.getLogger("mcp_websearch_tavily")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_ROTATE_EVERY = 50
MAX_RESULTS = 10
MAX_QUERY_CHARS = 400
MAX_TITLE_CHARS = 160
MAX_URL_CHARS = 2_048
MAX_SNIPPET_CHARS = 500
KEY_FAILURE_STATUSES = {
    401: "auth",
    402: "quota",
    403: "auth",
    429: "rate_limit",
    432: "quota",
    433: "quota",
}


def _parse_rotation(value: str | None) -> int:
    if value is None or not value.strip():
        return DEFAULT_ROTATE_EVERY
    try:
        parsed = int(value.strip())
    except ValueError:
        parsed = 0
    if parsed < 1:
        _log.warning(
            "Invalid TAVILY_ROTATE_EVERY; using default interval of %d",
            DEFAULT_ROTATE_EVERY,
        )
        return DEFAULT_ROTATE_EVERY
    return parsed


def _parse_keys(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Read comma-separated keys first, then fall back to the single-key form."""
    keys = tuple(
        key.strip()
        for key in environ.get("TAVILY_API_KEYS", "").split(",")
        if key.strip()
    )
    if keys:
        return keys
    single_key = environ.get("TAVILY_API_KEY", "").strip()
    return (single_key,) if single_key else ()


@dataclass(frozen=True)
class KeyLease:
    index: int
    label: str
    api_key: str = field(repr=False)


class KeyPool:
    """Process-local, concurrency-safe rotation and Tavily request telemetry.

    The scheduled key is selected using the number of actual provider request
    attempts already started: (attempt_count // rotate_every) % key_count.
    Retry attempts consume slots too. A key-specific auth, quota, or rate-limit
    response disables that key for this process and retries on the next usable
    key. A short thread lock protects scheduling and counters across concurrent
    requests without serializing the network calls.
    """

    def __init__(self, keys: tuple[str, ...] | list[str], rotate_every: int = 50):
        cleaned = tuple(key.strip() for key in keys if key.strip())
        self._keys = cleaned
        self.rotate_every = rotate_every if rotate_every > 0 else DEFAULT_ROTATE_EVERY
        self._lock = threading.Lock()
        self._attempt_count = 0
        self._last_key_index: int | None = None
        self._unavailable: dict[int, str] = {}
        self._key_attempts = [0 for _ in cleaned]
        self._metrics = {
            "successful_requests": 0,
            "zero_result_requests": 0,
            "provider_failures": 0,
            "rate_limit_failures": 0,
            "quota_failures": 0,
            "auth_failures": 0,
            "key_rotations": 0,
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> KeyPool:
        values = os.environ if environ is None else environ
        return cls(
            _parse_keys(values),
            rotate_every=_parse_rotation(values.get("TAVILY_ROTATE_EVERY")),
        )

    @property
    def configured(self) -> bool:
        return bool(self._keys)

    def __repr__(self) -> str:
        return (
            f"KeyPool(configured_keys={len(self._keys)}, "
            f"usable_keys={self.usable_key_count}, rotate_every={self.rotate_every})"
        )

    @property
    def usable_key_count(self) -> int:
        with self._lock:
            return len(self._keys) - len(self._unavailable)

    def begin_request(
        self,
        *,
        after_index: int | None = None,
        excluded: set[int] | None = None,
    ) -> KeyLease | None:
        """Reserve one actual provider request attempt and return its key."""
        with self._lock:
            key_count = len(self._keys)
            if not key_count:
                return None
            excluded_indices = excluded or set()
            if after_index is None:
                start = (self._attempt_count // self.rotate_every) % key_count
            else:
                start = (after_index + 1) % key_count

            for offset in range(key_count):
                index = (start + offset) % key_count
                if index in self._unavailable or index in excluded_indices:
                    continue
                self._attempt_count += 1
                self._key_attempts[index] += 1
                if self._last_key_index is not None and self._last_key_index != index:
                    self._metrics["key_rotations"] += 1
                self._last_key_index = index
                return KeyLease(
                    index=index, label=f"key_{index}", api_key=self._keys[index]
                )
            return None

    def record_key_failure(self, index: int, category: str) -> None:
        with self._lock:
            self._metrics["provider_failures"] += 1
            category_metric = {
                "rate_limit": "rate_limit_failures",
                "quota": "quota_failures",
                "auth": "auth_failures",
            }.get(category)
            if category_metric is not None:
                self._metrics[category_metric] += 1
            self._unavailable.setdefault(index, category)

    def record_provider_failure(self) -> None:
        with self._lock:
            self._metrics["provider_failures"] += 1

    def record_success(self, *, zero_results: bool) -> None:
        with self._lock:
            self._metrics["successful_requests"] += 1
            if zero_results:
                self._metrics["zero_result_requests"] += 1

    def telemetry(self) -> dict[str, object]:
        with self._lock:
            return {
                "total_requests": self._attempt_count,
                **self._metrics,
                "configured_keys": len(self._keys),
                "currently_usable_keys": len(self._keys) - len(self._unavailable),
                "keys": [
                    {
                        "label": f"key_{index}",
                        "usable": index not in self._unavailable,
                        "unavailable_reason": self._unavailable.get(index),
                        "request_attempts": self._key_attempts[index],
                    }
                    for index in range(len(self._keys))
                ],
            }

    def redact(self, value: str) -> str:
        for secret in self._keys:
            value = value.replace(secret, "[REDACTED]")
        return value


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchError(BaseModel):
    code: str
    message: str
    attempted_keys: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    error: SearchError | None = None


DESCRIPTION = """
Search the web with Tavily first and SearXNG only when Tavily is unavailable.
Return the same compact result shape whichever backend serves the request.
"""

tavily_server = FastMCP("websearch-tavily", instructions=DESCRIPTION)
KEY_POOL = KeyPool.from_env()


def _error(query: str, code: str, message: str, attempted: list[str]) -> SearchResponse:
    return SearchResponse(
        query=KEY_POOL.redact(query),
        results=[],
        error=SearchError(
            code=code,
            message=KEY_POOL.redact(message),
            attempted_keys=attempted,
        ),
    )


async def _searxng_fallback(query: str, max_results: int) -> SearchResponse:
    """Use SearXNG only after Tavily cannot serve this request."""
    instance_url = os.getenv("SEARXNG_URL", "http://localhost:18888").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{instance_url}/search",
                params={"q": query, "format": "json", "engines": "bing,duckduckgo,brave,mullvadleta,yahoo,presearch", "categories": "general", "safesearch": 1},
            )
            response.raise_for_status()
            payload = response.json()
        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise TypeError("invalid results payload")
        results: list[SearchResult] = []
        for item in raw_results[:max_results]:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            url = KEY_POOL.redact(str(item["url"]))
            if len(url) > MAX_URL_CHARS:
                continue
            results.append(
                SearchResult(
                    title=KEY_POOL.redact(str(item.get("title") or ""))[
                        :MAX_TITLE_CHARS
                    ],
                    url=url,
                    snippet=KEY_POOL.redact(str(item.get("content") or ""))[
                        :MAX_SNIPPET_CHARS
                    ],
                )
            )
        _log.info("SearXNG fallback completed: %d results", len(results))
        return SearchResponse(query=KEY_POOL.redact(query)[:MAX_QUERY_CHARS], results=results)
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        _log.warning("SearXNG fallback failed: %s", type(exc).__name__)
        return _error(query, "SEARCH_UNAVAILABLE", "Tavily and SearXNG are unavailable.", [])


@tavily_server.tool
async def search(
    query: Annotated[str, Field(description="Search query text")],
    ctx: Context,
    max_results: Annotated[
        int,
        Field(description="Maximum ranked results to return", ge=1, le=MAX_RESULTS),
    ] = 5,
) -> SearchResponse:
    """Search Tavily and return a bounded list of ranked results.

    API requests are counted when they start. Auth, quota, and rate-limit
    failures disable the affected key for this process and retry once per
    remaining usable key. Zero results are a successful search response.
    """
    del ctx
    safe_query = KEY_POOL.redact(query)[:MAX_QUERY_CHARS]
    if not KEY_POOL.configured:
        return await _searxng_fallback(safe_query, max_results)

    attempted: list[str] = []
    excluded: set[int] = set()
    lease = KEY_POOL.begin_request()
    if lease is None:
        return await _searxng_fallback(safe_query, max_results)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while lease is not None:
            attempted.append(lease.label)
            excluded.add(lease.index)
            _log.info("Tavily request started with %s", lease.label)
            try:
                response = await client.post(
                    TAVILY_SEARCH_URL,
                    json={
                        "api_key": lease.api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
            except httpx.RequestError:
                KEY_POOL.record_provider_failure()
                _log.warning("Tavily provider request failed with %s", lease.label)
                return await _searxng_fallback(safe_query, max_results)

            status = response.status_code
            if status in KEY_FAILURE_STATUSES:
                category = KEY_FAILURE_STATUSES[status]
                KEY_POOL.record_key_failure(lease.index, category)
                _log.warning(
                    "Tavily marked %s unavailable after HTTP %d", lease.label, status
                )
                lease = KEY_POOL.begin_request(
                    after_index=lease.index,
                    excluded=excluded,
                )
                if lease is None:
                    return await _searxng_fallback(safe_query, max_results)
                continue
            if status >= 500:
                KEY_POOL.record_provider_failure()
                _log.warning("Tavily provider returned HTTP %d", status)
                return await _searxng_fallback(safe_query, max_results)
            if status >= 400:
                KEY_POOL.record_provider_failure()
                _log.warning("Tavily rejected the search request with HTTP %d", status)
                return _error(
                    safe_query,
                    "TAVILY_PROVIDER_ERROR",
                    f"Tavily rejected the search request with HTTP {status}.",
                    attempted,
                )

            try:
                payload = response.json()
                raw_results = (
                    payload.get("results") if isinstance(payload, dict) else None
                )
                if not isinstance(raw_results, list):
                    raise TypeError
                results: list[SearchResult] = []
                for raw in raw_results[:max_results]:
                    if not isinstance(raw, dict) or not raw.get("url"):
                        continue
                    url = KEY_POOL.redact(str(raw["url"]))
                    if len(url) > MAX_URL_CHARS:
                        continue
                    results.append(
                        SearchResult(
                            title=KEY_POOL.redact(str(raw.get("title") or ""))[
                                :MAX_TITLE_CHARS
                            ],
                            url=url,
                            snippet=KEY_POOL.redact(str(raw.get("content") or ""))[
                                :MAX_SNIPPET_CHARS
                            ],
                        )
                    )
            except (AttributeError, TypeError, ValueError):
                KEY_POOL.record_provider_failure()
                _log.warning("Tavily returned an invalid search response")
                return await _searxng_fallback(safe_query, max_results)

            KEY_POOL.record_success(zero_results=not results)
            _log.info(
                "Tavily search completed with %s: %d results", lease.label, len(results)
            )
            return SearchResponse(query=safe_query, results=results)

    return await _searxng_fallback(safe_query, max_results)


@tavily_server.tool
async def telemetry() -> dict[str, object]:
    """Return process-local Tavily request counts and safe key labels only."""
    return KEY_POOL.telemetry()


def main() -> None:
    tavily_server.run(show_banner=False)
