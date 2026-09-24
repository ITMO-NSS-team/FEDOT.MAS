from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

SERVER_PATH = (
    Path(__file__).parents[4]
    / "mcp-servers/websearch-searxng/src/mcp_websearch_searxng/server.py"
)
spec = importlib.util.spec_from_file_location("test_searxng_server", SERVER_PATH)
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
fake_fastmcp = ModuleType("fastmcp")
fake_fastmcp.Context = type("Context", (), {})
fake_fastmcp.FastMCP = type(
    "FastMCP",
    (),
    {
        "__init__": lambda self, *_args, **_kwargs: None,
        "tool": lambda self, fn: fn,
    },
)
with patch.dict(sys.modules, {"fastmcp": fake_fastmcp}):
    spec.loader.exec_module(server)


def _client(response=None, error=None):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=response, side_effect=error)
    return client


@pytest.mark.asyncio
async def test_zero_results_are_successful():
    response = MagicMock()
    response.json.return_value = {"results": []}
    with patch.object(server.httpx, "AsyncClient", return_value=_client(response)):
        result = await server.search("missing", AsyncMock())
    assert result.number_of_results == 0
    assert result.results == []


@pytest.mark.asyncio
async def test_all_engine_failures_are_not_zero_results():
    response = MagicMock()
    response.json.return_value = {
        "results": [],
        "unresponsive_engines": [
            [name, "timeout"] for name in server.ENGINES.split(",")
        ],
    }
    with (
        patch.object(server.httpx, "AsyncClient", return_value=_client(response)),
        pytest.raises(RuntimeError, match="all configured engines are unavailable"),
    ):
        await server.search("query", AsyncMock())


@pytest.mark.asyncio
async def test_backend_failure_raises_real_tool_error():
    request = httpx.Request("GET", "http://localhost/search")
    error = httpx.ConnectError("connection refused", request=request)
    with (
        patch.object(server.httpx, "AsyncClient", return_value=_client(error=error)),
        pytest.raises(RuntimeError, match="SearXNG connection error"),
    ):
        await server.search("query", AsyncMock())


@pytest.mark.asyncio
async def test_http_backend_failure_raises_real_tool_error():
    request = httpx.Request("GET", "http://localhost/search")
    response = httpx.Response(503, request=request, text="unavailable")
    with (
        patch.object(server.httpx, "AsyncClient", return_value=_client(response)),
        pytest.raises(RuntimeError, match="SearXNG HTTP error 503"),
    ):
        await server.search("query", AsyncMock())


@pytest.mark.asyncio
async def test_results_are_compact():
    response = MagicMock()
    response.json.return_value = {
        "results": [
            {
                "url": f"https://example.com/{n}",
                "title": "t" * 300,
                "content": "s" * 1000,
                "engine": "x",
                "score": 1.0,
            }
            for n in range(20)
        ]
    }
    with patch.object(server.httpx, "AsyncClient", return_value=_client(response)):
        result = await server.search("query", AsyncMock())
    assert len(result.results) == server.MAX_RETURNED_RESULTS
    assert len(result.results[0].title) == server.MAX_TITLE_CHARS
    assert len(result.results[0].content) == server.MAX_SNIPPET_CHARS
    assert "engine" not in result.results[0].model_dump()
