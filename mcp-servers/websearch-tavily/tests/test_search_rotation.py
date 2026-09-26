from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SERVER_PATH = Path(__file__).parents[1] / "src/mcp_websearch_tavily/server.py"
spec = importlib.util.spec_from_file_location("test_tavily_search_server", SERVER_PATH)
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
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


def _response(status: int, payload: dict | None = None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload or {"results": []}
    return response


def _client(post_responses: list):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(side_effect=post_responses)
    client.get = AsyncMock(return_value=_response(200, {"results": []}))
    return client


@pytest.mark.asyncio
async def test_search_rotates_432_429_then_returns_key_c_success(monkeypatch):
    pool = server.KeyPool(["key-A", "key-B", "key-C"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    tavily = _client(
        [
            _response(432),
            _response(429),
            _response(
                200,
                {"results": [{"title": "C", "url": "https://c.example", "content": "success"}]},
            ),
        ]
    )
    searxng = _client([])

    with patch.object(server.httpx, "AsyncClient", side_effect=[tavily, searxng]):
        result = await server.search("query", MagicMock())

    assert [call.kwargs["json"]["api_key"] for call in tavily.post.await_args_list] == [
        "key-A",
        "key-B",
        "key-C",
    ]
    assert result.results[0].title == "C"
    assert [item["usable"] for item in pool.telemetry()["keys"]] == [False, False, True]
    assert searxng.get.await_count == 0


@pytest.mark.asyncio
async def test_search_falls_back_only_after_all_keys_fail(monkeypatch):
    pool = server.KeyPool(["key-A", "key-B", "key-C"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    tavily = _client([_response(432), _response(403), _response(429)])
    searxng = _client([])
    searxng.get.return_value = _response(
        200,
        {"results": [{"title": "Fallback", "url": "https://fallback.example", "content": "used"}]},
    )

    with patch.object(server.httpx, "AsyncClient", side_effect=[tavily, searxng]):
        result = await server.search("query", MagicMock())

    assert [call.kwargs["json"]["api_key"] for call in tavily.post.await_args_list] == [
        "key-A",
        "key-B",
        "key-C",
    ]
    assert result.results[0].title == "Fallback"
    assert searxng.get.await_count == 1
    assert [item["usable"] for item in pool.telemetry()["keys"]] == [False, False, False]
