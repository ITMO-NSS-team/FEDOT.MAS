from __future__ import annotations

import importlib.util
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SERVER_PATH = (
    Path(__file__).parents[4]
    / "mcp-servers/websearch-tavily/src/mcp_websearch_tavily/server.py"
)
spec = importlib.util.spec_from_file_location("test_tavily_server", SERVER_PATH)
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


def _response(status=200, payload=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload if payload is not None else {"results": []}
    return response


def _client(responses):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(side_effect=responses)
    client.get = AsyncMock(return_value=_response(payload={"results": []}))
    return client


def test_parses_fifteen_keys_strips_whitespace_and_ignores_empty_entries():
    keys = ", ".join(f" key-{index} " for index in range(15))
    pool = server.KeyPool.from_env(
        {
            "TAVILY_API_KEYS": f" , {keys}, ,",
            "TAVILY_API_KEY": "fallback",
            "TAVILY_ROTATE_EVERY": "1",
        }
    )

    assert pool.configured
    assert [pool.begin_request().api_key for _ in range(15)] == [
        f"key-{index}" for index in range(15)
    ]
    assert pool.telemetry()["configured_keys"] == 15


def test_single_key_fallback():
    pool = server.KeyPool.from_env(
        {"TAVILY_API_KEYS": " , ", "TAVILY_API_KEY": "  single-key  "}
    )

    assert pool.begin_request().api_key == "single-key"
    assert pool.telemetry()["configured_keys"] == 1


def test_tavily_quota_statuses_disable_the_key():
    for status in (402, 432, 433):
        assert server.KEY_FAILURE_STATUSES[status] == "quota"


@pytest.mark.asyncio
async def test_missing_key_is_structured_unavailable_without_network(monkeypatch):
    pool = server.KeyPool([])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    fallback = AsyncMock(return_value=server.SearchResponse(query="query", results=[]))
    monkeypatch.setattr(server, "_searxng_fallback", fallback)

    result = await server.search("query", AsyncMock())

    fallback.assert_awaited_once_with("query", 5)
    assert result.error is None
    assert result.results == []
    assert pool.telemetry()["total_requests"] == 0


def test_rotation_occurs_after_exactly_n_actual_attempts_and_wraps():
    pool = server.KeyPool(["key1", "key2", "key3"], rotate_every=2)

    labels = [pool.begin_request().label for _ in range(7)]

    assert labels == [
        "key_0",
        "key_0",
        "key_1",
        "key_1",
        "key_2",
        "key_2",
        "key_0",
    ]
    assert pool.telemetry()["key_rotations"] == 3


def test_concurrent_request_reservations_are_safe_and_counted():
    pool = server.KeyPool([f"key-{index}" for index in range(4)], rotate_every=3)

    def reserve(_):
        return pool.begin_request().index

    with ThreadPoolExecutor(max_workers=16) as executor:
        indices = list(executor.map(reserve, range(120)))

    assert len(indices) == 120
    assert [indices.count(index) for index in range(4)] == [30, 30, 30, 30]
    telemetry = pool.telemetry()
    assert telemetry["total_requests"] == 120
    assert sum(item["request_attempts"] for item in telemetry["keys"]) == 120


@pytest.mark.asyncio
async def test_rate_limited_key_retries_with_next_usable_key(monkeypatch):
    pool = server.KeyPool(["key-secret-one", "key-secret-two"], rotate_every=50)
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client([_response(429), _response(payload={"results": []})])

    with patch.object(server.httpx, "AsyncClient", return_value=client):
        result = await server.search("test query", AsyncMock())

    assert result.error is None
    assert result.results == []
    assert [call.kwargs["json"]["api_key"] for call in client.post.await_args_list] == [
        "key-secret-one",
        "key-secret-two",
    ]
    telemetry = pool.telemetry()
    assert telemetry["total_requests"] == 2
    assert telemetry["rate_limit_failures"] == 1
    assert telemetry["provider_failures"] == 1
    assert telemetry["currently_usable_keys"] == 1


@pytest.mark.asyncio
async def test_tavily_success_does_not_call_searxng(monkeypatch):
    pool = server.KeyPool(["key-A"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    tavily = _client([_response(payload={
        "results": [{"title": "Tavily result", "url": "https://tavily.example", "content": "found"}]
    })])
    searx = _client([])

    with patch.object(server.httpx, "AsyncClient", side_effect=[tavily, searx]):
        result = await server.search("query", AsyncMock())

    assert result.results[0].title == "Tavily result"
    assert tavily.post.await_count == 1
    assert searx.get.await_count == 0


@pytest.mark.asyncio
async def test_all_keys_unavailable_falls_back_to_searxng(monkeypatch):
    pool = server.KeyPool(["key-one", "key-two"], rotate_every=50)
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client([_response(401), _response(429)])
    fallback_client = _client([_response(payload={"results": []})])

    with patch.object(server.httpx, "AsyncClient", side_effect=[client, fallback_client]):
        result = await server.search("query", AsyncMock())

    assert result.error is None
    assert result.results == []
    assert pool.telemetry()["currently_usable_keys"] == 0
    assert pool.telemetry()["auth_failures"] == 1
    assert pool.telemetry()["rate_limit_failures"] == 1


@pytest.mark.asyncio
async def test_zero_results_are_success_not_provider_failure(monkeypatch):
    pool = server.KeyPool(["key-one"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client([_response(payload={"results": []})])

    with patch.object(server.httpx, "AsyncClient", return_value=client):
        result = await server.search("no matches", AsyncMock())

    assert result.error is None
    assert result.results == []
    telemetry = pool.telemetry()
    assert telemetry["successful_requests"] == 1
    assert telemetry["zero_result_requests"] == 1
    assert telemetry["provider_failures"] == 0


@pytest.mark.asyncio
async def test_results_are_bounded_and_keep_tavily_order(monkeypatch):
    pool = server.KeyPool(["key-one"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client(
        [
            _response(
                payload={
                    "results": [
                        {
                            "title": f"title-{index}-" + "t" * 300,
                            "url": f"https://example.test/{index}",
                            "content": "s" * 1000,
                            "score": index,
                        }
                        for index in range(4)
                    ]
                }
            )
        ]
    )

    with patch.object(server.httpx, "AsyncClient", return_value=client):
        result = await server.search("query", AsyncMock(), max_results=2)

    assert [item.url for item in result.results] == [
        "https://example.test/0",
        "https://example.test/1",
    ]
    assert len(result.results[0].title) == server.MAX_TITLE_CHARS
    assert len(result.results[0].snippet) == server.MAX_SNIPPET_CHARS


@pytest.mark.asyncio
async def test_server_failure_falls_back_instead_of_becoming_empty_results(monkeypatch):
    pool = server.KeyPool(["key-one"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client([_response(503)])
    fallback_client = _client([_response(payload={"results": []})])

    with patch.object(server.httpx, "AsyncClient", side_effect=[client, fallback_client]):
        result = await server.search("query", AsyncMock())

    assert result.error is None
    assert result.results == []
    assert pool.telemetry()["provider_failures"] == 1
    assert pool.telemetry()["successful_requests"] == 0


@pytest.mark.asyncio
async def test_invalid_request_does_not_fall_back_to_searxng(monkeypatch):
    pool = server.KeyPool(["key-one"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    tavily = _client([_response(400)])
    searxng = _client([])

    with patch.object(server.httpx, "AsyncClient", side_effect=[tavily, searxng]):
        result = await server.search("invalid query", AsyncMock())

    assert result.error.code == "TAVILY_PROVIDER_ERROR"
    assert searxng.get.await_count == 0


@pytest.mark.asyncio
async def test_api_keys_never_appear_in_logs_errors_or_results(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    secret = "tvly-super-secret-token"
    pool = server.KeyPool([secret])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client(
        [
            _response(
                payload={
                    "results": [
                        {
                            "title": f"Search result {secret}",
                            "url": f"https://example.test/{secret}",
                            "content": f"snippet {secret}",
                        }
                    ]
                }
            )
        ]
    )

    caplog.set_level(logging.INFO, logger="mcp_websearch_tavily")
    with patch.object(server.httpx, "AsyncClient", return_value=client):
        result = await server.search(f"query {secret}", AsyncMock())

    rendered = result.model_dump_json()
    assert secret not in rendered
    assert secret not in caplog.text
    assert secret not in repr(pool)
    assert "[REDACTED]" in rendered
    assert "key_0" in caplog.text
    assert secret not in str(await server.telemetry())

    denied_pool = server.KeyPool([secret])
    monkeypatch.setattr(server, "KEY_POOL", denied_pool)
    denied = _response(401)
    denied.text = f"Invalid key: {secret}"
    fallback_client = _client([_response(payload={"results": []})])
    with patch.object(server.httpx, "AsyncClient", side_effect=[_client([denied]), fallback_client]):
        error_result = await server.search("query", AsyncMock())

    assert error_result.error is None
    assert secret not in error_result.model_dump_json()
    assert secret not in caplog.text


@pytest.mark.parametrize("value", ["not-an-integer", "0", "-4"])
def test_invalid_rotation_interval_uses_default(value, caplog):
    caplog.set_level(logging.WARNING, logger="mcp_websearch_tavily")

    pool = server.KeyPool.from_env(
        {"TAVILY_API_KEYS": "key-one,key-two", "TAVILY_ROTATE_EVERY": value}
    )

    assert pool.rotate_every == server.DEFAULT_ROTATE_EVERY
    assert "Invalid TAVILY_ROTATE_EVERY" in caplog.text
    assert "key-one" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [(432, 429, 200), (432, 403, 429)])
async def test_actual_search_rotates_failed_keys_in_order_then_falls_back(
    monkeypatch, statuses
):
    keys = ["key-A", "key-B", "key-C"]
    pool = server.KeyPool(keys)
    monkeypatch.setattr(server, "KEY_POOL", pool)
    responses = [_response(status) for status in statuses]
    if statuses[-1] == 200:
        responses[-1] = _response(
            payload={"results": [{"title": "C result", "url": "https://c.example", "content": "from C"}]}
        )
    tavily = _client(responses)
    searx = _client([])
    searx.get.return_value = _response(payload={"results": [{"title": "fallback", "url": "https://fallback.example", "content": "fallback"}]})

    with patch.object(server.httpx, "AsyncClient", side_effect=[tavily, searx]):
        result = await server.search("query", AsyncMock())

    sent_keys = [call.kwargs["json"]["api_key"] for call in tavily.post.await_args_list]
    assert sent_keys == keys[: len(statuses)]
    telemetry = pool.telemetry()
    assert [item["usable"] for item in telemetry["keys"]] == [
        status not in server.KEY_FAILURE_STATUSES for status in statuses
    ]
    if statuses[-1] == 200:
        assert result.results[0].title == "C result"
        assert searx.get.await_count == 0
    else:
        assert searx.get.await_count == 1
        assert result.results[0].title == "fallback"


@pytest.mark.asyncio
async def test_exhausted_keys_are_skipped_on_later_search(monkeypatch):
    pool = server.KeyPool(["key-A", "key-B", "key-C"])
    monkeypatch.setattr(server, "KEY_POOL", pool)
    client = _client([
        _response(432), _response(429),
        _response(payload={"results": [{"title": "C first", "url": "https://c.example"}]}),
        _response(payload={"results": [{"title": "C second", "url": "https://c.example"}]}),
    ])
    with patch.object(server.httpx, "AsyncClient", return_value=client):
        first = await server.search("first", AsyncMock())
        second = await server.search("second", AsyncMock())

    assert [call.kwargs["json"]["api_key"] for call in client.post.await_args_list] == [
        "key-A", "key-B", "key-C", "key-C"
    ]
    assert first.results[0].title == "C first"
    assert second.results[0].title == "C second"
    assert [item["usable"] for item in pool.telemetry()["keys"]] == [False, False, True]
