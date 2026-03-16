import pytest
from fastmcp.client import Client

from mcp_sandbox_light.server import mcp


@pytest.fixture
async def client():
    async with Client(transport=mcp) as c:
        yield c


async def test_execute_simple(client: Client):
    result = await client.call_tool("execute", {"code": "return 2 + 2"})
    data = result.data
    assert data["success"] is True
    assert data["output"] == 4


async def test_execute_with_inputs(client: Client):
    result = await client.call_tool("execute", {"code": "return x * 2", "inputs": {"x": 21}})
    data = result.data
    assert data["success"] is True
    assert data["output"] == 42


async def test_execute_syntax_error(client: Client):
    result = await client.call_tool("execute", {"code": "if = bad"})
    assert result.data["success"] is False


async def test_execute_runtime_error(client: Client):
    result = await client.call_tool("execute", {"code": "return 1 / 0"})
    assert result.data["success"] is False


async def test_execute_timeout(client: Client):
    result = await client.call_tool("execute", {"code": "while True: pass", "timeout": 0.5})
    assert result.data["success"] is False


async def test_repl_create(client: Client):
    result = await client.call_tool("repl", {"code": "x = 42"})
    data = result.data
    assert data["success"] is True
    assert "session_id" in data


async def test_repl_state_preserved(client: Client):
    r1 = await client.call_tool("repl", {"code": "x = 10"})
    sid = r1.data["session_id"]
    r2 = await client.call_tool("repl", {"code": "return x + 5", "session_id": sid})
    assert r2.data["output"] == 15


async def test_repl_unknown_session_creates_new(client: Client):
    result = await client.call_tool("repl", {"code": "return 1", "session_id": "nonexistent"})
    data = result.data
    assert data["success"] is True
    assert data["session_id"] != "nonexistent"
