from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from e2b.exceptions import SandboxNotFoundException
from mcp_sandbox import server
from mcp_sandbox.server import mcp


async def test_server_name():
    assert mcp.name == "sandbox"


async def test_tools_registered():
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {"run_code", "run_command", "upload_file", "download_file"}


async def test_run_code_docstring_mentions_persistent():
    tool = await mcp.get_tool("run_code")
    assert "persistent" in tool.description.lower()


async def test_run_code_docstring_mentions_jupyter():
    tool = await mcp.get_tool("run_code")
    assert "jupyter" in tool.description.lower()


async def test_run_command_docstring_mentions_pip():
    tool = await mcp.get_tool("run_command")
    assert "pip" in tool.description.lower()


@pytest.mark.asyncio
async def test_sandbox_creation_uses_configured_lifetime(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_SANDBOX_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.setattr(server, "_sandbox", None)
    create = AsyncMock(return_value=object())
    monkeypatch.setattr(server.AsyncSandbox, "create", create)

    await server._get_sandbox()

    create.assert_awaited_once_with(api_key="test-key", timeout=1800)


def test_expired_sandbox_invalidates_cached_instance(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(server, "_sandbox", sentinel)

    result = server._error(SandboxNotFoundException("sandbox not found"))

    assert result["error_code"] == "SANDBOX_EXPIRED"
    assert "fresh sandbox" in result["error"]
    assert server._sandbox is None


async def test_tool_descriptions_explain_host_upload_boundary():
    upload = await mcp.get_tool("upload_file")
    run_code = await mcp.get_tool("run_code")
    assert "HOST" in upload.description
    assert "inside E2B" in run_code.description
