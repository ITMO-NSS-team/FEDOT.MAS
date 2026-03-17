from __future__ import annotations

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
