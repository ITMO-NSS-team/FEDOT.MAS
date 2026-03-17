from __future__ import annotations

import os
from typing import Any

from e2b_code_interpreter import AsyncSandbox
from fastmcp import FastMCP

mcp = FastMCP(name="sandbox")

_sandbox: AsyncSandbox | None = None


async def _get_sandbox() -> AsyncSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = await AsyncSandbox.create(api_key=os.environ["E2B_API_KEY"])
    return _sandbox


def _serialize_execution(exc) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if exc.logs.stdout:
        out["stdout"] = "\n".join(exc.logs.stdout)
    if exc.logs.stderr:
        out["stderr"] = "\n".join(exc.logs.stderr)
    if exc.error:
        out["error"] = {
            "name": exc.error.name,
            "value": exc.error.value,
            "traceback": exc.error.traceback,
        }
    for r in exc.results:
        if r.text is not None:
            out.setdefault("results", []).append(r.text)
        if r.png is not None:
            out.setdefault("images", []).append({"format": "png", "data": r.png})
        if r.json is not None:
            out.setdefault("json_results", []).append(r.json)
    return out


@mcp.tool
async def run_code(code: str) -> dict[str, Any]:
    """Execute Python code in a persistent sandbox. Variables, imports, and
    installed packages are preserved between calls (Jupyter-style).
    Use run_command('pip install ...') to add libraries first."""
    try:
        sandbox = await _get_sandbox()
        result = await sandbox.run_code(code)
        out = _serialize_execution(result)
        return {"success": result.error is None, **out}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
async def run_command(command: str) -> dict[str, Any]:
    """Run a shell command in the sandbox (pip install, apt-get, ls, wget, etc.)."""
    try:
        sandbox = await _get_sandbox()
        result = await sandbox.commands.run(command)
        out: dict[str, Any] = {
            "success": result.exit_code == 0,
            "exit_code": result.exit_code,
        }
        if result.stdout:
            out["stdout"] = result.stdout
        if result.stderr:
            out["stderr"] = result.stderr
        if result.error:
            out["error"] = result.error
        return out
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
async def upload_file(path: str, destination: str | None = None) -> dict[str, Any]:
    """Upload a local file to the sandbox."""
    try:
        sandbox = await _get_sandbox()
        dest = destination or os.path.basename(path)
        with open(path, "rb") as f:
            info = await sandbox.files.write(dest, f)
        return {"success": True, "path": info.path}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
async def download_file(sandbox_path: str, local_path: str) -> dict[str, Any]:
    """Download a file from the sandbox to the local filesystem."""
    try:
        sandbox = await _get_sandbox()
        content = await sandbox.files.read(sandbox_path, format="bytes")
        with open(local_path, "wb") as f:
            f.write(content)
        return {"success": True, "local_path": local_path}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def main() -> None:
    mcp.run(show_banner=False)
