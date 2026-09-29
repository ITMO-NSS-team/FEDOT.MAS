from __future__ import annotations

import logging
import os
from typing import Any

from e2b.exceptions import SandboxNotFoundException, TimeoutException
from e2b_code_interpreter import AsyncSandbox
from fastmcp import FastMCP

mcp = FastMCP(name="sandbox")

_sandbox: AsyncSandbox | None = None
_log = logging.getLogger(__name__)


def _sandbox_timeout() -> int:
    raw = os.getenv("FEDOTMAS_SANDBOX_TIMEOUT_SECONDS", "1800")
    try:
        timeout = int(raw)
    except ValueError:
        _log.warning("Invalid FEDOTMAS_SANDBOX_TIMEOUT_SECONDS=%r; using 1800", raw)
        return 1800
    if timeout < 1:
        _log.warning("Invalid FEDOTMAS_SANDBOX_TIMEOUT_SECONDS=%r; using 1800", raw)
        return 1800
    return timeout


async def _get_sandbox() -> AsyncSandbox:
    global _sandbox
    if _sandbox is None:
        timeout = _sandbox_timeout()
        _log.info("Creating E2B sandbox | lifetime=%ss", timeout)
        _sandbox = await AsyncSandbox.create(
            api_key=os.environ["E2B_API_KEY"], timeout=timeout
        )
    return _sandbox


def _is_expired_sandbox(exc: Exception) -> bool:
    if isinstance(exc, SandboxNotFoundException):
        return True
    message = str(exc).lower()
    return isinstance(exc, TimeoutException) and any(
        marker in message
        for marker in ("sandbox expired", "sandbox timeout", "unavailable")
    )


def _error(exc: Exception) -> dict[str, Any]:
    global _sandbox
    if _is_expired_sandbox(exc):
        _sandbox = None
        _log.error("E2B sandbox expired or is unavailable; invalidated cached sandbox")
        return {
            "success": False,
            "error_code": "SANDBOX_EXPIRED",
            "error": (
                "The E2B sandbox expired or is unavailable. Its in-memory variables "
                "and uploaded files are lost. A fresh sandbox will be created on the "
                "next call; re-upload required host files before continuing."
            ),
        }
    return {"success": False, "error": str(exc)}


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
    installed packages are preserved between calls (Jupyter-style). This executes
    inside E2B. Host paths are not accessible directly; use upload_file(path=HOST_PATH)
    once, then parse the uploaded file programmatically here. Avoid printing entire
    files through tool results. Use run_command('pip install ...') to add libraries."""
    try:
        sandbox = await _get_sandbox()
        result = await sandbox.run_code(code)
        out = _serialize_execution(result)
        return {"success": result.error is None, **out}
    except Exception as exc:
        return _error(exc)


@mcp.tool
async def run_command(command: str) -> dict[str, Any]:
    """Run a shell command inside E2B (pip install, apt-get, ls, wget, etc.).
    Host paths are not accessible; use upload_file(path=HOST_PATH) to transfer a
    host file once, then inspect it inside the sandbox without printing it all."""
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
        return _error(exc)


@mcp.tool
async def upload_file(path: str, destination: str | None = None) -> dict[str, Any]:
    """Read path from the HOST machine and upload it into E2B. The host path is
    not directly accessible to run_code or run_command. Upload once, then parse it
    programmatically in E2B; avoid printing the entire file through tool results."""
    try:
        sandbox = await _get_sandbox()
        dest = destination or os.path.basename(path)
        with open(path, "rb") as f:
            info = await sandbox.files.write(dest, f)
        return {"success": True, "path": info.path}
    except Exception as exc:
        return _error(exc)


@mcp.tool
async def download_file(sandbox_path: str, local_path: str) -> dict[str, Any]:
    """Download a file from inside E2B to the host filesystem."""
    try:
        sandbox = await _get_sandbox()
        content = await sandbox.files.read(sandbox_path, format="bytes")
        with open(local_path, "wb") as f:
            f.write(content)
        return {"success": True, "local_path": local_path}
    except Exception as exc:
        return _error(exc)


def main() -> None:
    mcp.run(show_banner=False)
