"""Report whether every discovered MCP server can actually be used.

Starting a server and listing its tools proves the Python side works.  It does
not prove the server can do anything: ``websearch-searxng`` advertises
``search`` with SearXNG down, and ``sandbox`` advertises ``run_code`` with no
API key.  Both then fail mid-run, which is the expensive place to find out.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

from fedotmas.mcp._config import DEFAULT_MCP_TIMEOUT_S, MCPServerConfig
from fedotmas.mcp.registry import create_toolset, get_mcp_servers

OK = "ok"
DEGRADED = "degraded"
FAIL = "fail"

_SEARXNG_DEFAULT_URL = "http://localhost:18888"
_SEARXNG_PROBE_TIMEOUT_S = 5
_ERROR_DETAIL_LEN = 120

#: Added to a server's own timeout before this command gives up on it.  Being
#: stricter than the runtime would report a clean machine's servers as broken:
#: with an empty uv cache they need well over a minute just to build a venv.
_PROBE_GRACE_S = 15

_UNBUILT_VENV_FIX = "just mcp-sync"

Registry = dict[str, MCPServerConfig] | None


@dataclass(frozen=True)
class Prerequisite:
    """Something a server needs that a successful tool listing does not prove."""

    check: Callable[[], str | None]
    fix: str
    #: True when the server cannot even start without it, so the probe is skipped.
    fatal: bool = False


@dataclass(frozen=True)
class ServerReport:
    name: str
    status: str
    seconds: float
    detail: str
    fix: str = ""


def _binary(name: str) -> Callable[[], str | None]:
    def check() -> str | None:
        return None if shutil.which(name) else f"{name} is not on PATH"

    return check


def _env_var(name: str) -> Callable[[], str | None]:
    def check() -> str | None:
        return None if os.getenv(name) else f"{name} is not set"

    return check


def _searxng() -> str | None:
    url = os.getenv("SEARXNG_URL", _SEARXNG_DEFAULT_URL).rstrip("/")
    try:
        with urllib.request.urlopen(
            f"{url}/search?q=test&format=json", timeout=_SEARXNG_PROBE_TIMEOUT_S
        ) as response:
            response.read(1)
    except (OSError, ValueError):
        return f"SearXNG is not answering at {url}"
    return None


#: Keyed by server name rather than declared in each server's ``pyproject.toml``:
#: this is the only consumer, and "the service answers" is not expressible as a
#: static declaration.  A server absent from this table gets the probe alone.
PREREQUISITES: dict[str, tuple[Prerequisite, ...]] = {
    "web-scraping": (
        Prerequisite(
            check=_binary("lightpanda"),
            fix="just lightpanda-install",
            fatal=True,
        ),
    ),
    "websearch-searxng": (Prerequisite(check=_searxng, fix="just searxng-start"),),
    "sandbox": (
        Prerequisite(
            check=_env_var("E2B_API_KEY"),
            fix="set E2B_API_KEY in .env (https://e2b.dev)",
        ),
    ),
}


@contextmanager
def _muffled(path: str) -> Iterator[None]:
    """Send this process's stdout and stderr to *path*, child output included.

    Servers spew tracebacks as their pipes close, and the registry logs each
    connection; redirecting the descriptors rather than ``sys.stderr`` is what
    catches the subprocesses too.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    sink = open(path, "ab")
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        sink.close()


async def _probe(
    name: str, timeout: float, registry: Registry = None
) -> tuple[str, str, str]:
    """Start the server and list its tools; return ``(status, detail, fix)``."""
    toolset = None
    try:
        toolset = create_toolset(name, registry=registry)
        tools = await asyncio.wait_for(toolset.get_tools(), timeout + _PROBE_GRACE_S)
    except TimeoutError:
        return FAIL, f"hung past {timeout + _PROBE_GRACE_S:.0f}s", ""
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:_ERROR_DETAIL_LEN]
        # The MCP client reports its own session-ready timeout as a connection
        # failure, and on a clean machine that is almost always dependency
        # resolution running inside the timeout rather than a broken server.
        if "timed out" in message.lower():
            detail = f"timed out after {timeout:.0f}s, venv likely unbuilt"
            return FAIL, detail, _UNBUILT_VENV_FIX
        return FAIL, f"{type(exc).__name__}: {message}", ""
    finally:
        if toolset is not None:
            try:
                await toolset.close()
            except Exception:
                pass
    count = len(tools)
    return OK, f"{count} tool{'s' if count != 1 else ''}", ""


async def check_server(name: str, registry: Registry = None) -> ServerReport:
    """Check one server's prerequisites, then its tool listing."""
    started = time.monotonic()
    # The message from this first call is what gets reported; checking again
    # later would re-run the probe and could disagree with itself.
    degraded: list[tuple[Prerequisite, str]] = []

    for prerequisite in PREREQUISITES.get(name, ()):
        problem = prerequisite.check()
        if problem is None:
            continue
        if prerequisite.fatal:
            elapsed = time.monotonic() - started
            return ServerReport(name, FAIL, elapsed, problem, prerequisite.fix)
        degraded.append((prerequisite, problem))

    timeout = _server_timeout(name, registry)
    status, detail, fix = await _probe(name, timeout, registry)
    elapsed = time.monotonic() - started

    if status == OK and degraded:
        problems = ", ".join(problem for _, problem in degraded)
        fixes = "; ".join(prerequisite.fix for prerequisite, _ in degraded)
        return ServerReport(name, DEGRADED, elapsed, f"{detail}, but {problems}", fixes)
    return ServerReport(name, status, elapsed, detail, fix)


def _server_timeout(name: str, registry: Registry) -> float:
    reg = registry if registry is not None else get_mcp_servers()
    server = reg.get(name)
    return getattr(server, "timeout", DEFAULT_MCP_TIMEOUT_S)


def _header(width: int) -> str:
    return (
        f"{'server'.ljust(width)}  {'status':<9} {'time':>6}  detail\n"
        f"{'-' * width}  {'-' * 9} {'-' * 6}  {'-' * 40}"
    )


def _row(report: ServerReport, width: int) -> str:
    return (
        f"{report.name.ljust(width)}  {report.status:<9} "
        f"{report.seconds:5.1f}s  {report.detail}"
    )


def _summary(reports: list[ServerReport], width: int) -> str:
    counts = {
        status: sum(r.status == status for r in reports)
        for status in (OK, DEGRADED, FAIL)
    }
    lines = [
        "",
        f"{counts[OK]} ok, {counts[DEGRADED]} degraded, {counts[FAIL]} failed",
    ]
    fixes = [
        f"  {r.name.ljust(width)}  {r.fix}" for r in reports if r.status != OK and r.fix
    ]
    if fixes:
        lines += ["", "to fix:", *fixes]
    return "\n".join(lines)


async def check_all(
    emit: Callable[[ServerReport], None] | None = None,
    log_path: str | None = None,
) -> list[ServerReport]:
    """Check every discovered server, one at a time, reporting as it goes.

    A cold server takes minutes, so each result is emitted as it lands rather
    than after the last one; *log_path* absorbs the noise in between.
    """
    quiet = (lambda: _muffled(log_path)) if log_path else nullcontext
    registry = get_mcp_servers()
    reports = []
    for name in sorted(registry):
        with quiet():
            report = await check_server(name, registry)
        if emit is not None:
            emit(report)
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fedotmas-doctor",
        description="Check every MCP server and the things it needs outside Python.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also print the server and registry output captured while probing",
    )
    args = parser.parse_args(argv)

    # mkstemp rather than NamedTemporaryFile: the path is opened again below,
    # which Windows refuses while the original handle is held.
    handle, path = tempfile.mkstemp(suffix=".log")
    os.close(handle)
    try:
        # Discovery logs a line per server; the header has to come out clean.
        with _muffled(path):
            width = max((len(name) for name in get_mcp_servers()), default=0)
        print(_header(width), flush=True)
        reports = asyncio.run(
            check_all(emit=lambda r: print(_row(r, width), flush=True), log_path=path)
        )
        with open(path, encoding="utf-8", errors="replace") as log:
            noise = log.read()
    finally:
        os.unlink(path)

    print(_summary(reports, width))
    if args.verbose and noise.strip():
        print("\ncaptured output:\n")
        print(noise)

    return 1 if any(r.status == FAIL for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
