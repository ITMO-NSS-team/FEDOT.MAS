import asyncio
import uuid
from typing import Any

import pydantic_monty
from fastmcp import FastMCP
from pydantic_monty import CollectString, MontyError, MontyRepl

mcp = FastMCP(name="sandbox-light")

_repls: dict[str, MontyRepl] = {}


@mcp.tool
async def execute(
    code: str,
    inputs: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    """Execute Python code in a sandbox. Only the standard library is available — no third-party libraries, no file access.

    Returns the value of the last expression in ``output`` and anything the code
    printed in ``stdout``.
    """
    try:
        input_keys = list(inputs.keys()) if inputs else []

        def _run():
            # CollectString captures ``print`` output via the native callback so
            # it is returned to the caller instead of leaking to the process
            # stdout, which is the MCP stdio JSON-RPC channel.
            collector = CollectString()
            m = pydantic_monty.Monty(code, inputs=input_keys)
            kwargs: dict[str, Any] = {
                "limits": {"max_duration_secs": timeout},
                "print_callback": collector,
            }
            if inputs:
                kwargs["inputs"] = inputs
            result = m.run(**kwargs)
            return result, collector.output

        result, stdout = await asyncio.to_thread(_run)
        return {"success": True, "output": result, "stdout": stdout}
    except MontyError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
async def repl(
    code: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Execute code in a persistent REPL session. State preserved between calls. Only the standard library is available — no third-party libraries, no file access.

    Returns the value of the last expression in ``output`` and anything the code
    printed in ``stdout``.
    """
    try:
        sid = session_id if session_id and session_id in _repls else str(uuid.uuid4())
        repl_session = _repls.get(sid)
        if repl_session is None:
            repl_session = MontyRepl()
            _repls[sid] = repl_session

        def _feed():
            # ``feed_run`` executes one snippet against the persisted session
            # state; CollectString diverts ``print`` away from the JSON-RPC
            # stdio channel and back to the caller.
            collector = CollectString()
            result = repl_session.feed_run(code, print_callback=collector)
            return result, collector.output

        result, stdout = await asyncio.to_thread(_feed)
        return {"success": True, "session_id": sid, "output": result, "stdout": stdout}
    except MontyError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def main() -> None:
    mcp.run(show_banner=False)
