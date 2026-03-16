import asyncio
import uuid
from typing import Any

import pydantic_monty
from fastmcp import FastMCP
from pydantic_monty import MontyError, MontyRepl

mcp = FastMCP(name="sandbox-light")

_repls: dict[str, MontyRepl] = {}


@mcp.tool
async def execute(
    code: str,
    inputs: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    """Execute Python code in a sandbox. Only builtins available — no imports, no third-party libraries, no file access."""
    try:
        input_keys = list(inputs.keys()) if inputs else []

        def _run():
            m = pydantic_monty.Monty(code, inputs=input_keys)
            kwargs: dict[str, Any] = {"limits": {"max_duration_secs": timeout}}
            if inputs:
                kwargs["inputs"] = inputs
            return m.run(**kwargs)

        result = await asyncio.to_thread(_run)
        return {"success": True, "output": result}
    except MontyError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
async def repl(
    code: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Execute code in a persistent REPL session. State preserved between calls. Only builtins available — no imports, no third-party libraries, no file access."""
    try:
        if session_id and session_id in _repls:
            output = await asyncio.to_thread(_repls[session_id].feed, code)
            return {"success": True, "session_id": session_id, "output": output}

        sid = str(uuid.uuid4())

        def _create():
            return MontyRepl.create(code)

        r, output = await asyncio.to_thread(_create)
        _repls[sid] = r
        return {"success": True, "session_id": sid, "output": output}
    except MontyError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def main() -> None:
    mcp.run(show_banner=False)
