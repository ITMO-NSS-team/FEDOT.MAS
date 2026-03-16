import asyncio
import uuid
from typing import Any, Dict, List, Optional

import pydantic_monty
from fastmcp import FastMCP

mcp = FastMCP(name="light-sandbox")

_sessions: Dict[str, Any] = {}


def _handle_state(state: Any) -> Dict[str, Any]:
    """Helper to process the stepped execution state."""
    fn_name = getattr(state, "function_name", None)
    if fn_name:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = state
        return {
            "done": False,
            "session_id": session_id,
            "function_name": fn_name,
            "args": getattr(state, "args", []),
            "kwargs": getattr(state, "kwargs", {}),
        }
    return {"done": True, "output": getattr(state, "output", None)}


@mcp.tool
async def monty_execute_async(
    code: str,
    inputs: Optional[Dict[str, Any]] = None,
    type_check: bool = False,
    type_check_stubs: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Python code in the sandbox without side dependencies.

    Args:
        code: Python code to execute.
        inputs: Dictionary of input variables available in the global scope.
        type_check: Whether to run static type checking before execution.
        type_check_stubs: Optional type definitions (stubs) string for type checking.

    Returns:
        Dict containing 'success' status and either 'output' or 'error'.

    Examples:
        >>> await monty_execute_async("return x * 2", {"x": 21})
        {'success': True, 'output': 42}

        >>> await monty_execute_async(
        ...     code="def add(a: int) -> int: return a + 1\\nreturn add('x')",
        ...     type_check=True
        ... )
        {'success': False, 'error': 'Type checking failed...'}
    """
    try:
        input_keys = list(inputs.keys()) if inputs else []

        def _run():
            m = pydantic_monty.Monty(
                code,
                inputs=input_keys,
                type_check=type_check,
                type_check_stubs=type_check_stubs,
            )
            return m.run(inputs=inputs or {})

        result = await asyncio.to_thread(_run)
        return {"success": True, "output": result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
def monty_validate(code: str) -> Dict[str, Any]:
    """Validate Python code for syntax and structural errors without executing it.

    Args:
        code: Python code to validate.

    Examples:
        >>> monty_validate("x = 1 + 2")
        {'valid': True}

        >>> monty_validate("if x = 1:") # Syntax error
        {'valid': False, 'error': "invalid syntax at line 1"}
    """
    try:
        pydantic_monty.Monty(code)
        return {"valid": True}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@mcp.tool
def monty_start_stepped(
    code: str,
    inputs: Dict[str, Any],
    external_functions: List[str],
) -> Dict[str, Any]:
    """Begin step-by-step execution. Pauses when `external_functions` are called.

    Args:
        code: Python code calling external functions.
        inputs: Initial variables.
        external_functions: List of function names that should pause execution.

    Returns:
        Session state: either the final result or a paused state with a session_id.

    Example:
        >>> monty_start_stepped("return get_weather('London')", {}, ["get_weather"])
        {'done': False, 'session_id': 'uuid...', 'function_name': 'get_weather', 'args': ['London']}
    """
    try:
        m = pydantic_monty.Monty(
            code,
            inputs=list(inputs.keys()),
            external_functions=external_functions,
        )
        return _handle_state(m.start(inputs=inputs))
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool
def monty_resume_stepped(session_id: str, return_value: Any = None) -> Dict[str, Any]:
    """Resume a paused execution by providing the external function's return value.

    Args:
        session_id: ID returned by `monty_start_stepped` or previous resume.
        return_value: The value to return to the sandboxed code as the function result.

    Example:
        >>> monty_resume_stepped("uuid...", return_value="Sunny")
        {'done': True, 'output': 'Sunny'}
    """
    state = _sessions.pop(session_id, None)
    if state is None:
        return {"success": False, "error": f"Session '{session_id}' not found"}
    try:
        # Resume execution with the provided value
        return _handle_state(state.resume(return_value=return_value))
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
