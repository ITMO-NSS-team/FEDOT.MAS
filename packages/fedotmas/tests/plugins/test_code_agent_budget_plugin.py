from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fedotmas.plugins._code_agent_budget import (
    CODE_AGENT_BUDGET_STATE_KEY,
    GAIA_TASK_FILE_PATH_STATE_KEY,
    TASK_DEADLINE_STATE_KEY,
    CodeAgentBudgetPlugin,
)


def _context(*, deadline: float, agent: str = "optimizer", call_id: str = "c1"):
    ctx = MagicMock()
    ctx.state = {TASK_DEADLINE_STATE_KEY: deadline}
    ctx._invocation_context.session.id = "session-1"
    ctx._invocation_context.agent.name = agent
    ctx.function_call_id = call_id
    return ctx


def _tool():
    tool = MagicMock()
    tool.name = "solve_with_code"
    return tool


@pytest.mark.asyncio
async def test_call_count_limit_blocks_further_nested_jobs():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=1,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200)
    first_args = {"task": "Compute", "files": ["/host/data.csv"], "max_execution_seconds": 90}

    assert await plugin.before_tool_callback(
        tool=_tool(), tool_args=first_args, tool_context=ctx
    ) is None
    assert first_args["max_execution_seconds"] == 60
    assert ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]["calls"] == 1

    blocked = await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "Retry"}, tool_context=ctx
    )
    assert blocked["error_code"] == "CODE_AGENT_BUDGET_EXHAUSTED"
    assert blocked["converge_now"] is True


@pytest.mark.asyncio
async def test_cumulative_nested_wall_time_is_reserved_and_enforced():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=5,
        total_seconds_per_agent=5,
        max_seconds_per_call=5,
        task_timeout_seconds=300,
        deadline_reserve_seconds=10,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="first")
    args = {"task": "Compute", "max_execution_seconds": 5}
    assert await plugin.before_tool_callback(
        tool=_tool(), tool_args=args, tool_context=ctx
    ) is None
    assert args["max_execution_seconds"] == 5
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=args,
        tool_context=ctx,
        result={"status": "completed", "telemetry": {"duration_seconds": 3.5}},
    )

    second = {"task": "Continue", "max_execution_seconds": 5}
    ctx.function_call_id = "second"
    assert await plugin.before_tool_callback(
        tool=_tool(), tool_args=second, tool_context=ctx
    ) is None
    assert second["max_execution_seconds"] == pytest.approx(1.5)
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=second,
        tool_context=ctx,
        result={"status": "completed", "telemetry": {"duration_seconds": 1.5}},
    )

    ctx.function_call_id = "third"
    blocked = await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "Again"}, tool_context=ctx
    )
    assert blocked["error_code"] == "CODE_AGENT_BUDGET_EXHAUSTED"


@pytest.mark.asyncio
async def test_nested_execution_is_blocked_before_outer_deadline_reserve():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=3,
        total_seconds_per_agent=360,
        max_seconds_per_call=120,
        task_timeout_seconds=1800,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 35)
    args = {"task": "Compute", "max_execution_seconds": 120}
    response = await plugin.before_tool_callback(
        tool=_tool(),
        tool_args=args,
        tool_context=ctx,
    )

    assert response is None
    assert 4.9 < args["max_execution_seconds"] <= 5
    ctx = _context(deadline=time.monotonic() + 20, call_id="near")
    response = await plugin.before_tool_callback(
        tool=_tool(),
        tool_args={"task": "Compute", "max_execution_seconds": 120},
        tool_context=ctx,
    )
    assert response["error_code"] == "TASK_DEADLINE_NEAR"
    assert response["converge_now"] is True
    assert ctx.state.get(CODE_AGENT_BUDGET_STATE_KEY, {}).get("optimizer", {}).get(
        "calls", 0
    ) == 0


@pytest.mark.asyncio
async def test_code_agent_budgets_are_per_outer_agent():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=1,
        total_seconds_per_agent=10,
        max_seconds_per_call=5,
        task_timeout_seconds=100,
        deadline_reserve_seconds=5,
    )
    first = _context(deadline=time.monotonic() + 50, agent="optimizer")
    second = _context(deadline=time.monotonic() + 50, agent="verifier")
    first.state = second.state = {}

    assert await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "x"}, tool_context=first
    ) is None
    assert await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "y"}, tool_context=second
    ) is None
    assert set(first.state[CODE_AGENT_BUDGET_STATE_KEY]) == {"optimizer", "verifier"}


@pytest.mark.asyncio
async def test_original_host_file_path_is_added_to_code_agent_context():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=1,
        total_seconds_per_agent=10,
        max_seconds_per_call=5,
        task_timeout_seconds=100,
    )
    callback = MagicMock()
    callback.state = {GAIA_TASK_FILE_PATH_STATE_KEY: "/host/input.txt"}
    callback._invocation_context.agent.name = "optimizer"
    request = MagicMock()
    request.tools_dict = {"solve_with_code": _tool()}

    await plugin.before_model_callback(
        callback_context=callback,
        llm_request=request,
    )

    instruction = request.append_instructions.call_args.args[0][0]
    assert "/host/input.txt" in instruction
    assert "files=[" in instruction
    assert "Do not copy its contents" in instruction
