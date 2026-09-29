from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fedotmas.plugins._code_agent_budget import (
    CODE_AGENT_BUDGET_STATE_KEY,
    CODE_AGENT_PHASE_STATE_KEY,
    GAIA_TASK_FILE_PATH_STATE_KEY,
    TASK_DEADLINE_STATE_KEY,
    CodeAgentBudgetPlugin,
    _is_document_reading_call,
    _outcome,
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


@pytest.mark.parametrize(
    "task",
    [
        "print the entire file",
        "output sections verbatim",
        "parse diagnostic and show headers/example rows",
        "list working directory and locate the staged file",
        "report token counts / field domains so I can build the optimizer later",
    ],
)
def test_pure_file_inspection_requests_are_classified(task):
    assert _is_document_reading_call({"task": task, "files": ["input.csv"]})


@pytest.mark.parametrize(
    "task",
    [
        "parse this file and solve the optimization problem",
        "count rows satisfying condition X and return the count",
        "load the file, build the model, optimize it, and return the exact optimum",
    ],
)
def test_compute_requests_are_not_classified_as_inspection(task):
    assert not _is_document_reading_call({"task": task, "files": ["input.csv"]})


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "completed", "answer": "x"}, "success_with_result"),
        ({"status": "completed"}, "success_without_result"),
        ({"status": "incomplete", "error_code": "CODE_AGENT_TIMEOUT"}, "timeout"),
        (
            {"status": "failed", "error_code": "CODE_AGENT_RUNTIME_ERROR"},
            "execution_error",
        ),
        (
            {"status": "failed", "error_code": "CODE_AGENT_INVALID_INPUT"},
            "parse_or_validation_error",
        ),
        ({"status": "blocked"}, "blocked"),
        (
            {"status": "blocked", "error_code": "CODE_AGENT_BUDGET_EXHAUSTED"},
            "budget_exhausted",
        ),
    ],
)
def test_code_agent_outcomes_are_normalized_structurally(payload, expected):
    assert _outcome(payload)[0] == expected


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
    first_args = {
        "task": "Compute",
        "files": ["/host/data.csv"],
        "max_execution_seconds": 90,
    }

    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=first_args, tool_context=ctx
        )
        is None
    )
    assert first_args["max_execution_seconds"] == 60
    assert ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]["calls"] == 1

    blocked = await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "Retry"}, tool_context=ctx
    )
    assert blocked["error_code"] == "CODE_AGENT_BUDGET_EXHAUSTED"
    assert blocked["converge_now"] is True


@pytest.mark.asyncio
async def test_failure_allows_exactly_one_recovery_across_algorithm_rewrites():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=4,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="first")
    first = {
        "task": "Parse input and solve exactly. Run once.",
        "files": ["input.csv"],
        "max_execution_seconds": 60,
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=first, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=first,
        tool_context=ctx,
        result={
            "status": "incomplete",
            "answer": "",
            "errors": ["timeout"],
            "error_code": "CODE_AGENT_TIMEOUT",
        },
    )
    ctx.function_call_id = "second"
    repeated = {
        "task": "Replace DFS with a MILP formulation and rewrite all parser code.",
        "files": ["input.csv"],
        "max_execution_seconds": 90,
        "call_intent": "targeted_recovery",
        "recovery_target": "CODE_AGENT_TIMEOUT",
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=repeated, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=repeated,
        tool_context=ctx,
        result={
            "status": "failed",
            "error_code": "CODE_AGENT_RUNTIME_ERROR",
            "errors": ["execution error"],
        },
    )
    ctx.function_call_id = "third"
    blocked = await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "Try CP-SAT instead"}, tool_context=ctx
    )
    assert blocked["error_code"] == "CODE_AGENT_PHASE_COMPLETE"
    assert ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]["calls"] == 2


@pytest.mark.asyncio
async def test_successful_primary_closes_phase_despite_different_implementation_prompt():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=4,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="first")
    first = {"task": "Parse input and solve exactly", "files": ["input.csv"]}
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=first, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=first,
        tool_context=ctx,
        result={
            "status": "completed",
            "answer": "42",
            "evidence": ["result marker"],
        },
    )
    ctx.function_call_id = "second"
    revised = {
        "task": "Change from DFS to MILP and solve the same objective",
        "files": ["input.csv"],
    }
    blocked = await plugin.before_tool_callback(
        tool=_tool(), tool_args=revised, tool_context=ctx
    )
    assert blocked["error_code"] == "CODE_AGENT_PHASE_COMPLETE"


@pytest.mark.asyncio
async def test_verifier_candidate_gets_one_verification_computation():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=4,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, agent="verifier", call_id="v1")
    ctx.state["_fedotmas_code_agent_policy"] = {
        "verifier": {"mode": "verify_candidate"}
    }
    args = {"task": "independently check the candidate"}
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=args, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=args,
        tool_context=ctx,
        result={"status": "completed", "answer": "verified"},
    )
    ctx.function_call_id = "v2"
    blocked = await plugin.before_tool_callback(
        tool=_tool(),
        tool_args={"task": "re-solve using another backend"},
        tool_context=ctx,
    )
    assert blocked["error_code"] == "CODE_AGENT_PHASE_COMPLETE"


@pytest.mark.asyncio
async def test_verifier_unresolved_upstream_gets_recovery_and_one_correction():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=4,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, agent="verifier", call_id="v1")
    ctx.state["_fedotmas_code_agent_policy"] = {
        "verifier": {"mode": "verifier_recovery"}
    }
    first = {"task": "recover unresolved result"}
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=first, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=first,
        tool_context=ctx,
        result={"status": "failed", "error_code": "CODE_AGENT_RUNTIME_ERROR"},
    )
    ctx.function_call_id = "v2"
    correction = {
        "task": "correct the observed runtime defect only",
        "call_intent": "targeted_recovery",
        "recovery_target": "CODE_AGENT_RUNTIME_ERROR",
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=correction, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=correction,
        tool_context=ctx,
        result={"status": "failed", "error_code": "CODE_AGENT_RUNTIME_ERROR"},
    )
    ctx.function_call_id = "v3"
    blocked = await plugin.before_tool_callback(
        tool=_tool(), tool_args={"task": "try another variant"}, tool_context=ctx
    )
    assert blocked["error_code"] == "CODE_AGENT_PHASE_COMPLETE"


@pytest.mark.asyncio
async def test_code_agent_tool_is_removed_after_recovery():
    from types import SimpleNamespace

    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=4,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="primary")
    first = {"task": "solve"}
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=first, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=first,
        tool_context=ctx,
        result={"status": "failed", "error_code": "CODE_AGENT_RUNTIME_ERROR"},
    )
    ctx.function_call_id = "recovery"
    second = {
        "task": "fix the runtime error",
        "call_intent": "targeted_recovery",
        "recovery_target": "CODE_AGENT_RUNTIME_ERROR",
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=second, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=second,
        tool_context=ctx,
        result={"status": "failed", "error_code": "CODE_AGENT_RUNTIME_ERROR"},
    )
    callback = MagicMock()
    callback.state = ctx.state
    callback._invocation_context.agent.name = "optimizer"
    declaration = SimpleNamespace(name="solve_with_code")
    group = SimpleNamespace(function_declarations=[declaration])
    request = MagicMock()
    request.tools_dict = {"solve_with_code": _tool()}
    request.config.tools = [group]
    await plugin.before_model_callback(callback_context=callback, llm_request=request)
    assert request.config.tools == []
    assert ctx.state[CODE_AGENT_PHASE_STATE_KEY]["optimizer"] == "complete"


@pytest.mark.asyncio
async def test_file_dump_recommendation_does_not_consume_code_agent_budget():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=2,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="read")
    read_args = {
        "task": "Read and print the full contents of the provided CSV file.",
        "files": ["/host/data.csv"],
        "max_execution_seconds": 60,
    }

    recommendation = await plugin.before_tool_callback(
        tool=_tool(), tool_args=read_args, tool_context=ctx
    )

    assert recommendation["error_code"] == "CODE_AGENT_DOCUMENT_READING_RECOMMENDED"
    guidance = recommendation["errors"][0]
    assert "If document tools are available" in guidance
    assert "Otherwise, do not retry file inspection" in guidance
    assert "solve_with_code(files=[...])" in guidance
    assert ctx.state.get(CODE_AGENT_BUDGET_STATE_KEY, {}) == {}
    assert ctx.state[CODE_AGENT_PHASE_STATE_KEY]["optimizer"] == "inspection"
    ctx.function_call_id = "compute"
    compute_args = {
        "task": "Parse the CSV and calculate the mean value in the amount column.",
        "files": ["/host/data.csv"],
        "max_execution_seconds": 60,
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=compute_args, tool_context=ctx
        )
        is None
    )
    budget = ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]
    assert compute_args["max_execution_seconds"] == 60
    assert budget["calls"] == 1
    assert budget["reserved_seconds"] == 60


@pytest.mark.asyncio
async def test_explicit_parse_inspection_does_not_consume_computation_phase():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=3,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="inspect")
    inspect = {
        "task": "validate headers and row widths",
        "files": ["input.csv"],
        "call_intent": "inspect",
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=inspect, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=inspect,
        tool_context=ctx,
        result={"status": "completed", "answer": "3 columns, consistent rows"},
    )
    budget = ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]
    assert budget["calls"] == 1
    assert budget["phase"] == "primary_available"
    ctx.function_call_id = "compute"
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args={"task": "solve"}, tool_context=ctx
        )
        is None
    )
    budget = ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]
    assert budget["calls"] == 2
    assert budget["phase"] == "primary_running"


@pytest.mark.asyncio
async def test_nested_inspection_recommendation_refunds_reserved_budget():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=2,
        total_seconds_per_agent=120,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
    )
    ctx = _context(deadline=time.monotonic() + 200)
    args = {"task": "Compute the token counts", "files": ["input.csv"]}
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=args, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=args,
        tool_context=ctx,
        result={"error_code": "CODE_AGENT_DOCUMENT_READING_RECOMMENDED"},
    )
    budget = ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]
    assert budget["calls"] == 0
    assert budget["seconds"] == 0
    assert budget["reserved_seconds"] == 0


@pytest.mark.asyncio
async def test_inspection_recommendation_preserves_recovery_only_phase():
    plugin = CodeAgentBudgetPlugin(
        max_calls_per_agent=4,
        total_seconds_per_agent=180,
        max_seconds_per_call=60,
        task_timeout_seconds=300,
        deadline_reserve_seconds=30,
    )
    ctx = _context(deadline=time.monotonic() + 200, call_id="primary")
    primary = {"task": "Compute result", "files": ["input.csv"]}
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=primary, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=primary,
        tool_context=ctx,
        result={"status": "timed_out", "error_code": "CODE_AGENT_TIMEOUT"},
    )
    budget = ctx.state[CODE_AGENT_BUDGET_STATE_KEY]["optimizer"]
    assert budget["phase"] == "recovery_available"

    ctx.function_call_id = "inspect"
    inspection = {
        "task": "Read the headers from the file",
        "files": ["input.csv"],
        "call_intent": "inspect",
    }
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=inspection, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=inspection,
        tool_context=ctx,
        result={"error_code": "CODE_AGENT_DOCUMENT_READING_RECOMMENDED"},
    )
    assert budget["phase"] == "recovery_available"

    ctx.function_call_id = "unqualified"
    blocked = await plugin.before_tool_callback(
        tool=_tool(),
        tool_args={"task": "Compute again", "files": ["input.csv"]},
        tool_context=ctx,
    )
    assert blocked["error_code"] == "CODE_AGENT_TARGETED_RECOVERY_REQUIRED"


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
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=args, tool_context=ctx
        )
        is None
    )
    assert args["max_execution_seconds"] == 5
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=args,
        tool_context=ctx,
        result={
            "status": "failed",
            "error_code": "CODE_AGENT_RUNTIME_ERROR",
            "telemetry": {"duration_seconds": 3.5},
        },
    )

    second = {"task": "Continue", "max_execution_seconds": 5}
    second.update(
        call_intent="targeted_recovery", recovery_target="CODE_AGENT_RUNTIME_ERROR"
    )
    ctx.function_call_id = "second"
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args=second, tool_context=ctx
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=_tool(),
        tool_args=second,
        tool_context=ctx,
        result={
            "status": "failed",
            "error_code": "CODE_AGENT_RUNTIME_ERROR",
            "telemetry": {"duration_seconds": 1.5},
        },
    )
    assert second["max_execution_seconds"] == pytest.approx(1.5)
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
    assert (
        ctx.state.get(CODE_AGENT_BUDGET_STATE_KEY, {})
        .get("optimizer", {})
        .get("calls", 0)
        == 0
    )


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

    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args={"task": "x"}, tool_context=first
        )
        is None
    )
    assert (
        await plugin.before_tool_callback(
            tool=_tool(), tool_args={"task": "y"}, tool_context=second
        )
        is None
    )
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
