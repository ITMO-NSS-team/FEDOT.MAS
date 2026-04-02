"""Tests for Controller.run_with_recovery() and the fix tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._controller import Controller, _is_retryable
from fedotmas.control._run import ControlledRun, RunError
from fedotmas.control.fixes._guardrails import run_config_guardrails
from fedotmas.core.runner import PipelineResult
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.meta.maw_debugger import ErrorClassification, OutputEvaluation
from fedotmas.plugins._checkpoint import Checkpoint


def _agent(name: str) -> MAWAgentConfig:
    return MAWAgentConfig(name=name, instruction=f"Do {name}", output_key=name)


def _config(*names: str) -> MAWConfig:
    agents = [_agent(n) for n in names]
    pipeline = MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name=n) for n in names],
    )
    return MAWConfig(agents=agents, pipeline=pipeline)


def _mock_maw() -> MagicMock:
    maw = MagicMock()
    maw._session_service = None
    maw._memory_service = None
    maw.meta_model = None
    maw.worker_models = None
    maw.temperature = None
    maw.mcp_registry = None
    maw.generate_config = AsyncMock()
    maw.build = MagicMock(return_value=MagicMock())
    return maw


# ---------------------------------------------------------------------------
# _is_retryable unit tests
# ---------------------------------------------------------------------------


def test_is_retryable_none():
    assert _is_retryable(None) is False


@pytest.mark.parametrize("msg", [
    "connection refused",
    "Request timeout after 30s",
    "auth error: invalid token",
    "Unauthorized access",
    "rate limit exceeded",
    "quota exhausted for project",
])
def test_is_retryable_fatal(msg: str):
    assert _is_retryable(RunError(agent_name="x", message=msg)) is False


@pytest.mark.parametrize("msg", [
    "Agent 'b' failed: bad output",
    "Invalid JSON response from model",
    "Agent produced empty result",
    "Tool 'nonexistent_tool' not found",
    "The author of the paper disagrees",
    "authorization_flow setup required",
])
def test_is_retryable_true(msg: str):
    assert _is_retryable(RunError(agent_name="x", message=msg)) is True


# ---------------------------------------------------------------------------
# run_config_guardrails unit tests
# ---------------------------------------------------------------------------


def test_guardrails_valid_config():
    config = _config("a", "b")
    assert run_config_guardrails(config) == []


def test_guardrails_unused_agent():
    agents = [_agent("a"), _agent("b"), _agent("unused")]
    pipeline = MAWStepConfig(
        type="sequential",
        children=[MAWStepConfig(agent_name="a"), MAWStepConfig(agent_name="b")],
    )
    config = MAWConfig(agents=agents, pipeline=pipeline)
    errors = run_config_guardrails(config)
    assert len(errors) == 1
    assert "unused" in errors[0].lower() or "Unused" in errors[0]


def test_guardrails_terminal_parallel():
    agents = [_agent("a"), _agent("b")]
    pipeline = MAWStepConfig(
        type="parallel",
        children=[MAWStepConfig(agent_name="a"), MAWStepConfig(agent_name="b")],
    )
    config = MAWConfig(agents=agents, pipeline=pipeline)
    errors = run_config_guardrails(config)
    assert any("parallel" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# fix_instruction tool unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fix_instruction_updates_state():
    from fedotmas.control.fixes._fix_instruction import fix_instruction

    config = _config("a", "b")
    mock_ctx = MagicMock()
    mock_ctx.state = {"config": config.model_dump_json()}

    result = await fix_instruction(
        tool_context=mock_ctx,
        agent_name="b",
        new_instruction="Fixed instruction for b",
        reasoning="instruction was too vague",
    )

    assert "Updated instruction" in result
    new_config = MAWConfig.model_validate_json(mock_ctx.state["config"])
    agent_b = next(a for a in new_config.agents if a.name == "b")
    assert "Fixed instruction" in agent_b.instruction


@pytest.mark.asyncio
async def test_fix_instruction_unknown_agent():
    from fedotmas.control.fixes._fix_instruction import fix_instruction

    config = _config("a", "b")
    mock_ctx = MagicMock()
    mock_ctx.state = {"config": config.model_dump_json()}

    result = await fix_instruction(
        tool_context=mock_ctx,
        agent_name="nonexistent",
        new_instruction="whatever",
        reasoning="test",
    )

    assert "Error" in result
    assert "nonexistent" in result


# ---------------------------------------------------------------------------
# guardrail_validate_config callback unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_callback_passes_valid():
    from fedotmas.control.fixes._guardrails import guardrail_validate_config

    config = _config("a", "b")
    mock_tool = MagicMock()
    mock_tool.name = "fix_instruction"
    mock_ctx = MagicMock()
    mock_ctx.state = {"config": config.model_dump_json()}

    result = await guardrail_validate_config(
        mock_tool, {}, mock_ctx, {},
    )
    assert result is None


@pytest.mark.asyncio
async def test_guardrail_callback_catches_invalid():
    from fedotmas.control.fixes._guardrails import guardrail_validate_config

    mock_tool = MagicMock()
    mock_tool.name = "fix_instruction"
    mock_ctx = MagicMock()
    mock_ctx.state = {"config": {"agents": [], "pipeline": {"type": "agent"}}}

    result = await guardrail_validate_config(
        mock_tool, {}, mock_ctx, {},
    )
    assert result is not None
    assert "error" in result


# ---------------------------------------------------------------------------
# run_with_recovery integration tests
# ---------------------------------------------------------------------------


def _mock_debugger_returning(new_config: MAWConfig):
    """Patch _run_debugger to return a fixed config."""
    return patch.object(
        Controller,
        "_run_debugger",
        new=AsyncMock(return_value=new_config),
    )


@pytest.mark.asyncio
async def test_happy_path_fails_then_recovers():
    """Pipeline fails once, debugger fixes the agent, resume succeeds."""
    maw = _mock_maw()
    config = _config("a", "b")
    fixed_config = config.replace_agent(
        "b", _agent("b").model_copy(update={"instruction": "Fixed b"}),
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(fixed_config):

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "success"
    assert run.state == {"a": "ok", "b": "fixed"}


@pytest.mark.asyncio
async def test_all_retries_exhausted():
    """Pipeline fails N+1 times, returns last error."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config):

        mock_run.side_effect = RuntimeError("Agent 'b' failed: bad output")
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "error"
    assert run.error.agent_name == "b"
    assert mock_run.call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_error_regex():
    """Connection error returns immediately without debugger call."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = RuntimeError("Agent 'b' failed: connection timeout")
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "error"
    assert mock_run.call_count == 1
    mock_debugger.assert_not_called()


@pytest.mark.asyncio
async def test_success_on_first_try():
    """No debugger call when first run succeeds."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.return_value = PipelineResult(state={"a": "ok", "b": "ok"})
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task")

    assert run.status == "success"
    mock_debugger.assert_not_called()


@pytest.mark.asyncio
async def test_config_none_auto_generates():
    """When config=None, auto-generates config then recovers."""
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config):

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task")

    maw.generate_config.assert_called_once()
    assert run.status == "success"


@pytest.mark.asyncio
async def test_llm_classification_retryable():
    """LLM classifier says retryable → calls debugger."""
    maw = _mock_maw()
    config = _config("a", "b")

    classification = ErrorClassification(
        retryable=True,
        category="bad_instruction",
        reasoning="The instruction is too vague",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        mock_meta.return_value = MagicMock(
            raw_output=classification.model_dump(),
            prompt_tokens=50,
            completion_tokens=20,
            elapsed=0.5,
        )

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            max_retries=1,
            llm_error_detection=True,
        )

    assert run.status == "success"
    mock_debugger.assert_called_once()


@pytest.mark.asyncio
async def test_llm_classification_fatal():
    """LLM classifier says non-retryable → returns immediately."""
    maw = _mock_maw()
    config = _config("a", "b")

    classification = ErrorClassification(
        retryable=False,
        category="auth_error",
        reasoning="The API key is invalid",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = RuntimeError("Agent 'b' failed: some error")
        maw.generate_config.return_value = config

        mock_meta.return_value = MagicMock(
            raw_output=classification.model_dump(),
            prompt_tokens=50,
            completion_tokens=20,
            elapsed=0.5,
        )

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            max_retries=2,
            llm_error_detection=True,
        )

    assert run.status == "error"
    mock_debugger.assert_not_called()


@pytest.mark.asyncio
async def test_error_hint_passed_to_classifier():
    """Verify error_hint reaches the classifier prompt."""
    maw = _mock_maw()
    config = _config("a", "b")

    classification = ErrorClassification(
        retryable=False,
        category="custom",
        reasoning="matched hint",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.side_effect = RuntimeError("Agent 'b' failed: some error")
        maw.generate_config.return_value = config

        mock_meta.return_value = MagicMock(
            raw_output=classification.model_dump(),
            prompt_tokens=50,
            completion_tokens=20,
            elapsed=0.5,
        )

        ctrl = Controller(maw)
        await ctrl.run_with_recovery(
            "test task",
            llm_error_detection=True,
            error_hint="agent produces empty output",
        )

    call_kwargs = mock_meta.call_args.kwargs
    assert "agent produces empty output" in call_kwargs["instruction"]


@pytest.mark.asyncio
async def test_max_retries_negative_raises():
    """max_retries < 0 raises ValueError."""
    maw = _mock_maw()
    ctrl = Controller(maw)
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        await ctrl.run_with_recovery("task", max_retries=-1)


@pytest.mark.asyncio
async def test_unknown_agent_skips_recovery():
    """Error with unrecognised agent name returns without recovery attempt."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = RuntimeError("Something went very wrong")
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "error"
    assert run.error.agent_name == "unknown"
    assert mock_run.call_count == 1
    mock_debugger.assert_not_called()


@pytest.mark.asyncio
async def test_custom_fix_tool_via_di():
    """A custom fix tool passed via fix_tools is forwarded to _run_debugger."""
    maw = _mock_maw()
    config = _config("a", "b")

    async def my_custom_fix(tool_context, agent_name: str) -> str:
        return "custom fix applied"

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        await ctrl.run_with_recovery(
            "test task",
            fix_tools=[my_custom_fix],
        )

    # _run_debugger was called with fix_tools=[my_custom_fix]
    call_kwargs = mock_debugger.call_args.kwargs
    assert call_kwargs["fix_tools"] == [my_custom_fix]


@pytest.mark.asyncio
async def test_default_fix_tools_is_none():
    """When fix_tools not specified, _run_debugger receives None (uses default)."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        await ctrl.run_with_recovery("test task")

    call_kwargs = mock_debugger.call_args.kwargs
    assert call_kwargs["fix_tools"] is None


@pytest.mark.asyncio
async def test_checks_triggers_recovery():
    """A check function that returns an error triggers debugger recovery."""
    maw = _mock_maw()
    config = _config("a", "b")
    fixed_config = config.replace_agent(
        "b", _agent("b").model_copy(update={"instruction": "Fixed b"}),
    )

    def check_b(state: dict) -> str | None:
        if state.get("b") != "correct":
            return "Agent b returned wrong value, fix instruction"
        return None

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(fixed_config) as mock_debugger:

        from fedotmas.plugins._eval import EvaluationError

        mock_run.side_effect = [
            EvaluationError("b", "Agent b returned wrong value, fix instruction"),
            PipelineResult(state={"a": "ok", "b": "correct"}),
        ]
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            checks={"b": check_b},
        )

    assert run.status == "success"
    mock_debugger.assert_called_once()


@pytest.mark.asyncio
async def test_checks_pass_no_recovery():
    """When checks pass, no recovery is triggered."""
    maw = _mock_maw()
    config = _config("a", "b")

    def check_b(state: dict) -> str | None:
        return None

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.return_value = PipelineResult(state={"a": "ok", "b": "ok"})
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            checks={"b": check_b},
        )

    assert run.status == "success"
    mock_debugger.assert_not_called()


@pytest.mark.asyncio
async def test_error_hint_eval_runs_with_zero_retries():
    """error_hint eval must run even when max_retries=0 (no recovery, but eval happens)."""
    maw = _mock_maw()
    config = _config("a", "b")

    eval_fail = OutputEvaluation(
        passed=False, agent_name="b", reasoning="Output is wrong",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.evaluate_output", new=AsyncMock(return_value=eval_fail)) as mock_eval, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.return_value = PipelineResult(state={"a": "ok", "b": "bad"})
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            max_retries=0,
            error_hint="b should return 'fixed'",
        )

    assert run.status == "error"
    assert run.error is not None
    assert run.error.agent_name == "b"
    mock_eval.assert_called_once()
    mock_debugger.assert_not_called()


@pytest.mark.asyncio
async def test_error_hint_triggers_llm_eval():
    """When error_hint is set and pipeline succeeds, LLM eval can trigger recovery."""
    maw = _mock_maw()
    config = _config("a", "b")

    eval_fail = OutputEvaluation(
        passed=False, agent_name="b", reasoning="Output is wrong",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.evaluate_output", new=AsyncMock(return_value=eval_fail)) as mock_eval, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.side_effect = [
            PipelineResult(state={"a": "ok", "b": "bad"}),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            max_retries=1,
            error_hint="b should return 'fixed'",
        )

    mock_eval.assert_called()
    mock_debugger.assert_called_once()


@pytest.mark.asyncio
async def test_error_hint_passes_llm_eval():
    """When LLM eval passes, no recovery is triggered."""
    maw = _mock_maw()
    config = _config("a", "b")

    eval_pass = OutputEvaluation(
        passed=True, agent_name="", reasoning="Output is correct",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.evaluate_output", new=AsyncMock(return_value=eval_pass)) as mock_eval, \
         _mock_debugger_returning(config) as mock_debugger:

        mock_run.return_value = PipelineResult(state={"a": "ok", "b": "ok"})
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            error_hint="b should return ok",
        )

    assert run.status == "success"
    mock_eval.assert_called_once()
    mock_debugger.assert_not_called()
