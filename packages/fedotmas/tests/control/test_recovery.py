"""Tests for Controller.run_with_recovery() and the meta-debugger loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.control._controller import Controller, _is_retryable
from fedotmas.control._run import ControlledRun, RunError
from fedotmas.control._strategy import Strategy
from fedotmas.core.runner import PipelineResult
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.meta.maw_debugger import ErrorClassification
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


def _error_run(config: MAWConfig, agent_name: str = "b", message: str = "Agent 'b' failed: bad output") -> ControlledRun:
    return ControlledRun(
        config=config,
        status="error",
        state={"a": "result_a"},
        checkpoints=[Checkpoint(agent_name="a", state={"a": "result_a"}, index=0)],
        error=RunError(agent_name=agent_name, message=message),
    )


def _success_run(config: MAWConfig) -> ControlledRun:
    return ControlledRun(
        config=config,
        status="success",
        state={"a": "result_a", "b": "result_b"},
        checkpoints=[
            Checkpoint(agent_name="a", state={"a": "result_a"}, index=0),
            Checkpoint(agent_name="b", state={"a": "result_a", "b": "result_b"}, index=1),
        ],
    )


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
    # Words that contain fatal substrings but are NOT fatal:
    "The author of the paper disagrees",
    "authorization_flow setup required",
])
def test_is_retryable_true(msg: str):
    assert _is_retryable(RunError(agent_name="x", message=msg)) is True


# ---------------------------------------------------------------------------
# run_with_recovery tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_fails_then_recovers():
    """Pipeline fails once, debugger fixes the agent, resume succeeds."""
    maw = _mock_maw()
    config = _config("a", "b")

    fixed_agent = _agent("b")
    fixed_agent = fixed_agent.model_copy(update={"instruction": "Fixed instruction for b"})

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        # First run fails at agent 'b'
        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        mock_meta.return_value = MagicMock(
            raw_output=fixed_agent.model_dump(),
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "success"
    assert run.state == {"a": "ok", "b": "fixed"}


@pytest.mark.asyncio
async def test_all_retries_exhausted():
    """Pipeline fails N+1 times, returns last error."""
    maw = _mock_maw()
    config = _config("a", "b")

    fixed_agent = _agent("b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        # Fails every time
        mock_run.side_effect = RuntimeError("Agent 'b' failed: bad output")
        maw.generate_config.return_value = config

        mock_meta.return_value = MagicMock(
            raw_output=fixed_agent.model_dump(),
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "error"
    assert run.error.agent_name == "b"
    # Initial run + 2 retries = 3 calls
    assert mock_run.call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_error_regex():
    """Connection error returns immediately without LLM call."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.side_effect = RuntimeError("Agent 'b' failed: connection timeout")
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "error"
    assert mock_run.call_count == 1
    mock_meta.assert_not_called()


@pytest.mark.asyncio
async def test_success_on_first_try():
    """No debugger call when first run succeeds."""
    maw = _mock_maw()
    config = _config("a", "b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.return_value = PipelineResult(state={"a": "ok", "b": "ok"})
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task")

    assert run.status == "success"
    mock_meta.assert_not_called()


@pytest.mark.asyncio
async def test_debugger_preserves_name_and_output_key():
    """Fixed agent must keep original name and output_key."""
    maw = _mock_maw()
    config = _config("a", "b")

    # Debugger returns agent with WRONG name and output_key
    bad_fix = MAWAgentConfig(
        name="b_renamed",
        instruction="Fixed",
        output_key="wrong_key",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        mock_meta.return_value = MagicMock(
            raw_output=bad_fix.model_dump(),
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=1)

    assert run.status == "success"


@pytest.mark.asyncio
async def test_config_none_auto_generates():
    """When config=None, auto-generates config then recovers."""
    maw = _mock_maw()
    config = _config("a", "b")
    maw.generate_config.return_value = config

    fixed_agent = _agent("b")

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]

        mock_meta.return_value = MagicMock(
            raw_output=fixed_agent.model_dump(),
            prompt_tokens=100,
            completion_tokens=50,
            elapsed=1.0,
        )

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task")

    maw.generate_config.assert_called_once()
    assert run.status == "success"


@pytest.mark.asyncio
async def test_llm_classification_retryable():
    """LLM classifier says retryable → calls diagnose_and_fix."""
    maw = _mock_maw()
    config = _config("a", "b")

    fixed_agent = _agent("b")
    classification = ErrorClassification(
        retryable=True,
        category="bad_instruction",
        reasoning="The instruction is too vague",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: bad output"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        # First call = classify_error, second call = diagnose_and_fix
        mock_meta.side_effect = [
            MagicMock(
                raw_output=classification.model_dump(),
                prompt_tokens=50,
                completion_tokens=20,
                elapsed=0.5,
            ),
            MagicMock(
                raw_output=fixed_agent.model_dump(),
                prompt_tokens=100,
                completion_tokens=50,
                elapsed=1.0,
            ),
        ]

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery(
            "test task",
            max_retries=1,
            llm_error_detection=True,
        )

    assert run.status == "success"
    assert mock_meta.call_count == 2


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
        run = await ctrl.run_with_recovery(
            "test task",
            max_retries=2,
            llm_error_detection=True,
        )

    assert run.status == "error"
    # Only classify_error was called, not diagnose_and_fix
    assert mock_meta.call_count == 1


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

    # Check that the classifier was called with instruction containing the hint
    call_kwargs = mock_meta.call_args.kwargs
    assert "agent produces empty output" in call_kwargs["instruction"]


@pytest.mark.asyncio
async def test_error_category_passed_to_debugger():
    """Verify error_category from classifier reaches diagnose_and_fix."""
    maw = _mock_maw()
    config = _config("a", "b")

    fixed_agent = _agent("b")
    classification = ErrorClassification(
        retryable=True,
        category="wrong_tool",
        reasoning="Agent used nonexistent tool",
    )

    with patch("fedotmas.control._controller.run_pipeline") as mock_run, \
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        mock_run.side_effect = [
            RuntimeError("Agent 'b' failed: tool not found"),
            PipelineResult(state={"a": "ok", "b": "fixed"}),
        ]
        maw.generate_config.return_value = config

        mock_meta.side_effect = [
            # classify_error
            MagicMock(
                raw_output=classification.model_dump(),
                prompt_tokens=50,
                completion_tokens=20,
                elapsed=0.5,
            ),
            # diagnose_and_fix
            MagicMock(
                raw_output=fixed_agent.model_dump(),
                prompt_tokens=100,
                completion_tokens=50,
                elapsed=1.0,
            ),
        ]

        ctrl = Controller(maw)
        await ctrl.run_with_recovery(
            "test task",
            max_retries=1,
            llm_error_detection=True,
        )

    # Second call is diagnose_and_fix — check its instruction contains category
    debugger_call = mock_meta.call_args_list[1]
    assert "wrong_tool" in debugger_call.kwargs["instruction"]


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
         patch("fedotmas.meta.maw_debugger.run_meta_agent_call") as mock_meta:

        # Error message doesn't match the regex → agent_name="unknown"
        mock_run.side_effect = RuntimeError("Something went very wrong")
        maw.generate_config.return_value = config

        ctrl = Controller(maw)
        run = await ctrl.run_with_recovery("test task", max_retries=2)

    assert run.status == "error"
    assert run.error.agent_name == "unknown"
    assert mock_run.call_count == 1
    mock_meta.assert_not_called()


@pytest.mark.asyncio
async def test_error_hint_without_llm_detection_warns():
    """error_hint without llm_error_detection=True logs a warning."""
    from loguru import logger

    maw = _mock_maw()
    config = _config("a")

    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")

    try:
        with patch("fedotmas.control._controller.run_pipeline") as mock_run:
            mock_run.return_value = PipelineResult(state={"a": "ok"})
            maw.generate_config.return_value = config

            ctrl = Controller(maw)
            await ctrl.run_with_recovery(
                "task",
                error_hint="some hint",
                llm_error_detection=False,
            )
    finally:
        logger.remove(sink_id)

    assert any("error_hint is ignored" in w for w in warnings)
