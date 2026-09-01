"""Runner edge-case tests — mock Runner, test event processing."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tenacity
from tenacity import RetryError

from fedotmas.core.runner import (
    SEARCH_LIMIT_RECOVERY_PROMPT,
    PipelineResult,
    _enter_finalize_mode,
    _is_search_limit_exceeded,
    run_pipeline,
)
from fedotmas.plugins import WebSearchLimitExceeded, WebSearchLimitPlugin

from .conftest import FakeActions, FakeEvent, FakeSession, FakeUsageMetadata


def _retry_error_wrapping(exc: BaseException) -> RetryError:
    """Build a ``RetryError`` whose last attempt raised *exc* (as in production)."""
    future = tenacity.Future(1)
    future.set_exception(exc)
    return RetryError(future)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_agent(name: str = "root") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _patch_runner(events: list[FakeEvent]):
    """Return patches for Runner as async context manager yielding events.

    Also patches ``App`` so that ``MagicMock`` agents pass Pydantic
    validation when ``run_pipeline`` wraps them in an ``App``.
    """

    async def fake_run_async(**_kwargs):
        for e in events:
            yield e

    runner_instance = MagicMock()
    runner_instance.run_async = fake_run_async

    @asynccontextmanager
    async def fake_runner_cm(*_args, **_kwargs):
        yield runner_instance

    class _FakeApp:
        """Lightweight stand-in for App that accepts MagicMock agents."""

        def __init__(self, *, name: str = "fedotmas", root_agent, plugins=None):
            self.name = name
            self.root_agent = root_agent
            self.plugins = plugins or []

    runner_patch = patch("fedotmas.core.runner.Runner", side_effect=fake_runner_cm)
    app_patch = patch("fedotmas.core.runner.App", _FakeApp)

    from contextlib import ExitStack

    @asynccontextmanager
    async def combined():
        with ExitStack() as stack:
            stack.enter_context(app_patch)
            mock = stack.enter_context(runner_patch)
            yield mock

    return combined()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPartialEventsSkipped:
    """Rule 1: partial events do not affect result."""

    @pytest.mark.asyncio
    async def test_partial_skipped(self, mock_session_service):
        events = [
            FakeEvent(partial=True),
            FakeEvent(partial=True),
        ]
        async with _patch_runner(events):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
            )
        assert isinstance(result, PipelineResult)
        assert result.total_prompt_tokens == 0
        assert result.total_completion_tokens == 0


class TestTokenAccumulation:
    """Rule 2: usage_metadata tokens are summed."""

    @pytest.mark.asyncio
    async def test_tokens_summed(self, mock_session_service):
        events = [
            FakeEvent(
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=10, candidates_token_count=5
                )
            ),
            FakeEvent(
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=20, candidates_token_count=15
                )
            ),
        ]
        async with _patch_runner(events):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
            )
        assert result.total_prompt_tokens == 30
        assert result.total_completion_tokens == 20


class TestLlmErrorRaises:
    """Rule 3: event with error_code → RuntimeError."""

    @pytest.mark.asyncio
    async def test_error_raises(self, mock_session_service):
        events = [
            FakeEvent(error_code="RATE_LIMIT", error_message="Too many requests"),
        ]
        async with _patch_runner(events):
            with pytest.raises(RuntimeError, match="RATE_LIMIT"):
                await run_pipeline(
                    _fake_agent(),
                    "hello",
                    session_service=mock_session_service,
                )


class TestMaxTokensIsNotFatal:
    """Rule 3a: one step running out of budget must not discard the others."""

    @pytest.mark.asyncio
    async def test_max_tokens_does_not_abort_the_pipeline(self, mock_session_service):
        from google.genai import types

        events = [
            FakeEvent(
                author="calculator",
                error_code=types.FinishReason.MAX_TOKENS,
                error_message="Maximum tokens reached",
            ),
            FakeEvent(
                author="writer",
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=7, candidates_token_count=3
                ),
            ),
        ]
        async with _patch_runner(events):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
            )

        # the later step still ran and was still counted
        assert result.total_prompt_tokens == 7
        assert result.total_completion_tokens == 3


class TestSessionLostAfterRun:
    """Rule 4: get_session returns None → RuntimeError."""

    @pytest.mark.asyncio
    async def test_session_lost(self, mock_session_service):
        mock_session_service.get_session = AsyncMock(return_value=None)
        events: list[FakeEvent] = []
        async with _patch_runner(events):
            with pytest.raises(RuntimeError, match="lost"):
                await run_pipeline(
                    _fake_agent(),
                    "hello",
                    session_service=mock_session_service,
                )


class TestInitialStateMerged:
    """Rule 5: initial_state is passed into create_session."""

    @pytest.mark.asyncio
    async def test_initial_state(self, mock_session_service):
        events: list[FakeEvent] = []
        async with _patch_runner(events):
            await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
                initial_state={"k": "v"},
            )
        call_kwargs = mock_session_service.create_session.call_args
        state_arg = call_kwargs.kwargs.get("state") or call_kwargs[1].get("state")
        assert state_arg["user_query"] == "hello"
        assert state_arg["k"] == "v"


class TestStateInResult:
    """Rule 6: final session state → PipelineResult.state."""

    @pytest.mark.asyncio
    async def test_state_returned(self, mock_session_service):
        final_session = FakeSession(state={"answer": "42", "user_query": "q"})
        mock_session_service.get_session = AsyncMock(return_value=final_session)
        events: list[FakeEvent] = []
        async with _patch_runner(events):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
            )
        assert result.state["answer"] == "42"


class TestEmptyOutputWarning:
    """Rule 7: state_delta with None value → no crash (logging in plugin)."""

    @pytest.mark.asyncio
    async def test_none_value_no_crash(self, mock_session_service):
        events = [
            FakeEvent(actions=FakeActions(state_delta={"key": None})),
        ]
        async with _patch_runner(events):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
            )
        assert isinstance(result, PipelineResult)


class TestPluginsPassed:
    """Rule 8: plugins list is passed through to App (then to Runner)."""

    @pytest.mark.asyncio
    async def test_plugins_forwarded(self, mock_session_service):
        from google.adk.plugins import BasePlugin

        class StubPlugin(BasePlugin):
            def __init__(self):
                super().__init__(name="stub")

        plugin = StubPlugin()
        events: list[FakeEvent] = []
        async with _patch_runner(events) as runner_patch:
            await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
                plugins=[plugin],
            )
        # Runner receives the App wrapping the agent and plugins
        call_kwargs = runner_patch.call_args
        app = call_kwargs.kwargs.get("app")
        assert app is not None
        assert plugin in app.plugins


class TestSearchLimitRecovery:
    """Search-limit tool exceptions trigger final-answer synthesis."""

    @pytest.mark.asyncio
    async def test_web_search_limit_recovers_with_synthesis_prompt(
        self, mock_session_service
    ):
        calls = []

        async def failing_run_async(**kwargs):
            calls.append(kwargs)
            raise WebSearchLimitExceeded("limit hit")
            yield  # pragma: no cover

        async def recovery_run_async(**kwargs):
            calls.append(kwargs)
            yield FakeEvent(
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=7,
                    candidates_token_count=3,
                )
            )

        run_async_calls = [failing_run_async, recovery_run_async]

        def run_async(**kwargs):
            return run_async_calls.pop(0)(**kwargs)

        runner_instance = MagicMock()
        runner_instance.run_async = run_async

        @asynccontextmanager
        async def fake_runner_cm(*_args, **_kwargs):
            yield runner_instance

        class _FakeApp:
            def __init__(self, *, name: str = "fedotmas", root_agent, plugins=None):
                self.name = name
                self.root_agent = root_agent
                self.plugins = plugins or []

        with (
            patch("fedotmas.core.runner.App", _FakeApp),
            patch("fedotmas.core.runner.Runner", side_effect=fake_runner_cm),
        ):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
            )

        assert isinstance(result, PipelineResult)
        assert result.total_prompt_tokens == 7
        assert result.total_completion_tokens == 3
        assert len(calls) == 2
        recovery_text = calls[1]["new_message"].parts[0].text
        assert recovery_text == SEARCH_LIMIT_RECOVERY_PROMPT


class TestSearchLimitDetection:
    """`_is_search_limit_exceeded` must see through exception wrappers."""

    def test_plain(self):
        assert _is_search_limit_exceeded(WebSearchLimitExceeded("x")) is True

    def test_wrapped_in_retry_error(self):
        # The production failure mode: a worker-model retry wraps the plugin
        # exception, so the outermost type is RetryError, not the limit error.
        wrapped = _retry_error_wrapping(WebSearchLimitExceeded("x"))
        assert _is_search_limit_exceeded(wrapped) is True

    def test_wrapped_in_cause_chain(self):
        try:
            try:
                raise WebSearchLimitExceeded("x")
            except WebSearchLimitExceeded as inner:
                raise RuntimeError("wrapper") from inner
        except RuntimeError as outer:
            assert _is_search_limit_exceeded(outer) is True

    def test_unrelated_error(self):
        assert _is_search_limit_exceeded(ValueError("nope")) is False
        assert _is_search_limit_exceeded(_retry_error_wrapping(TimeoutError())) is False


class TestFinalizeMode:
    """Recovery disables web tools so the finalization turn cannot re-trip."""

    def test_enter_finalize_mode_sets_flag(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=3)
        assert plugin.finalizing is False
        _enter_finalize_mode([plugin, object()])  # non-plugin ignored
        assert plugin.finalizing is True

    @pytest.mark.asyncio
    async def test_recovery_fires_for_wrapped_exception_and_finalizes(
        self, mock_session_service
    ):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=3, hard_fail=True)
        calls = []

        async def failing_run_async(**kwargs):
            calls.append(kwargs)
            raise _retry_error_wrapping(WebSearchLimitExceeded("limit hit"))
            yield  # pragma: no cover

        async def recovery_run_async(**kwargs):
            calls.append(kwargs)
            yield FakeEvent(
                usage_metadata=FakeUsageMetadata(
                    prompt_token_count=7, candidates_token_count=3
                )
            )

        run_async_calls = [failing_run_async, recovery_run_async]
        runner_instance = MagicMock()
        runner_instance.run_async = lambda **kw: run_async_calls.pop(0)(**kw)

        @asynccontextmanager
        async def fake_runner_cm(*_args, **_kwargs):
            yield runner_instance

        class _FakeApp:
            def __init__(self, *, name: str = "fedotmas", root_agent, plugins=None):
                self.name = name
                self.root_agent = root_agent
                self.plugins = plugins or []

        with (
            patch("fedotmas.core.runner.App", _FakeApp),
            patch("fedotmas.core.runner.Runner", side_effect=fake_runner_cm),
        ):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
                plugins=[plugin],
            )

        assert isinstance(result, PipelineResult)
        assert len(calls) == 2  # recovery turn ran despite the RetryError wrapper
        assert calls[1]["new_message"].parts[0].text == SEARCH_LIMIT_RECOVERY_PROMPT
        assert plugin.finalizing is True


class TestExecutionTimeoutSalvage:
    """A pipeline that exceeds its timeout returns partial state, not an error."""

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_state(self, mock_session_service):
        partial = FakeSession(state={"sub_answer": "42", "user_query": "q"})
        mock_session_service.get_session = AsyncMock(return_value=partial)

        async def hanging_run_async(**_kwargs):
            await asyncio.sleep(5)
            yield  # pragma: no cover

        runner_instance = MagicMock()
        runner_instance.run_async = hanging_run_async

        @asynccontextmanager
        async def fake_runner_cm(*_args, **_kwargs):
            yield runner_instance

        class _FakeApp:
            def __init__(self, *, name: str = "fedotmas", root_agent, plugins=None):
                self.name = name
                self.root_agent = root_agent
                self.plugins = plugins or []

        with (
            patch("fedotmas.core.runner.App", _FakeApp),
            patch("fedotmas.core.runner.Runner", side_effect=fake_runner_cm),
        ):
            result = await run_pipeline(
                _fake_agent(),
                "hello",
                session_service=mock_session_service,
                timeout=0.05,
            )

        assert isinstance(result, PipelineResult)
        assert result.state["sub_answer"] == "42"
