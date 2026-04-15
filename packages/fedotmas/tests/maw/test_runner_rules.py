"""Runner edge-case tests — mock ADK Runner, test event processing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fedotmas.backends.adk.runner import ADKRunner
from fedotmas.interfaces.runner import PipelineResult

from .conftest import FakeActions, FakeEvent, FakeSession, FakeUsageMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_agent(name: str = "root") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _patch_runner(events: list[FakeEvent]):
    """Return patches for ADK Runner as async context manager yielding events.

    Also patches ``App`` and ``build_adk_tree`` so that MagicMock agents
    pass Pydantic validation.
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

    runner_patch = patch("fedotmas.backends.adk.runner.Runner", side_effect=fake_runner_cm)
    app_patch = patch("fedotmas.backends.adk.runner.App", _FakeApp)
    build_patch = patch(
        "fedotmas.backends.adk.runner.build_adk_tree",
        return_value=_fake_agent(),
    )

    from contextlib import ExitStack

    @asynccontextmanager
    async def combined():
        with ExitStack() as stack:
            stack.enter_context(app_patch)
            stack.enter_context(build_patch)
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
            runner = ADKRunner(session_service=mock_session_service)
            result = await runner.run_pipeline(
                _fake_agent(),
                "hello",
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
            runner = ADKRunner(session_service=mock_session_service)
            result = await runner.run_pipeline(
                _fake_agent(),
                "hello",
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
            runner = ADKRunner(session_service=mock_session_service)
            with pytest.raises(RuntimeError, match="RATE_LIMIT"):
                await runner.run_pipeline(
                    _fake_agent(),
                    "hello",
                )


class TestSessionLostAfterRun:
    """Rule 4: get_session returns None → RuntimeError."""

    @pytest.mark.asyncio
    async def test_session_lost(self, mock_session_service):
        mock_session_service.get_session = AsyncMock(return_value=None)
        events: list[FakeEvent] = []
        async with _patch_runner(events):
            runner = ADKRunner(session_service=mock_session_service)
            with pytest.raises(RuntimeError, match="lost"):
                await runner.run_pipeline(
                    _fake_agent(),
                    "hello",
                )


class TestInitialStateMerged:
    """Rule 5: initial_state is passed into create_session."""

    @pytest.mark.asyncio
    async def test_initial_state(self, mock_session_service):
        events: list[FakeEvent] = []
        async with _patch_runner(events):
            runner = ADKRunner(session_service=mock_session_service)
            await runner.run_pipeline(
                _fake_agent(),
                "hello",
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
            runner = ADKRunner(session_service=mock_session_service)
            result = await runner.run_pipeline(
                _fake_agent(),
                "hello",
            )
        assert result.state["answer"] == "42"


class TestEmptyOutputWarning:
    """Rule 7: state_delta with None value → no crash."""

    @pytest.mark.asyncio
    async def test_none_value_no_crash(self, mock_session_service):
        events = [
            FakeEvent(actions=FakeActions(state_delta={"key": None})),
        ]
        async with _patch_runner(events):
            runner = ADKRunner(session_service=mock_session_service)
            result = await runner.run_pipeline(
                _fake_agent(),
                "hello",
            )
        assert isinstance(result, PipelineResult)


class TestPluginsPassed:
    """Rule 8: backend_plugins list is passed through to App."""

    @pytest.mark.asyncio
    async def test_plugins_forwarded(self, mock_session_service):
        from google.adk.plugins import BasePlugin

        class StubPlugin(BasePlugin):
            def __init__(self):
                super().__init__(name="stub")

        plugin = StubPlugin()
        events: list[FakeEvent] = []
        async with _patch_runner(events) as runner_patch:
            runner = ADKRunner(session_service=mock_session_service)
            await runner.run_pipeline(
                _fake_agent(),
                "hello",
                backend_plugins=[plugin],
            )
        # Runner receives the App wrapping the agent and plugins
        call_kwargs = runner_patch.call_args
        app = call_kwargs.kwargs.get("app")
        assert app is not None
        assert plugin in app.plugins
