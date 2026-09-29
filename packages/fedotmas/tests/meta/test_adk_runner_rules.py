"""ADK runner rules — retry, session errors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fedotmas.meta._adk_runner import (
    _MetaUsageTracker,
    _resolve_max_output_tokens,
    _retry_message,
    run_meta_agent_call,
)
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    name: str
    model: str | None = None


class TestMaxOutputTokens:
    """Meta-agent output token limit comes from env with a finite default."""

    def test_default(self, monkeypatch):
        monkeypatch.delenv("FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS", raising=False)

        assert _resolve_max_output_tokens() == 8192

    def test_disabled_with_zero(self, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS", "0")

        assert _resolve_max_output_tokens() is None

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS", "bad")

        assert _resolve_max_output_tokens() == 8192


# ---------------------------------------------------------------------------
# Retry rules
# ---------------------------------------------------------------------------


class TestRetryOnTransientError:
    """Rule 5: retry succeeds after transient error."""

    async def test_retry_succeeds(self, model_config):
        call_count = 0

        async def _fake_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            from fedotmas.meta._adk_runner import LLMCallResult

            return LLMCallResult(
                raw_output={"result": "ok"},
                prompt_tokens=10,
                completion_tokens=20,
                elapsed=1.0,
            )

        with (
            patch(
                "fedotmas.meta._adk_runner._execute_meta_call",
                side_effect=_fake_execute,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_meta_agent_call(
                agent_name="test",
                instruction="test",
                user_message="test",
                output_schema=_DummySchema,
                output_key="result",
                model=model_config,
                temperature=0.3,
                max_retries=2,
            )
            assert result.raw_output == {"result": "ok"}
            assert call_count == 2

    async def test_invalid_structured_output_feedback_and_usage_survive_retry(
        self, model_config
    ):
        calls: list[tuple[str, dict[str, int | float]]] = []

        async def _fake_execute(**kwargs):
            usage = kwargs["usage_totals"]
            message = kwargs["user_message"]
            calls.append((message, usage))
            tracker = _MetaUsageTracker(usage)
            if len(calls) == 1:
                await tracker.after_model_callback(
                    callback_context=None,
                    llm_response=SimpleNamespace(
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=11, candidates_token_count=5
                        )
                    ),
                )
                raise RuntimeError(
                    "ValidationError: Pipeline references unknown agent 'bad'. "
                    "Available: ['solver']"
                )
            await tracker.after_model_callback(
                callback_context=None,
                llm_response=SimpleNamespace(
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=7, candidates_token_count=3
                    )
                ),
            )
            from fedotmas.meta._adk_runner import LLMCallResult

            return LLMCallResult(
                raw_output={"name": "ok"},
                prompt_tokens=7,
                completion_tokens=3,
                elapsed=1.0,
            )

        with (
            patch(
                "fedotmas.meta._adk_runner._execute_meta_call",
                side_effect=_fake_execute,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_meta_agent_call(
                agent_name="test",
                instruction="test",
                user_message="generate config",
                output_schema=_DummySchema,
                output_key="result",
                model=model_config,
                temperature=0.3,
                max_retries=1,
            )

        assert "unknown agent 'bad'" in calls[1][0]
        assert "Available: ['solver']" in calls[1][0]
        assert "agent_name values exactly as declared" in calls[1][0]
        assert result.prompt_tokens == 18
        assert result.completion_tokens == 8

    def test_truncated_meta_output_is_identified_explicitly(self):
        message = _retry_message(
            "generate config", RuntimeError("finish_reason=length; MAX_TOKENS")
        )
        assert "truncated by the output-token limit" in message
        assert "FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS" in message


class TestRetriesExhausted:
    """Rule 6: raises after all retries exhausted."""

    async def test_raises_after_exhaustion(self, model_config):
        async def _always_fail(**kwargs):
            raise RuntimeError("permanent failure")

        with (
            patch(
                "fedotmas.meta._adk_runner._execute_meta_call", side_effect=_always_fail
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="permanent failure"),
        ):
            await run_meta_agent_call(
                agent_name="test",
                instruction="test",
                user_message="test",
                output_schema=_DummySchema,
                output_key="result",
                model=model_config,
                temperature=0.3,
                max_retries=1,
            )


class TestTimeoutFailFast:
    """Timeouts should not retry the same meta prompt."""

    async def test_timeout_does_not_retry(self, model_config):
        call_count = 0

        async def _timeout(**kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError()

        with (
            patch("fedotmas.meta._adk_runner._execute_meta_call", side_effect=_timeout),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(TimeoutError),
        ):
            await run_meta_agent_call(
                agent_name="test",
                instruction="test",
                user_message="test",
                output_schema=_DummySchema,
                output_key="result",
                model=model_config,
                temperature=0.3,
                max_retries=3,
            )

        assert call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Session error rules
# ---------------------------------------------------------------------------


class TestSessionLost:
    """Rule 7: get_session returning None raises RuntimeError."""

    async def test_session_lost(self, mock_session_service, model_config):
        mock_session_service.get_session = AsyncMock(return_value=None)

        fake_event = MagicMock()
        fake_event.partial = False
        fake_event.usage_metadata = None
        fake_event.content = None
        fake_event.error_code = None

        async def _fake_run_async(**kwargs):
            yield fake_event

        with (
            patch("fedotmas.meta._adk_runner.LlmAgent"),
            patch("fedotmas.meta._adk_runner.make_llm"),
            patch("fedotmas.meta._adk_runner.Runner") as mock_runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run_async = _fake_run_async
            mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
            mock_runner.__aexit__ = AsyncMock(return_value=False)
            mock_runner_cls.return_value = mock_runner

            with pytest.raises(RuntimeError, match="session lost"):
                await run_meta_agent_call(
                    agent_name="test",
                    instruction="test",
                    user_message="test",
                    output_schema=_DummySchema,
                    output_key="result",
                    model=model_config,
                    temperature=0.3,
                    session_service=mock_session_service,
                    max_retries=0,
                )


class TestOutputKeyMissing:
    """Rule 8: missing output_key in session state raises RuntimeError."""

    async def test_output_key_missing(self, mock_session_service, model_config):
        # Session exists but state is empty → key missing

        fake_event = MagicMock()
        fake_event.partial = False
        fake_event.usage_metadata = None
        fake_event.content = None
        fake_event.error_code = None

        async def _fake_run_async(**kwargs):
            yield fake_event

        with (
            patch("fedotmas.meta._adk_runner.LlmAgent"),
            patch("fedotmas.meta._adk_runner.make_llm"),
            patch("fedotmas.meta._adk_runner.Runner") as mock_runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run_async = _fake_run_async
            mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
            mock_runner.__aexit__ = AsyncMock(return_value=False)
            mock_runner_cls.return_value = mock_runner

            with pytest.raises(RuntimeError, match="did not produce"):
                await run_meta_agent_call(
                    agent_name="test",
                    instruction="test",
                    user_message="test",
                    output_schema=_DummySchema,
                    output_key="missing_key",
                    model=model_config,
                    temperature=0.3,
                    session_service=mock_session_service,
                    max_retries=0,
                )


class TestRunnerCleanupFailure:
    async def test_cleanup_error_does_not_mask_primary_llm_validation_error(
        self, mock_session_service, model_config
    ):
        fake_event = MagicMock()
        fake_event.partial = False
        fake_event.usage_metadata = None
        fake_event.content = None
        fake_event.error_code = "ValidationError"
        fake_event.error_message = "invalid MAWConfig"

        async def _fake_run_async(**kwargs):
            yield fake_event

        with (
            patch("fedotmas.meta._adk_runner.LlmAgent"),
            patch("fedotmas.meta._adk_runner.make_llm"),
            patch("fedotmas.meta._adk_runner.Runner") as mock_runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run_async = _fake_run_async
            mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
            mock_runner.__aexit__ = AsyncMock(
                side_effect=ValueError("OpenTelemetry context cleanup failed")
            )
            mock_runner_cls.return_value = mock_runner

            with pytest.raises(
                RuntimeError, match="ValidationError.*invalid MAWConfig"
            ):
                await run_meta_agent_call(
                    agent_name="test",
                    instruction="test",
                    user_message="test",
                    output_schema=_DummySchema,
                    output_key="result",
                    model=model_config,
                    temperature=0.3,
                    session_service=mock_session_service,
                    max_retries=0,
                )
