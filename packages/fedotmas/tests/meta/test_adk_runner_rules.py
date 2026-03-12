"""ADK runner rules — retry, session errors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from fedotmas.meta._adk_runner import run_meta_agent_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    name: str
    model: str | None = None


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
        ):
            with pytest.raises(RuntimeError, match="permanent failure"):
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
