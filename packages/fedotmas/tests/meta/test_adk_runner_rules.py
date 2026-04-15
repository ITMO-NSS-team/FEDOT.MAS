"""ADK runner rules — retry, session errors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from fedotmas.interfaces.runner import SingleAgentResult
from fedotmas.meta._adk_runner import run_meta_agent_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    name: str
    model: str | None = None


def _mock_backend(run_single_agent_fn):
    """Create a mock backend whose runner delegates to the given function."""
    runner = AsyncMock()
    runner.run_single_agent = run_single_agent_fn
    backend = MagicMock()
    backend.create_runner = MagicMock(return_value=runner)
    return patch("fedotmas.meta._adk_runner.get_backend", return_value=backend)


# ---------------------------------------------------------------------------
# Retry rules
# ---------------------------------------------------------------------------


class TestRetryOnTransientError:
    """Rule 5: retry succeeds after transient error."""

    @pytest.mark.asyncio
    async def test_retry_succeeds(self, model_config):
        call_count = 0

        async def _fake_run(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return SingleAgentResult(
                raw_output={"result": "ok"},
                prompt_tokens=10,
                completion_tokens=20,
                elapsed=1.0,
            )

        with _mock_backend(_fake_run), \
             patch("asyncio.sleep", new_callable=AsyncMock):
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

    @pytest.mark.asyncio
    async def test_raises_after_exhaustion(self, model_config):
        async def _always_fail(**kwargs):
            raise RuntimeError("permanent failure")

        with _mock_backend(_always_fail), \
             patch("asyncio.sleep", new_callable=AsyncMock):
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
    """Rule 7: run_single_agent returning None raw_output raises RuntimeError."""

    @pytest.mark.asyncio
    async def test_session_lost(self, model_config):
        async def _return_none(**kwargs):
            return SingleAgentResult(
                raw_output=None,
                prompt_tokens=10,
                completion_tokens=20,
                elapsed=1.0,
            )

        with _mock_backend(_return_none), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="did not produce"):
                await run_meta_agent_call(
                    agent_name="test",
                    instruction="test",
                    user_message="test",
                    output_schema=_DummySchema,
                    output_key="result",
                    model=model_config,
                    temperature=0.3,
                    max_retries=0,
                )


class TestOutputKeyMissing:
    """Rule 8: missing output_key (None raw_output) raises RuntimeError."""

    @pytest.mark.asyncio
    async def test_output_key_missing(self, model_config):
        async def _return_none(**kwargs):
            return SingleAgentResult(
                raw_output=None,
                prompt_tokens=10,
                completion_tokens=20,
                elapsed=1.0,
            )

        with _mock_backend(_return_none), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="did not produce"):
                await run_meta_agent_call(
                    agent_name="test",
                    instruction="test",
                    user_message="test",
                    output_schema=_DummySchema,
                    output_key="missing_key",
                    model=model_config,
                    temperature=0.3,
                    max_retries=0,
                )
