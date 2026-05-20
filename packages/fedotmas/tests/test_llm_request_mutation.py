"""Regression guard for in-place mutation of ``LlmRequest.model``.

Verifies that mutating ``llm_request.model`` inside ``before_model_callback``
propagates through ``LiteLlm`` to the underlying ``acompletion`` call. The
model routing module relies on this contract: if ADK ever introduces a
defensive copy between the callback chain and the transport, or freezes
``LlmRequest``, routing decisions would silently no-op and these tests
would catch it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from fedotmas._settings import ModelConfig
from fedotmas.common.llm import make_llm


class _CapturedCall(Exception):
    """Raised inside the mocked ``acompletion`` to short-circuit
    further processing once the args have been captured."""


def _make_request(model: str) -> LlmRequest:
    return LlmRequest(
        model=model,
        contents=[
            types.Content(role="user", parts=[types.Part(text="hello")])
        ],
    )


def _build_llm_with_capture() -> tuple[object, dict[str, object]]:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured["model"] = kwargs.get("model")
        raise _CapturedCall

    llm = make_llm(
        ModelConfig(
            model="openai/original-model",
            api_base="http://unused.invalid",
            api_key="no-key",
        )
    )
    llm.llm_client.acompletion = AsyncMock(side_effect=fake_acompletion)
    return llm, captured


@pytest.mark.asyncio
async def test_control_no_mutation_uses_original_model() -> None:
    llm, captured = _build_llm_with_capture()
    req = _make_request("openai/original-model")

    with pytest.raises(_CapturedCall):
        async for _ in llm.generate_content_async(req):
            pass

    assert captured["model"] == "openai/original-model"


@pytest.mark.asyncio
async def test_mutated_request_model_reaches_acompletion() -> None:
    llm, captured = _build_llm_with_capture()
    req = _make_request("openai/original-model")

    req.model = "openai/routed-target-model"

    with pytest.raises(_CapturedCall):
        async for _ in llm.generate_content_async(req):
            pass

    assert captured["model"] == "openai/routed-target-model"
