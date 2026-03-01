"""Factory dispatch tests — make_llm routes to the correct provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas.config.settings import ModelConfig
from fedotmas.llm import make_llm
from fedotmas.llm.bifrost import BifrostLlm
from fedotmas.llm.openrouter import OpenRouterLlm


class TestDefaultProvider:
    """provider=None → LiteLlm."""

    @patch("fedotmas.llm.LiteLlm")
    def test_default_litellm(self, mock_lite):
        cfg = ModelConfig(model="openai/gpt-4o")
        make_llm(cfg)
        mock_lite.assert_called_once_with(model="openai/gpt-4o")

    @patch("fedotmas.llm.LiteLlm")
    def test_default_with_api_base(self, mock_lite):
        cfg = ModelConfig(model="m", api_base="http://x", api_key="sk-1")
        make_llm(cfg)
        mock_lite.assert_called_once_with(
            model="m", api_base="http://x", api_key="sk-1"
        )


class TestOpenRouterProvider:
    """provider="openrouter" → OpenRouterLlm."""

    def test_creates_openrouter(self):
        cfg = ModelConfig(
            model="openrouter/meta-llama/llama-3-70b",
            provider="openrouter",
            api_key="or-key",
        )
        llm = make_llm(cfg)
        assert isinstance(llm, OpenRouterLlm)
        assert llm.api_key == "or-key"

    def test_strips_prefix(self):
        llm = OpenRouterLlm(model="openrouter/foo/bar")
        assert llm._resolve_model("openrouter/foo/bar") == "foo/bar"

    def test_no_prefix_passthrough(self):
        llm = OpenRouterLlm(model="foo/bar")
        assert llm._resolve_model("foo/bar") == "foo/bar"


class TestBifrostProvider:
    """provider="bifrost" → BifrostLlm."""

    def test_creates_bifrost(self):
        cfg = ModelConfig(
            model="bifrost/my-model",
            provider="bifrost",
        )
        llm = make_llm(cfg)
        assert isinstance(llm, BifrostLlm)

    def test_custom_api_base(self):
        cfg = ModelConfig(
            model="bifrost/m",
            provider="bifrost",
            api_base="http://custom:9090/v1",
        )
        llm = make_llm(cfg)
        assert isinstance(llm, BifrostLlm)
        assert llm.api_base == "http://custom:9090/v1"

    def test_strips_prefix(self):
        llm = BifrostLlm(model="bifrost/my-model")
        assert llm._resolve_model("bifrost/my-model") == "my-model"

    def test_no_prefix_passthrough(self):
        llm = BifrostLlm(model="my-model")
        assert llm._resolve_model("my-model") == "my-model"
