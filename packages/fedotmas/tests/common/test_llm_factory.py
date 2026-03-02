"""Factory tests — make_llm always returns LiteLlm with correct kwargs."""

from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm

from fedotmas.config.settings import ModelConfig
from fedotmas.common.llm import make_llm


class TestMakeLlm:
    def test_creates_litellm_with_model(self):
        cfg = ModelConfig(model="openrouter/meta-llama/llama-3-70b")
        llm = make_llm(cfg)
        assert isinstance(llm, LiteLlm)
        assert llm.model == "openrouter/meta-llama/llama-3-70b"

    def test_passes_api_base(self):
        cfg = ModelConfig(model="gpt-4o", api_base="http://localhost:9090/litellm")
        llm = make_llm(cfg)
        assert llm._additional_args.get("api_base") == "http://localhost:9090/litellm"

    def test_passes_api_key(self):
        cfg = ModelConfig(model="gpt-4o", api_key="sk-test")
        llm = make_llm(cfg)
        assert llm._additional_args.get("api_key") == "sk-test"

    def test_no_extra_kwargs_when_defaults(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        llm = make_llm(cfg)
        assert "api_base" not in llm._additional_args
        assert "api_key" not in llm._additional_args

    def test_openai_model(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        llm = make_llm(cfg)
        assert isinstance(llm, LiteLlm)
        assert llm.model == "openai/gpt-4o"

    def test_bifrost_via_api_base(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
            api_key="sk-key",
        )
        llm = make_llm(cfg)
        assert isinstance(llm, LiteLlm)
        assert llm._additional_args["api_base"] == "http://localhost:9090/litellm"
