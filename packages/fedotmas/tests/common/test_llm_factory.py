"""Factory tests — make_llm returns LiteLlm with correct transport."""

from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm

from fedotmas.common.llm import _ProxyClient, make_llm
from fedotmas.config.settings import ModelConfig, resolve_model_config


class TestMakeLlm:
    def test_creates_litellm_with_model(self):
        cfg = ModelConfig(model="openrouter/meta-llama/llama-3-70b")
        llm = make_llm(cfg)
        assert isinstance(llm, LiteLlm)
        assert llm.model == "openrouter/meta-llama/llama-3-70b"

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

    def test_direct_mode_no_proxy_client(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        llm = make_llm(cfg)
        assert not isinstance(llm.llm_client, _ProxyClient)


class TestProxyMode:
    def test_sets_proxy_client(self):
        cfg = ModelConfig(
            model="openrouter/openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert isinstance(llm.llm_client, _ProxyClient)

    def test_empty_additional_args(self):
        cfg = ModelConfig(
            model="openrouter/openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert not llm._additional_args

    def test_preserves_model(self):
        cfg = ModelConfig(
            model="openrouter/openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert llm.model == "openrouter/openai/gpt-4o"

    def test_default_api_key(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert isinstance(llm.llm_client, _ProxyClient)
        assert llm.llm_client._client.api_key == "no-key"

    def test_custom_api_key(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
            api_key="sk-proxy-key",
        )
        llm = make_llm(cfg)
        assert llm.llm_client._client.api_key == "sk-proxy-key"

    def test_base_url_set(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert str(llm.llm_client._client.base_url) == "http://localhost:9090/litellm/"


class TestResolveModelConfig:
    def test_picks_up_env_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9090/litellm")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = resolve_model_config("openrouter/openai/gpt-4o")
        assert cfg.api_base == "http://localhost:9090/litellm"
        assert cfg.api_key is None

    def test_picks_up_env_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        cfg = resolve_model_config("openai/gpt-4o")
        assert cfg.api_key == "sk-test-key"
        assert cfg.api_base is None

    def test_no_env_defaults_to_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = resolve_model_config("openai/gpt-4o")
        assert cfg.api_base is None
        assert cfg.api_key is None

    def test_explicit_modelconfig_unchanged(self):
        original = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://custom:8080",
            api_key="sk-custom",
        )
        result = resolve_model_config(original)
        assert result is original
