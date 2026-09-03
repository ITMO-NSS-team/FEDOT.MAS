"""Rules for resolving models from the environment."""

import pytest

from fedotmas._settings import DEFAULT_WORKER_MODELS, get_worker_models


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("FEDOTMAS_WORKER_MODELS", raising=False)
    monkeypatch.delenv("FEDOTMAS_DEFAULT_MODEL", raising=False)
    return monkeypatch


class TestWorkerModelsNeverEmpty:
    """A list that parses to nothing falls through instead of yielding []."""

    def test_unset(self, clean_env):
        assert get_worker_models() == DEFAULT_WORKER_MODELS

    def test_separators_only(self, clean_env):
        clean_env.setenv("FEDOTMAS_WORKER_MODELS", ",")
        assert get_worker_models() == DEFAULT_WORKER_MODELS

    def test_whitespace_only(self, clean_env):
        clean_env.setenv("FEDOTMAS_WORKER_MODELS", " , ")
        assert get_worker_models() == DEFAULT_WORKER_MODELS

    def test_separators_only_falls_through_to_default_model(self, clean_env):
        clean_env.setenv("FEDOTMAS_WORKER_MODELS", ",")
        clean_env.setenv("FEDOTMAS_DEFAULT_MODEL", "openai/gpt-4o")
        assert get_worker_models() == ["openai/gpt-4o"]


class TestWorkerModelsParsing:
    """Named models win over both defaults."""

    def test_single(self, clean_env):
        clean_env.setenv("FEDOTMAS_WORKER_MODELS", "openai/gpt-4o")
        assert get_worker_models() == ["openai/gpt-4o"]

    def test_multiple_with_padding(self, clean_env):
        clean_env.setenv("FEDOTMAS_WORKER_MODELS", " openai/gpt-4o , gemini/flash ")
        assert get_worker_models() == ["openai/gpt-4o", "gemini/flash"]
