"""Offline checks; no credentials or requests to an LLM are needed."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("protollm_review_example", Path(__file__).with_name("review_report.py"))
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_missing_key_and_empty_report(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="empty"):
        example.review_report(" ", "model")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        example.review_report("report", "model")


@pytest.mark.parametrize("old_key", [None, "other-service-key"])
@pytest.mark.parametrize("factory_fails", [False, True])
def test_connector_contract_and_environment_restoration(monkeypatch, old_key, factory_fails):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    if old_key is None:
        monkeypatch.delenv("LLM_SERVICE_KEY", raising=False)
    else:
        monkeypatch.setenv("LLM_SERVICE_KEY", old_key)
    captured = []

    def factory(url, **kwargs):
        assert url == "https://openrouter.ai/api/v1;qwen/qwen3-32b"
        assert os.environ["LLM_SERVICE_KEY"] == "test-key"
        assert kwargs["max_retries"] == 0
        if factory_fails:
            raise RuntimeError("constructor failed")

        def invoke(messages):
            captured.extend(messages)
            assert os.environ.get("LLM_SERVICE_KEY") == old_key
            return SimpleNamespace(content="Needs validation")

        return SimpleNamespace(invoke=invoke)

    connectors = ModuleType("protollm.connectors")
    connectors.create_llm_connector = factory
    messages = ModuleType("langchain_core.messages")
    messages.SystemMessage = messages.HumanMessage = SimpleNamespace
    monkeypatch.setitem(sys.modules, "protollm.connectors", connectors)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)
    if factory_fails:
        with pytest.raises(RuntimeError, match="constructor failed"):
            example.review_report("Report unchanged", "qwen/qwen3-32b")
    else:
        assert example.review_report("Report unchanged", "qwen/qwen3-32b") == "Needs validation"
        assert len(captured) == 2
        assert captured[1].content == "Report unchanged"
    assert os.environ.get("LLM_SERVICE_KEY") == old_key
