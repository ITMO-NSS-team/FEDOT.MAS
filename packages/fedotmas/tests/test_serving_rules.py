from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App

from fedotmas.backends.adk.serving import _AgentLoader, serve_adk
from fedotmas.interfaces.agent import AgentDescriptor


# ---------------------------------------------------------------------------
# _AgentLoader
# ---------------------------------------------------------------------------


class TestLoaderEmpty:
    def test_list_agents_empty(self):
        loader = _AgentLoader()
        assert loader.list_agents() == []

    def test_load_from_empty_raises_with_none_hint(self):
        loader = _AgentLoader()
        with pytest.raises(KeyError, match=r"\(none\)"):
            loader.load_agent("anything")


class TestLoaderRegister:
    def test_load_returns_same_object(self):
        loader = _AgentLoader()
        agent = MagicMock(spec=BaseAgent)
        loader.register("a", agent)
        assert loader.load_agent("a") is agent

    def test_list_agents_sorted(self):
        loader = _AgentLoader()
        for name in ("charlie", "alice", "bob"):
            loader.register(name, MagicMock(spec=BaseAgent))
        assert loader.list_agents() == ["alice", "bob", "charlie"]

    def test_load_missing_lists_available(self):
        loader = _AgentLoader()
        loader.register("alpha", MagicMock(spec=BaseAgent))
        loader.register("beta", MagicMock(spec=BaseAgent))
        with pytest.raises(KeyError, match="alpha, beta"):
            loader.load_agent("gamma")

    def test_register_overwrites(self):
        loader = _AgentLoader()
        first = MagicMock(spec=BaseAgent)
        second = MagicMock(spec=BaseAgent)
        loader.register("x", first)
        loader.register("x", second)
        assert loader.load_agent("x") is second

    def test_app_object_accepted(self):
        loader = _AgentLoader()
        app = MagicMock(spec=App)
        loader.register("my_app", app)
        assert loader.load_agent("my_app") is app


# ---------------------------------------------------------------------------
# serve_adk
# ---------------------------------------------------------------------------

_PATCH_GET = "fedotmas.backends.adk.serving.get_fast_api_app"
_PATCH_BUILD = "fedotmas.backends.adk.serving.build_adk_tree"


class TestCreateApiApp:
    @patch(_PATCH_BUILD, return_value=MagicMock(spec=BaseAgent))
    @patch(_PATCH_GET)
    def test_default_session_uri(self, mock_get, _mock_build):
        serve_adk({})
        _, kwargs = mock_get.call_args
        assert kwargs["session_service_uri"] == "memory://"

    @patch(_PATCH_BUILD, return_value=MagicMock(spec=BaseAgent))
    @patch(_PATCH_GET)
    def test_custom_session_uri(self, mock_get, _mock_build):
        serve_adk({}, session_service_uri="sqlite:///s.db")
        _, kwargs = mock_get.call_args
        assert kwargs["session_service_uri"] == "sqlite:///s.db"

    @patch(_PATCH_BUILD, return_value=MagicMock(spec=BaseAgent))
    @patch(_PATCH_GET)
    def test_empty_agents_dict(self, mock_get, _mock_build):
        serve_adk({})
        _, kwargs = mock_get.call_args
        loader = kwargs["agent_loader"]
        assert isinstance(loader, _AgentLoader)
        assert loader.list_agents() == []

    @patch(_PATCH_BUILD, return_value=MagicMock(spec=BaseAgent))
    @patch(_PATCH_GET)
    def test_all_kwargs_forwarded(self, mock_get, _mock_build):
        serve_adk(
            {},
            session_service_uri="pg://db",
            memory_service_uri="mem://m",
            artifact_service_uri="art://a",
            web=True,
            host="0.0.0.0",
            port=9000,
        )
        _, kwargs = mock_get.call_args
        assert kwargs["agents_dir"] == "."
        assert kwargs["session_service_uri"] == "pg://db"
        assert kwargs["memory_service_uri"] == "mem://m"
        assert kwargs["artifact_service_uri"] == "art://a"
        assert kwargs["web"] is True
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000

    @patch(_PATCH_BUILD, return_value=MagicMock(spec=BaseAgent))
    @patch(_PATCH_GET)
    def test_agents_registered_in_loader(self, mock_get, _mock_build):
        tree_a = AgentDescriptor(name="a", instruction="do a")
        tree_b = AgentDescriptor(name="b", instruction="do b")
        serve_adk({"a": tree_a, "b": tree_b})
        _, kwargs = mock_get.call_args
        loader: _AgentLoader = kwargs["agent_loader"]
        assert isinstance(loader, _AgentLoader)
        # The loader should have both agents registered (as App wrappers)
        assert sorted(loader.list_agents()) == ["a", "b"]
