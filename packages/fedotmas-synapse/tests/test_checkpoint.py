from __future__ import annotations

from unittest.mock import MagicMock

from fedotmas_synapse.checkpoint import CheckpointCallback


def test_before_agent_called(checkpoint: CheckpointCallback) -> None:
    mock_ctx = MagicMock()
    checkpoint.before_agent(callback_context=mock_ctx)


def test_after_agent_called(checkpoint: CheckpointCallback) -> None:
    mock_ctx = MagicMock()
    checkpoint.after_agent(callback_context=mock_ctx)


def test_stores_storage_uri(mongo_uri: str) -> None:
    cb = CheckpointCallback(storage_uri=mongo_uri)
    assert cb.storage_uri == mongo_uri
