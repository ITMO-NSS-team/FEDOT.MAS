from __future__ import annotations

from unittest.mock import MagicMock

from fedotmas_synapse.otel import OtelEventCallback, configure_otel


def test_configure_otel_initializes() -> None:
    configure_otel()


def test_otel_callback_processes_event(otel_callback: OtelEventCallback) -> None:
    mock_event = MagicMock()
    otel_callback(mock_event)


def test_otel_callback_is_callable(otel_callback: OtelEventCallback) -> None:
    assert callable(otel_callback)
