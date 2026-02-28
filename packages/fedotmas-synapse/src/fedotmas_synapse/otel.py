"""OpenTelemetry integration for FEDOT.MAS ADK events.

See INCOMPATIBILITIES.md §3 for the sync-context-manager vs async-hooks
tension and the chosen resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.events import Event

    from telemetry.tracer import SynapseTracer


def configure_otel(
    endpoint: str = "http://localhost:4318/v1/traces",
    service_name: str = "fedotmas",
) -> None:
    """Initialize OTEL TracerProvider with OTLP HTTP exporter.

    Intended for **standalone** FEDOT.MAS deployments (outside CodeSynapse).
    When running inside CodeSynapse, pass a ``SynapseTracer`` to
    :class:`OtelEventCallback` instead.

    Args:
        endpoint: OTLP HTTP exporter endpoint.
        service_name: OTEL service name.
    """
    raise NotImplementedError("configure_otel")


class OtelEventCallback:
    """Event callback that creates point-in-time OTEL spans from ADK events.

    When *tracer* is a ``SynapseTracer``, uses ``tracer.tracer.start_span()``
    directly (not the context-manager ``start_span()``) to create discrete
    spans for each event.  OTEL SDK calls are non-blocking.

    Args:
        tracer: CodeSynapse ``SynapseTracer`` instance.  When ``None``,
            the callback is a no-op until a tracer is provided.
    """

    def __init__(self, *, tracer: SynapseTracer | None = None) -> None:
        self._tracer = tracer

    def __call__(self, event: Event) -> None:
        """Create a point-in-time span for *event*.

        Args:
            event: ADK event to record as an OTEL span.
        """
        raise NotImplementedError("OtelEventCallback")
