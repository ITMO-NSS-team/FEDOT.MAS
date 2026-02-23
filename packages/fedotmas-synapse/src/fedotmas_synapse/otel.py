from __future__ import annotations


def configure_otel(
    endpoint: str = "http://localhost:4318/v1/traces",
    service_name: str = "fedotmas",
) -> None:
    """Initialize OTEL TracerProvider with OTLP HTTP exporter."""
    raise NotImplementedError("configure_otel")


class OtelEventCallback:
    """Event callback that annotates OTEL spans from ADK events."""

    def __call__(self, event) -> None:  # noqa: ARG002
        raise NotImplementedError("OtelEventCallback")
