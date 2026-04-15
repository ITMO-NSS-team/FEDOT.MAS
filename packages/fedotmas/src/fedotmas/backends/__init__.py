from __future__ import annotations

from typing import Any, Callable

from fedotmas.interfaces.backend import BackendProtocol

_BACKENDS: dict[str, Callable[..., BackendProtocol]] = {}


def register_backend(name: str, factory: Callable[..., BackendProtocol]) -> None:
    """Register a backend factory under *name*."""
    _BACKENDS[name] = factory


def get_backend(name: str, **kwargs: Any) -> BackendProtocol:
    """Look up and instantiate a registered backend by *name*."""
    if name not in _BACKENDS:
        available = sorted(_BACKENDS) or ["(none registered)"]
        raise ValueError(
            f"Unknown backend '{name}'. Available: {available}. "
            f"Install the appropriate extra (e.g. pip install fedotmas[adk])."
        )
    return _BACKENDS[name](**kwargs)


def _auto_register_adk() -> None:
    """Auto-register the ADK backend if google-adk is installed."""
    try:
        from fedotmas.backends.adk import ADKBackend

        register_backend("adk", ADKBackend)
    except ImportError:
        pass


_auto_register_adk()
