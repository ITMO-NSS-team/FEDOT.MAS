"""Legacy compatibility shim — delegates to ``backends.adk.serving``."""
from __future__ import annotations

from fedotmas.backends.adk.serving import serve_adk as serve  # noqa: F401

__all__ = ["serve"]
