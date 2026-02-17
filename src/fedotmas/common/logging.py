from __future__ import annotations

import os
import sys

from loguru import Logger, logger

_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "{extra[name]}:{function}:{line} | {message}"
)


def setup_logging(level: str | None = None) -> None:
    """Configure loguru with pipe-delimited format.

    Reads ``FEDOTMAS_LOG_LEVEL`` from the environment if *level* is not given.
    """
    resolved = level or os.getenv("FEDOTMAS_LOG_LEVEL", "INFO")
    logger.remove()
    logger.add(sys.stderr, format=_FORMAT, level=resolved.upper())


def get_logger(name: str) -> Logger:
    """Return a logger bound to *name*."""
    return logger.bind(name=name)


# Auto-setup on import so every ``get_logger`` call works immediately.
setup_logging()
