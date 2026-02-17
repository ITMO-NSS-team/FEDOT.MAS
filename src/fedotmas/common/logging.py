from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

_FORMAT = (
    "<fg 216,222,233>{time:YYYY-MM-DD HH:mm:ss.SSS}</fg 216,222,233> | "
    "{level:<8} | "
    "<fg 136,192,208>{extra[name]}</fg 136,192,208>:"
    "<fg 129,161,193>{function}</fg 129,161,193>:"
    "<fg 235,203,139>{line}</fg 235,203,139> | "
    "{message}"
)


def setup_logging(level: str | None = None) -> None:
    """Configure loguru with pipe-delimited format.

    Reads ``FEDOTMAS_LOG_LEVEL`` from the environment if *level* is not given.
    """
    resolved = level or os.getenv("FEDOTMAS_LOG_LEVEL", "DEBUG")
    logger.remove()
    logger.add(sys.stderr, format=_FORMAT, level=resolved.upper())


def get_logger(name: str) -> Logger:
    """Return a logger bound to *name*."""
    return logger.bind(name=name)


setup_logging()
