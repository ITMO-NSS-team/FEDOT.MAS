from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs
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

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level:<8} | "
    "{extra[name]}:{function}:{line} | "
    "{message}"
)


def _log_dir() -> Path:
    """Platform-aware log directory, overridable via ``FEDOTMAS_LOG_DIR``."""
    env = os.getenv("FEDOTMAS_LOG_DIR")
    if env:
        return Path(env)
    return Path(platformdirs.user_log_dir("fedotmas"))


def setup_logging(level: str | None = None) -> None:
    """Configure loguru with pipe-delimited format.

    Reads ``FEDOTMAS_LOG_LEVEL`` from the environment if *level* is not given.
    """
    resolved = level or os.getenv("FEDOTMAS_LOG_LEVEL", "DEBUG")
    logger.remove()
    logger.add(sys.stderr, format=_FORMAT, level=resolved.upper())

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "{time:YYYY-MM-DD_HH-mm-ss}.log",
        format=_FILE_FORMAT,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )


_initialized = False


def get_logger(name: str) -> Logger:
    """Return a logger bound to *name*."""
    global _initialized  # noqa: PLW0603
    if not _initialized:
        setup_logging()
        _initialized = True
    return logger.bind(name=name)
