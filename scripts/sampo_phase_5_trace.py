"""Durable trace I/O for the Phase 5 harnesses."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def write_trace_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a trace so readers see either the old or the complete new JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = file.name
            json.dump(payload, file)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def read_trace(path: Path, *, attempts: int = 20, delay_s: float = 0.05) -> dict[str, Any]:
    """Read a trace briefly retrying empty/partial files after process failure."""
    for _ in range(attempts):
        try:
            if path.exists():
                with path.open(encoding="utf-8") as file:
                    payload = json.load(file)
                if isinstance(payload, dict):
                    return payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        time.sleep(delay_s)
    return {"model_calls": [], "tool_calls": [], "failures": ["trace_unreadable"]}
