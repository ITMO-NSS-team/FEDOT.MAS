"""Reject file extraction handoffs backed only by archive/download metadata.

Receipts come exclusively from executed tool callbacks, never model prose.
This detects missing reads, not factual correctness or complete source coverage.
"""

import hashlib
import json
import posixpath
from typing import Any

from fedotmas.maw.handoffs import describe_requirement, parse_artifact
from fedotmas.maw.models import MAWAgentConfig
from fedotmas.mcp.capabilities import normalize_tool_name

FILE_EVIDENCE_STATE_KEY = "__fedotmas_file_evidence"
_PREPARATION = {"list_zip_contents", "extract_zip", "download", "download_file"}
_READERS = {"read_document", "find_document", "read_file"}
_WEB_READERS = {
    "markdown",
    "html",
    "extract",
    "eval",
    "evaluate",
    "structureddata",
    "tree",
    "semantic_tree",
}


def begin_file_evidence(state: Any, agent: str) -> None:
    """A later loop/recovery invocation must not borrow an earlier read receipt."""
    root = dict(state.get(FILE_EVIDENCE_STATE_KEY) or {})
    agents = dict(root.get("agents") or {})
    agents[agent] = {}
    root["agents"] = agents
    state[FILE_EVIDENCE_STATE_KEY] = root


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _unwrap(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        try:
            return _unwrap(json.loads(value), depth + 1)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_unwrap(v, depth + 1) for v in value[:128]]
    if isinstance(value, dict):
        if (
            any(value.get(k) for k in ("error", "isError", "is_error", "error_code"))
            or value.get("success") is False
        ):
            return None
        for key in ("structuredContent", "structured_content", "result"):
            if key in value:
                return _unwrap(value[key], depth + 1)
        if isinstance(value.get("content"), list):
            return _unwrap(value["content"], depth + 1)
        if value.get("type") == "text" or set(value) == {"text"}:
            return _unwrap(value.get("text"), depth + 1)
    return value


def _scalars(value: Any, depth: int = 0) -> set[str]:
    if value is None:
        return set()
    if depth > 20:
        return {"unsupported_depth"}
    if isinstance(value, dict):
        if len(value) > 256:
            return {"unsupported_width"}
        return set().union(
            *(_scalars(v, depth + 1) for v in list(value.values())[:256])
        )
    if isinstance(value, list):
        if len(value) > 256:
            return {"unsupported_width"}
        return set().union(*(_scalars(v, depth + 1) for v in value[:256]))
    return {_digest(value)}


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (_content(v) for v in value)))
    if isinstance(value, dict):
        # Counts, cursors and paths do not prove that source contents were read.
        return _content(
            value.get("content") or value.get("text") or value.get("matches")
        )
    return ""


def record_file_evidence(
    state: Any, agent: str, tool: str, args: dict, response: Any
) -> None:
    name = normalize_tool_name(tool)
    if name not in _PREPARATION | _READERS | _WEB_READERS | {"solve_with_code"}:
        return
    root = dict(state.get(FILE_EVIDENCE_STATE_KEY) or {})
    agents = dict(root.get("agents") or {})
    entry = dict(agents.get(agent) or {})
    payload = _unwrap(response)
    if name in _PREPARATION:
        entry["prepared"] = True
        # A file locator may legitimately hand off paths/names returned by a
        # listing or download, without reading the files themselves.
        values = set(entry.get("metadata_values") or []) | _scalars(payload)
        if isinstance(payload, dict):
            directory, files = payload.get("output_dir"), payload.get("files")
            if isinstance(directory, str) and isinstance(files, list):
                values |= {
                    _digest(posixpath.normpath(posixpath.join(directory, f)))
                    for f in files
                    if isinstance(f, str)
                }
            # The archive's own path is a legitimate locator too.
            values |= {
                _digest(args[k])
                for k in ("file_path", "path", "url")
                if isinstance(args.get(k), str)
            }
        entry["metadata_values"] = sorted(v for v in values if len(v) == 64)[:2048]
    else:
        text = _content(payload)
        if name == "solve_with_code":
            text = (
                str(payload.get("answer") or "")
                if (
                    isinstance(payload, dict)
                    and payload.get("status") == "completed"
                    and payload.get("files_used")
                    and args.get("files")
                )
                else ""
            )
        if not text:
            return
        reads = list(entry.get("reads") or [])
        receipt = {
            "tool": name,
            "source": _digest(
                args.get("file_path")
                or args.get("path")
                or args.get("url")
                or args.get("files")
            ),
            "content": _digest(text),
            "range": {
                k: args[k]
                for k in ("start_line", "start_char", "max_lines", "max_chars")
                if k in args
            },
        }
        if receipt not in reads:
            entry["reads"] = [*reads, receipt][-64:]
    agents[agent] = entry
    root["agents"] = agents
    state[FILE_EVIDENCE_STATE_KEY] = root


def _has_read_evidence(state: Any, cfg: MAWAgentConfig) -> bool:
    root = state.get(FILE_EVIDENCE_STATE_KEY) or {}
    if ((root.get("agents") or {}).get(cfg.name) or {}).get("reads"):
        return True
    artifacts = root.get("artifacts") or {}
    return any(
        artifacts.get(r.source_key) == _digest(state.get(r.source_key))
        and not describe_requirement(state, r)[1]
        for r in cfg.input_requirements
    )


def unsupported_file_artifact(state: Any, cfg: MAWAgentConfig, value: Any) -> bool:
    root = state.get(FILE_EVIDENCE_STATE_KEY) or {}
    entry = (root.get("agents") or {}).get(cfg.name) or {}
    if not entry.get("prepared") or _has_read_evidence(state, cfg):
        return False
    artifact = parse_artifact(value, cfg.output_contract)
    if artifact is None:
        return False  # Existing contract validation handles malformed artifacts.
    return bool(_scalars(artifact) - set(entry.get("metadata_values") or []))


def remember_file_artifact(state: Any, cfg: MAWAgentConfig) -> None:
    if not _has_read_evidence(state, cfg):
        return
    root = dict(state.get(FILE_EVIDENCE_STATE_KEY) or {})
    artifacts = dict(root.get("artifacts") or {})
    artifacts[cfg.output_key] = _digest(state.get(cfg.output_key))
    root["artifacts"] = artifacts
    state[FILE_EVIDENCE_STATE_KEY] = root


def reject_file_artifact(state: Any, output_key: str, reason: str) -> str:
    value = json.dumps({"status": "unresolved", "reason": reason})
    state[output_key] = value
    root = dict(state.get(FILE_EVIDENCE_STATE_KEY) or {})
    rejected = dict(root.get("rejected") or {})
    rejected[output_key] = _digest(value)
    root["rejected"] = rejected
    state[FILE_EVIDENCE_STATE_KEY] = root
    return value


def rejected_file_inputs(state: Any, keys: list[str]) -> list[str]:
    rejected = (state.get(FILE_EVIDENCE_STATE_KEY) or {}).get("rejected") or {}
    return [key for key in keys if rejected.get(key) == _digest(state.get(key))]
