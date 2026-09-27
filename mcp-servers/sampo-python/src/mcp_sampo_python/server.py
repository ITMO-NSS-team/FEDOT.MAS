from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
WORKSPACE = "/home/oai/share"
MAX_OUTPUT_CHARS = 12_000

load_dotenv(ROOT / ".env")

mcp = FastMCP("sampo-python")
_sandbox: AsyncSandbox | None = None
_prepared = False


def _public_inputs() -> list[dict[str, str]]:
    with (PUBLIC / "benchmark_inputs.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _public_labels() -> set[str]:
    with (PUBLIC / "allowed_target_labels.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        return {row["target_label"] for row in csv.DictReader(file)}


async def _get_sandbox() -> AsyncSandbox:
    global _sandbox
    if _sandbox is None:
        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            raise RuntimeError(
                "E2B_API_KEY is required for the isolated SAMPO Python workspace. "
                "Set it in .env and rerun."
            )
        _sandbox = await AsyncSandbox.create(
            api_key=api_key, allow_internet_access=False
        )
    return _sandbox


async def _prepare() -> AsyncSandbox:
    global _prepared
    sandbox = await _get_sandbox()
    if not _prepared:
        await sandbox.files.write(
            f"{WORKSPACE}/benchmark_inputs.csv",
            (PUBLIC / "benchmark_inputs.csv").read_text(encoding="utf-8"),
        )
        await sandbox.files.write(
            f"{WORKSPACE}/allowed_target_labels.csv",
            (PUBLIC / "allowed_target_labels.csv").read_text(encoding="utf-8"),
        )
        _prepared = True
    return sandbox


def _compact(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= MAX_OUTPUT_CHARS:
        return value
    return f"{value[:MAX_OUTPUT_CHARS]}\n...[truncated]"


def _serialize_execution(exc: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if exc.logs.stdout:
        output["stdout"] = _compact("\n".join(exc.logs.stdout))
    if exc.logs.stderr:
        output["stderr"] = _compact("\n".join(exc.logs.stderr))
    if exc.error:
        output["error"] = {
            "name": exc.error.name,
            "value": _compact(exc.error.value),
        }
    for result in exc.results:
        if result.text is not None:
            output.setdefault("results", []).append(_compact(result.text))
        if result.json is not None:
            output.setdefault("json_results", []).append(result.json)
    return output


@mcp.tool
async def prepare_public_workspace() -> dict[str, Any]:
    """Create a persistent isolated Python workspace containing only benchmark_inputs.csv and allowed_target_labels.csv.

    The workspace has no access to the FEDOT.MAS host filesystem, databases, or
    private ground truth. Its outbound network is disabled. Use the returned
    paths with run_code for normal Python data-science and batch processing.
    """
    try:
        await _prepare()
        return {
            "success": True,
            "workspace": WORKSPACE,
            "input_path": f"{WORKSPACE}/benchmark_inputs.csv",
            "labels_path": f"{WORKSPACE}/allowed_target_labels.csv",
            "examples": len(_public_inputs()),
            "allowed_labels": len(_public_labels()),
        }
    except Exception as exc:  # noqa: BLE001 - MCP tools return structured errors.
        return {"success": False, "error": str(exc)}


@mcp.tool
async def run_code(code: str) -> dict[str, Any]:
    """Run normal Python code in the persistent isolated workspace.

    The workspace supports filesystem-based batch processing and installed
    data-science libraries. Call prepare_public_workspace first. Code can read
    only the two public benchmark files supplied there and files it creates.
    """
    try:
        sandbox = await _prepare()
        result = await sandbox.run_code(code)
        return {"success": result.error is None, **_serialize_execution(result)}
    except Exception as exc:  # noqa: BLE001 - MCP tools return structured errors.
        return {"success": False, "error": str(exc)}


def _validate_predictions(content: str) -> tuple[list[dict[str, str]], int]:
    rows = list(csv.DictReader(io.StringIO(content)))
    required = {"example_id", "top_1", "top_2", "top_3"}
    if not rows or set(rows[0]) != required:
        raise ValueError(
            "Prediction CSV must contain exactly: example_id, top_1, top_2, top_3"
        )
    inputs = _public_inputs()
    valid_ids = {row["example_id"] for row in inputs}
    labels = _public_labels()
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids).issubset(valid_ids):
        raise ValueError(
            "Prediction CSV contains duplicate or unknown example_id values"
        )
    for row in rows:
        invalid = [
            row[column]
            for column in ("top_1", "top_2", "top_3")
            if row[column] not in labels
        ]
        if invalid:
            raise ValueError(
                "Every prediction must be one of the public allowed labels"
            )
    return rows, len(inputs)


@mcp.tool
async def save_predictions(
    sandbox_path: str, output_filename: str = "mas_predictions.csv"
) -> dict[str, Any]:
    """Validate and export a CSV prediction file created in the isolated workspace.

    The CSV must have example_id, top_1, top_2, top_3 and use only public
    example IDs and allowed labels. It is saved to the public mas_runs folder;
    no arbitrary host-file read or write is available.
    """
    try:
        sandbox = await _prepare()
        content = await sandbox.files.read(sandbox_path, format="text")
        rows, total = _validate_predictions(content)
        filename = Path(output_filename).name
        if not filename.endswith(".csv"):
            raise ValueError("output_filename must end in .csv")
        destination = PUBLIC / "mas_runs" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["example_id", "top_1", "top_2", "top_3"]
            )
            writer.writeheader()
            writer.writerows(rows)
        return {
            "success": True,
            "prediction_path": str(destination.relative_to(ROOT)),
            "examples": len(rows),
            "public_examples_available": total,
        }
    except Exception as exc:  # noqa: BLE001 - MCP tools return structured errors.
        return {"success": False, "error": str(exc)}


def main() -> None:
    mcp.run(show_banner=False)
