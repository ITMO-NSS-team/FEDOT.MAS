import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "gaia_logs" / "log"
LOG_START_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
    r"(?P<level>\w+)\s+\| "
    r"(?P<source>[^|]+) \| "
    r"(?P<message>.*)$"
)
TRACE_RE = re.compile(r"name=gaia:(?P<run_id>[^:]+):(?P<task_id>[0-9a-f-]+)")
FAILED_TASK_RE = re.compile(
    r"Failed task (?P<task_id>[0-9a-f-]+) after all retries: (?P<error>.*)"
)
RAISED_RE = re.compile(r"raised (?P<error_type>\w+)>")
ATTEMPT_RE = re.compile(
    r"(?P<stage>\w+) attempt (?P<attempt>\d+)/(?P<max_attempts>\d+) failed: "
    r"(?P<error>.*?)(?:, retrying in \d+s\.\.\.|$)",
    re.DOTALL,
)
FAILED_STAGE_RE = re.compile(
    r"(?P<stage>\w+) failed after (?P<attempts>\d+) attempts: ?(?P<error>.*)",
    re.DOTALL,
)
MODEL_ERROR_RE = re.compile(
    r"Langfuse generation error \| agent=(?P<agent>\S+) error=(?P<error>.*)",
    re.DOTALL,
)
TOOL_CALL_RE = re.compile(r"Tool call \| agent=(?P<agent>\S+) tool=(?P<tool>\S+)")


def read_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.rstrip("\n")
            match = LOG_START_RE.match(line)
            if match:
                if current:
                    records.append(current)
                current = match.groupdict()
            elif current:
                current["message"] += "\n" + line

    if current:
        records.append(current)
    return records


def short_error(error: str) -> str:
    error = " ".join(error.split())
    if not error:
        return ""
    if "maximum context length" in error:
        return "maximum context length exceeded"
    if "Cannot specify both 'agent_name' and 'children'" in error:
        return "Cannot specify both agent_name and children"
    if "must have at least one child" in error:
        return "sequential node has no children"
    if "validation error for MAWConfig" in error:
        return "MAWConfig validation error"
    if "TimeoutError" in error:
        return "TimeoutError"
    return error[:240]


def classify_cause(data: dict[str, Any]) -> str:
    model_errors = " ".join(data["model_errors"])
    validation_errors = " ".join(data["validation_errors"])
    failed_stage_errors = " ".join(data["failed_stage_errors"])
    final_error = data.get("final_error", "")

    if (
        "maximum context length" in model_errors
        or data.get("final_error_type") == "BadRequestError"
    ):
        return "model_context_limit"
    if validation_errors or "validation error for MAWConfig" in failed_stage_errors:
        if data.get("final_error_type") == "TimeoutError":
            return "pipeline_config_validation_then_timeout"
        return "pipeline_config_validation"
    if data.get("failed_stage") == "pipeline_generator" and (
        data.get("final_error_type") in {"", "TimeoutError"}
        or "TimeoutError" in final_error
    ):
        return "pipeline_generator_timeout"
    if data["attempt_failures"] and all(
        failure["stage"] == "pipeline_generator" for failure in data["attempt_failures"]
    ):
        return "pipeline_generator_timeout"
    if data.get("final_error_type"):
        return data["final_error_type"]
    if data["attempt_failures"]:
        return "retry_attempts_without_final_error"
    return "no_failure_detected"


def direct_cause_text(data: dict[str, Any]) -> str:
    cause = data["cause"]
    if cause == "pipeline_generator_timeout":
        return "pipeline_generator did not return within the meta-agent timeout across retries"
    if cause == "pipeline_config_validation_then_timeout":
        return (
            "pipeline_generator produced invalid MAWConfig on at least one retry, "
            "then ended with TimeoutError"
        )
    if cause == "pipeline_config_validation":
        detail = (
            data["validation_errors"][-1]
            if data["validation_errors"]
            else "MAWConfig validation error"
        )
        return f"pipeline_generator produced invalid MAWConfig: {detail}"
    if cause == "model_context_limit":
        return "model prompt exceeded the provider context window"
    if cause == "retry_attempts_without_final_error":
        return "retry attempts failed, but the log ended before a final error line"
    if cause == "no_failure_detected":
        return "no error pattern detected"
    return data.get("final_error_type") or cause


def analyze_file(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "file": str(path),
        "run_id": None,
        "task_id": None,
        "start_time": None,
        "end_time": None,
        "final_error": "",
        "final_error_type": "",
        "failed_stage": "",
        "failed_stage_errors": [],
        "attempt_failures": [],
        "validation_errors": [],
        "model_errors": [],
        "langfuse_timeouts": 0,
        "tools": Counter(),
    }

    records = read_records(path)
    if records:
        data["start_time"] = records[0]["time"]
        data["end_time"] = records[-1]["time"]

    for record in records:
        message = record["message"]

        trace_match = TRACE_RE.search(message)
        if trace_match:
            data["run_id"] = trace_match.group("run_id")
            data["task_id"] = trace_match.group("task_id")

        failed_task_match = FAILED_TASK_RE.search(message)
        if failed_task_match:
            data["task_id"] = failed_task_match.group("task_id")
            data["final_error"] = failed_task_match.group("error")
            raised_match = RAISED_RE.search(data["final_error"])
            if raised_match:
                data["final_error_type"] = raised_match.group("error_type")

        attempt_match = ATTEMPT_RE.search(message)
        if attempt_match:
            failure = {
                "stage": attempt_match.group("stage"),
                "attempt": int(attempt_match.group("attempt")),
                "max_attempts": int(attempt_match.group("max_attempts")),
                "error": short_error(attempt_match.group("error")),
            }
            data["attempt_failures"].append(failure)
            if "validation error for MAWConfig" in message:
                data["validation_errors"].append(short_error(message))

        failed_stage_match = FAILED_STAGE_RE.search(message)
        if failed_stage_match:
            data["failed_stage"] = failed_stage_match.group("stage")
            stage_error = short_error(failed_stage_match.group("error"))
            data["failed_stage_errors"].append(stage_error)
            if "validation error for MAWConfig" in message:
                data["validation_errors"].append(short_error(message))

        model_error_match = MODEL_ERROR_RE.search(message)
        if model_error_match:
            data["model_errors"].append(short_error(model_error_match.group("error")))

        if (
            "Langfuse flush timed out" in message
            or "Langfuse shutdown timed out" in message
        ):
            data["langfuse_timeouts"] += 1

        tool_call_match = TOOL_CALL_RE.search(message)
        if tool_call_match:
            data["tools"][tool_call_match.group("tool")] += 1

    data["tools"] = dict(data["tools"])
    data["cause"] = classify_cause(data)
    data["direct_cause"] = direct_cause_text(data)
    return data


def load_analyses(log_dir: Path, run_id: str | None) -> list[dict[str, Any]]:
    analyses = []
    for path in sorted(log_dir.glob("*.log")):
        analysis = analyze_file(path)
        if run_id and analysis["run_id"] != run_id:
            continue
        analyses.append(analysis)
    return analyses


def summarize(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item for item in analyses if item["cause"] != "no_failure_detected"]
    final_error_types = Counter(
        item["final_error_type"] or "unknown" for item in failed
    )
    failed_stages = Counter(item["failed_stage"] or "unknown" for item in failed)
    causes = Counter(item["cause"] for item in failed)
    direct_causes = Counter(item["direct_cause"] for item in failed)

    unique_failed_tasks = {
        item["task_id"] for item in failed if item.get("task_id") is not None
    }

    return {
        "log_files": len(analyses),
        "failed_log_files": len(failed),
        "unique_failed_tasks": len(unique_failed_tasks),
        "causes": dict(causes),
        "direct_causes": dict(direct_causes),
        "final_error_types": dict(final_error_types),
        "failed_stages": dict(failed_stages),
        "langfuse_timeout_files": sum(
            1 for item in analyses if item["langfuse_timeouts"] > 0
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("GAIA error log analysis")
    print()
    print(f"Log files: {summary['log_files']}")
    print(f"Failed log files: {summary['failed_log_files']}")
    print(f"Unique failed tasks: {summary['unique_failed_tasks']}")
    print(
        f"Files with Langfuse flush/shutdown timeouts: {summary['langfuse_timeout_files']}"
    )

    print()
    print("Direct causes")
    for cause, count in sorted(
        summary["direct_causes"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {cause}: {count}")

    print()
    print("Final error types")
    for error_type, count in sorted(
        summary["final_error_types"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {error_type}: {count}")

    print()
    print("Failed stages")
    for stage, count in sorted(
        summary["failed_stages"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {stage}: {count}")


def print_details(analyses: list[dict[str, Any]]) -> None:
    print()
    print("Failures")
    for item in analyses:
        if item["cause"] == "no_failure_detected":
            continue
        print(
            f"  {Path(item['file']).name} task={item['task_id']} "
            f"run={item['run_id']} cause={item['cause']} "
            f"final={item['final_error_type'] or 'unknown'} "
            f"stage={item['failed_stage'] or 'unknown'}"
        )
        print(f"    direct_cause={item['direct_cause']}")
        if item["model_errors"]:
            print(f"    model_error={item['model_errors'][-1]}")
        if item["validation_errors"]:
            print(f"    validation={item['validation_errors'][-1]}")
        elif item["failed_stage_errors"]:
            print(f"    stage_error={item['failed_stage_errors'][-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detailed GAIA debug logs.")
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing timestamped .log files.",
    )
    parser.add_argument(
        "--run-id",
        help="Only analyze logs whose Langfuse trace name has this GAIA run id.",
    )
    parser.add_argument(
        "--details", action="store_true", help="Print per-file failures."
    )
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args()

    analyses = load_analyses(args.log_dir.resolve(), args.run_id)
    summary = summarize(analyses)

    if args.json:
        print(json.dumps({"summary": summary, "logs": analyses}, indent=2))
        return

    print_summary(summary)
    if args.details:
        print_details(analyses)


if __name__ == "__main__":
    main()
