import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent
    / "gaia_logs"
    / "run_be8ea22a-34ed-440a-b0be-cf70f516251a"
)

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}")
RUN_START_RE = re.compile(r"Full-auto run for task:")
RESULT_RE = re.compile(r"\[(?P<status>CORRECT|WRONG)\] task=(?P<task_id>[0-9a-f-]+)")
FAILED_TASK_RE = re.compile(r"Failed task (?P<task_id>[0-9a-f-]+) after all retries")
AGENT_STARTED_RE = re.compile(r"Agent started \| name=(?P<agent>\S+)")
AGENT_DONE_RE = re.compile(
    r"Agent done \| name=(?P<agent>\S+) elapsed=(?P<elapsed>[\d.]+)s"
)
GEN_END_RE = re.compile(
    r"Langfuse generation ended \| agent=(?P<agent>\S+) usage=(?P<usage>\{.*\})"
)
META_COMPLETE_RE = re.compile(
    r"(?P<stage>pool_generator|pipeline_generator) complete "
    r"\| elapsed=(?P<elapsed>[\d.]+)s prompt=(?P<prompt>\d+) completion=(?P<completion>\d+)"
)
ATTEMPT_FAILED_RE = re.compile(
    r"(?P<stage>\w+) attempt (?P<attempt>\d+)/(?P<max_attempts>\d+) failed: "
    r"(?P<error>.*?)(?:, retrying in (?P<delay>\d+)s\.\.\.|$)",
    re.DOTALL,
)
FAILED_AFTER_RE = re.compile(
    r"(?P<stage>\w+) failed after (?P<attempts>\d+) attempts: ?(?P<error>.*)",
    re.DOTALL,
)
PIPELINE_COMPLETE_RE = re.compile(
    r"Pipeline complete \| total_elapsed=(?P<elapsed>[\d.]+)s "
    r"total_prompt=(?P<prompt>\d+) total_completion=(?P<completion>\d+)"
)
TOOL_CALL_RE = re.compile(r"Tool call \| agent=(?P<agent>\S+) tool=(?P<tool>\S+)")


@dataclass
class Record:
    time: datetime
    message: str


@dataclass
class Attempt:
    stage: str
    attempt: int
    started_at: datetime | None = None
    ended_at: datetime | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""
    status: str = "unknown"

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class TaskRun:
    index: int
    started_at: datetime
    task_id: str | None = None
    status: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    meta_complete: dict[str, dict[str, float | int]] = field(default_factory=dict)
    pipeline_elapsed: float | None = None
    pipeline_prompt: int = 0
    pipeline_completion: int = 0
    tool_calls: Counter[str] = field(default_factory=Counter)
    searxng_errors: int = 0
    langfuse_flush_timeouts: int = 0
    failed_stage: str | None = None
    failed_error: str = ""

    @property
    def ended_at(self) -> datetime | None:
        times: list[datetime] = []
        for attempt in self.attempts:
            if attempt.ended_at:
                times.append(attempt.ended_at)
        return max(times) if times else None


def read_records(path: Path) -> list[Record]:
    records: list[Record] = []
    current_time: datetime | None = None
    current_lines: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            match = TIMESTAMP_RE.search(line)
            if match:
                if current_time is not None:
                    records.append(Record(current_time, "\n".join(current_lines)))
                current_time = datetime.strptime(match.group(0), "%Y-%m-%d %H:%M:%S.%f")
                current_lines = [line[match.end() :].strip()]
            elif current_time is not None:
                current_lines.append(line.strip())

    if current_time is not None:
        records.append(Record(current_time, "\n".join(current_lines)))
    return records


def load_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(run_dir.glob("task_*/result.json")):
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        task_id = str(data.get("task_id") or path.parent.name.removeprefix("task_"))
        results[task_id] = data
    return results


def parse_usage(text: str) -> dict[str, int]:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {k: int(v) for k, v in value.items() if isinstance(v, int)}


def short_error(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return "timeout_or_empty_error"
    if "Cannot specify both 'agent_name' and 'children'" in text:
        return "agent_name_children_conflict"
    if "Invalid JSON" in text:
        return "invalid_json"
    if "validation error for MAWConfig" in text:
        return "mawconfig_validation"
    if "maximum context length" in text:
        return "context_limit"
    return text[:160]


def analyze_records(records: list[Record]) -> list[TaskRun]:
    runs: list[TaskRun] = []
    current: TaskRun | None = None
    active_attempts: dict[str, Attempt] = {}
    stage_attempt_counts: defaultdict[str, int] = defaultdict(int)

    def ensure_run(record: Record) -> TaskRun:
        nonlocal current
        if current is None:
            current = TaskRun(index=len(runs) + 1, started_at=record.time)
            runs.append(current)
        return current

    for record in records:
        msg = record.message

        if RUN_START_RE.search(msg):
            current = TaskRun(index=len(runs) + 1, started_at=record.time)
            runs.append(current)
            active_attempts = {}
            stage_attempt_counts = defaultdict(int)
            continue

        run = ensure_run(record)

        result_match = RESULT_RE.search(msg)
        if result_match:
            run.task_id = result_match.group("task_id")
            run.status = result_match.group("status")

        failed_task_match = FAILED_TASK_RE.search(msg)
        if failed_task_match:
            run.task_id = failed_task_match.group("task_id")
            run.status = "ERROR"

        agent_started = AGENT_STARTED_RE.search(msg)
        if agent_started:
            agent = agent_started.group("agent")
            if agent in {"pool_generator", "pipeline_generator"}:
                stage_attempt_counts[agent] += 1
                attempt = Attempt(
                    stage=agent,
                    attempt=stage_attempt_counts[agent],
                    started_at=record.time,
                )
                run.attempts.append(attempt)
                active_attempts[agent] = attempt

        gen_end = GEN_END_RE.search(msg)
        if gen_end:
            agent = gen_end.group("agent")
            usage = parse_usage(gen_end.group("usage"))
            attempt = active_attempts.get(agent)
            if attempt is not None:
                attempt.ended_at = record.time
                attempt.prompt_tokens = usage.get("input", 0)
                attempt.completion_tokens = usage.get("output", 0)
                attempt.total_tokens = usage.get("total", 0)
                attempt.status = "model_returned"

        agent_done = AGENT_DONE_RE.search(msg)
        if agent_done:
            agent = agent_done.group("agent")
            attempt = active_attempts.get(agent)
            if attempt is not None and attempt.ended_at is None:
                attempt.ended_at = record.time
                attempt.status = "agent_done"

        meta_complete = META_COMPLETE_RE.search(msg)
        if meta_complete:
            stage = meta_complete.group("stage")
            run.meta_complete[stage] = {
                "elapsed": float(meta_complete.group("elapsed")),
                "prompt": int(meta_complete.group("prompt")),
                "completion": int(meta_complete.group("completion")),
            }
            attempt = active_attempts.get(stage)
            if attempt is not None:
                attempt.status = "success"
                attempt.ended_at = attempt.ended_at or record.time

        attempt_failed = ATTEMPT_FAILED_RE.search(msg)
        if attempt_failed:
            stage = attempt_failed.group("stage")
            attempt = active_attempts.get(stage)
            if attempt is not None:
                attempt.ended_at = attempt.ended_at or record.time
                attempt.error = short_error(attempt_failed.group("error"))
                attempt.status = "failed"

        failed_after = FAILED_AFTER_RE.search(msg)
        if failed_after:
            stage = failed_after.group("stage")
            run.failed_stage = stage
            run.failed_error = short_error(failed_after.group("error"))
            attempt = active_attempts.get(stage)
            if attempt is not None:
                attempt.ended_at = attempt.ended_at or record.time
                attempt.error = run.failed_error
                attempt.status = "failed_final"

        pipeline_complete = PIPELINE_COMPLETE_RE.search(msg)
        if pipeline_complete:
            run.pipeline_elapsed = float(pipeline_complete.group("elapsed"))
            run.pipeline_prompt = int(pipeline_complete.group("prompt"))
            run.pipeline_completion = int(pipeline_complete.group("completion"))

        tool_call = TOOL_CALL_RE.search(msg)
        if tool_call:
            run.tool_calls[tool_call.group("tool")] += 1

        if "SearXNG" in msg and "connection error" in msg:
            run.searxng_errors += 1
        if "Langfuse flush timed out" in msg or "Langfuse shutdown timed out" in msg:
            run.langfuse_flush_timeouts += 1

    return runs


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "avg": round(mean(values), 2),
        "median": round(median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def summarize(
    runs: list[TaskRun], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    attempts = [attempt for run in runs for attempt in run.attempts]
    by_stage = defaultdict(list)
    completion_by_stage = defaultdict(list)
    prompt_by_stage = defaultdict(list)
    errors = Counter()
    status = Counter()
    long_attempts = []

    for attempt in attempts:
        if attempt.elapsed is not None:
            by_stage[attempt.stage].append(attempt.elapsed)
        if attempt.completion_tokens:
            completion_by_stage[attempt.stage].append(float(attempt.completion_tokens))
        if attempt.prompt_tokens:
            prompt_by_stage[attempt.stage].append(float(attempt.prompt_tokens))
        if attempt.error:
            errors[attempt.error] += 1
        status[attempt.status] += 1
        if (attempt.elapsed or 0) >= 60 or attempt.completion_tokens >= 8000:
            long_attempts.append(attempt)

    result_elapsed = [
        float(item["elapsed"])
        for item in results.values()
        if isinstance(item.get("elapsed"), int | float)
    ]
    result_errors = Counter(
        "error" if item.get("error") else "completed" for item in results.values()
    )
    result_correct = sum(
        1 for item in results.values() if item.get("is_correct") is True
    )

    return {
        "tasks_in_results": len(results),
        "tasks_in_logs": len(runs),
        "result_status": dict(result_errors),
        "result_accuracy": round(result_correct / len(results) * 100, 2)
        if results
        else None,
        "result_elapsed": describe(result_elapsed),
        "stage_elapsed": {
            stage: describe(values) for stage, values in by_stage.items()
        },
        "stage_prompt_tokens": {
            stage: describe(values) for stage, values in prompt_by_stage.items()
        },
        "stage_completion_tokens": {
            stage: describe(values) for stage, values in completion_by_stage.items()
        },
        "attempt_status": dict(status),
        "attempt_errors": dict(errors),
        "pipeline_runtime": describe(
            [run.pipeline_elapsed for run in runs if run.pipeline_elapsed is not None]
        ),
        "tool_calls": dict(sum((run.tool_calls for run in runs), Counter())),
        "searxng_connection_errors": sum(run.searxng_errors for run in runs),
        "langfuse_flush_timeouts": sum(run.langfuse_flush_timeouts for run in runs),
        "long_meta_attempts": [
            {
                "task_index": find_run_index(runs, attempt),
                "stage": attempt.stage,
                "attempt": attempt.attempt,
                "elapsed": round(attempt.elapsed or 0, 2),
                "prompt": attempt.prompt_tokens,
                "completion": attempt.completion_tokens,
                "status": attempt.status,
                "error": attempt.error,
            }
            for attempt in long_attempts
        ],
    }


def find_run_index(runs: list[TaskRun], target: Attempt) -> int:
    for run in runs:
        if target in run.attempts:
            return run.index
    return -1


def print_summary(summary: dict[str, Any]) -> None:
    print("GAIA Timing Analysis")
    print()
    print(
        f"Tasks: results={summary['tasks_in_results']} logs={summary['tasks_in_logs']} "
        f"accuracy={summary['result_accuracy']}%"
    )
    print(
        "Result elapsed: "
        f"avg={summary['result_elapsed']['avg']}s "
        f"median={summary['result_elapsed']['median']}s "
        f"max={summary['result_elapsed']['max']}s"
    )
    print()
    print("Meta Stage Elapsed")
    for stage, stats in sorted(summary["stage_elapsed"].items()):
        print(
            f"  {stage}: count={stats['count']} avg={stats['avg']}s "
            f"median={stats['median']}s max={stats['max']}s"
        )
    print()
    print("Meta Completion Tokens")
    for stage, stats in sorted(summary["stage_completion_tokens"].items()):
        print(
            f"  {stage}: count={stats['count']} avg={stats['avg']} "
            f"median={stats['median']} max={stats['max']}"
        )
    print()
    print("Failures / Retries")
    print(f"  attempt_status={summary['attempt_status']}")
    print(f"  attempt_errors={summary['attempt_errors']}")
    print()
    print("Pipeline Runtime")
    stats = summary["pipeline_runtime"]
    print(
        f"  count={stats['count']} avg={stats['avg']}s "
        f"median={stats['median']}s max={stats['max']}s"
    )
    print()
    print("Tools / Infra")
    print(f"  tool_calls={summary['tool_calls']}")
    print(f"  searxng_connection_errors={summary['searxng_connection_errors']}")
    print(f"  langfuse_flush_timeouts={summary['langfuse_flush_timeouts']}")
    if summary["long_meta_attempts"]:
        print()
        print("Long Meta Attempts")
        for item in summary["long_meta_attempts"][:20]:
            print(
                f"  task#{item['task_index']} {item['stage']} attempt={item['attempt']} "
                f"elapsed={item['elapsed']}s completion={item['completion']} "
                f"status={item['status']} error={item['error']}"
            )


def print_details(runs: list[TaskRun]) -> None:
    print()
    print("Runs")
    for run in runs:
        label = f"task={run.task_id}" if run.task_id else f"task_index={run.index}"
        print(f"  #{run.index} {label} status={run.status or 'unknown'}")
        for attempt in run.attempts:
            print(
                f"    {attempt.stage} attempt={attempt.attempt} "
                f"elapsed={round(attempt.elapsed or 0, 2)}s "
                f"prompt={attempt.prompt_tokens} completion={attempt.completion_tokens} "
                f"status={attempt.status} error={attempt.error}"
            )
        if run.pipeline_elapsed is not None:
            print(
                f"    pipeline elapsed={run.pipeline_elapsed}s "
                f"prompt={run.pipeline_prompt} completion={run.pipeline_completion}"
            )
        if run.tool_calls:
            print(
                f"    tools={dict(run.tool_calls)} searxng_errors={run.searxng_errors}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GAIA run timing, meta-stage retries, tokens, and tool loops."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Path to examples/gaia/gaia_logs/run_<uuid>.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Explicit log file. Defaults to <run_dir>/logs.log.",
    )
    parser.add_argument(
        "--details", action="store_true", help="Print per-run attempts."
    )
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_file = (args.log_file or run_dir / "logs.log").resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    records = read_records(log_file)
    runs = analyze_records(records)
    results = load_results(run_dir)
    summary = summarize(runs, results)

    if args.json:
        payload = {
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "summary": summary,
            "runs": [
                {
                    "index": run.index,
                    "task_id": run.task_id,
                    "status": run.status,
                    "pipeline_elapsed": run.pipeline_elapsed,
                    "tool_calls": dict(run.tool_calls),
                    "searxng_errors": run.searxng_errors,
                    "attempts": [
                        {
                            "stage": attempt.stage,
                            "attempt": attempt.attempt,
                            "elapsed": attempt.elapsed,
                            "prompt": attempt.prompt_tokens,
                            "completion": attempt.completion_tokens,
                            "status": attempt.status,
                            "error": attempt.error,
                        }
                        for attempt in run.attempts
                    ],
                }
                for run in runs
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print_summary(summary)
    if args.details:
        print_details(runs)


if __name__ == "__main__":
    main()
