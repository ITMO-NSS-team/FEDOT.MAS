import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_LOGS_DIR = Path(__file__).resolve().parent / "gaia_logs"
TOKEN_FIELDS = (
    "meta_prompt",
    "meta_completion",
    "pipeline_prompt",
    "pipeline_completion",
    "total_prompt",
    "total_completion",
)


def latest_run_dir(logs_dir: Path) -> Path:
    run_dirs = sorted(
        (path for path in logs_dir.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {logs_dir}")
    return run_dirs[0]


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    result_paths = sorted(run_dir.glob("task_*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No task_*/result.json files found under {run_dir}")

    results = []
    for path in result_paths:
        try:
            with path.open(encoding="utf-8") as file:
                result = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        result["_path"] = str(path)
        results.append(result)
    return results


def error_type(error: str) -> str:
    if not error:
        return ""
    return error.split("[", 1)[0].split(":", 1)[0].strip() or error


def percent(part: int, total: int) -> float:
    return round((part / total) * 100, 2) if total else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for result in results if result.get("is_correct") is True)
    failures = [result for result in results if result.get("error")]
    completed = [result for result in results if not result.get("error")]
    completed_correct = sum(
        1 for result in completed if result.get("is_correct") is True
    )

    by_level: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "completed": 0, "failed": 0}
    )
    for result in results:
        level = str(result.get("difficulty", "unknown"))
        by_level[level]["total"] += 1
        if result.get("is_correct") is True:
            by_level[level]["correct"] += 1
        if result.get("error"):
            by_level[level]["failed"] += 1
        else:
            by_level[level]["completed"] += 1

    elapsed_values = [
        float(result["elapsed"])
        for result in results
        if isinstance(result.get("elapsed"), int | float)
    ]

    token_totals = {field: 0 for field in TOKEN_FIELDS}
    token_result_count = 0
    for result in results:
        tokens = result.get("tokens")
        if not isinstance(tokens, dict):
            continue
        token_result_count += 1
        for field in TOKEN_FIELDS:
            value = tokens.get(field, 0)
            if isinstance(value, int | float):
                token_totals[field] += int(value)

    return {
        "total_tasks": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": percent(correct, total),
        "completed_tasks": len(completed),
        "failed_tasks": len(failures),
        "failure_rate": percent(len(failures), total),
        "completed_accuracy": percent(completed_correct, len(completed)),
        "by_level": {
            level: {
                **stats,
                "accuracy": percent(stats["correct"], stats["total"]),
                "failure_rate": percent(stats["failed"], stats["total"]),
            }
            for level, stats in sorted(
                by_level.items(),
                key=lambda item: (item[0] == "unknown", item[0]),
            )
        },
        "errors": dict(
            Counter(error_type(str(result.get("error", ""))) for result in failures)
        ),
        "elapsed": {
            "count": len(elapsed_values),
            "total_seconds": round(sum(elapsed_values), 2),
            "avg_seconds": round(mean(elapsed_values), 2) if elapsed_values else 0.0,
            "median_seconds": round(median(elapsed_values), 2)
            if elapsed_values
            else 0.0,
        },
        "tokens": {
            "result_count": token_result_count,
            "totals": token_totals,
            "grand_total": token_totals["total_prompt"]
            + token_totals["total_completion"],
        },
    }


def print_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    print(f"GAIA log analysis: {run_dir}")
    print()
    print("Overall")
    print(
        f"  Accuracy: {summary['accuracy']:.2f}% "
        f"({summary['correct']}/{summary['total_tasks']})"
    )
    print(
        f"  Completed: {summary['completed_tasks']}  "
        f"Failed: {summary['failed_tasks']} "
        f"({summary['failure_rate']:.2f}%)"
    )
    print(f"  Completed-only accuracy: {summary['completed_accuracy']:.2f}%")

    print()
    print("By level")
    for level, stats in summary["by_level"].items():
        print(
            f"  Level {level}: {stats['accuracy']:.2f}% "
            f"({stats['correct']}/{stats['total']}), "
            f"completed={stats['completed']}, failed={stats['failed']}"
        )

    if summary["errors"]:
        print()
        print("Errors")
        for name, count in sorted(
            summary["errors"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {name}: {count}")

    elapsed = summary["elapsed"]
    if elapsed["count"]:
        print()
        print("Elapsed")
        print(
            f"  Total: {elapsed['total_seconds']:.2f}s  "
            f"Avg: {elapsed['avg_seconds']:.2f}s  "
            f"Median: {elapsed['median_seconds']:.2f}s"
        )

    tokens = summary["tokens"]
    if tokens["result_count"]:
        totals = tokens["totals"]
        print()
        print("Tokens")
        print(
            f"  Results with tokens: {tokens['result_count']}  "
            f"Grand total: {tokens['grand_total']:,}"
        )
        print(
            f"  Meta: {totals['meta_prompt'] + totals['meta_completion']:,} "
            f"(prompt={totals['meta_prompt']:,}, "
            f"completion={totals['meta_completion']:,})"
        )
        print(
            f"  Pipeline: "
            f"{totals['pipeline_prompt'] + totals['pipeline_completion']:,} "
            f"(prompt={totals['pipeline_prompt']:,}, "
            f"completion={totals['pipeline_completion']:,})"
        )


def print_details(results: list[dict[str, Any]]) -> None:
    print()
    print("Tasks")
    for result in results:
        status = "ERROR" if result.get("error") else "OK"
        if result.get("is_correct") is True:
            score = "correct"
        else:
            score = "wrong"
        print(
            f"  {result.get('task_id')}  "
            f"level={result.get('difficulty')}  "
            f"{status}  {score}"
        )
        if result.get("error"):
            print(f"    error={result['error']}")
        elif result.get("is_correct") is not True:
            print(f"    response={result.get('response', '')}")
            print(f"    ground_truth={result.get('ground_truth', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved GAIA run logs.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to benchmarks/gaia/gaia_logs/run_<uuid>. Defaults to latest run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON summary.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-task status and wrong answers.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir(DEFAULT_LOGS_DIR)
    run_dir = run_dir.resolve()
    results = load_results(run_dir)
    summary = summarize(results)

    if args.json:
        print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
        return

    print_summary(run_dir, summary)
    if args.details:
        print_details(results)


if __name__ == "__main__":
    main()
