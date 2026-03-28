import argparse
import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from fedotmas import MAW
from fedotmas.common.logging import get_logger

from examples.gaia.data import GaiaBenchmark

load_dotenv()

RUN_ID = uuid.uuid4()
_log = get_logger("fedotmas.examples.gaia")


def extract_solution(text: str) -> str:
    """Extract answer from <solution> tags, or return stripped text."""
    pattern = r"<solution>(.*?)</solution>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def extract_answer_from_state(state: dict[str, Any]) -> str:
    """Extract final answer from session state.

    First looks for <solution> tags in any state value.
    Falls back to the last non-null, non-user_query value.
    """
    # Search all values for <solution> tags (last match wins)
    solution = None
    for value in state.values():
        if value is None:
            continue
        text = str(value)
        found = extract_solution(text)
        if found != text.strip():  # tags were found
            solution = found

    if solution is not None:
        return solution

    # Fall back: last non-null, non-user_query value
    for key in reversed(list(state.keys())):
        if key == "user_query":
            continue
        value = state[key]
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def compute_metrics_by_level(results: list) -> dict:
    difficulty_stats: dict[int, dict] = {}

    for result in results:
        difficulty = int(result["difficulty"])
        is_correct = result["is_correct"]

        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {"total": 0, "correct": 0}

        difficulty_stats[difficulty]["total"] += 1
        if is_correct:
            difficulty_stats[difficulty]["correct"] += 1

    metrics_by_level = {}
    overall_total = 0
    overall_correct = 0

    for difficulty in sorted(difficulty_stats.keys()):
        stats = difficulty_stats[difficulty]
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0

        metrics_by_level[f"level_{difficulty}"] = {
            "total_tasks": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(accuracy, 2),
        }

        overall_total += stats["total"]
        overall_correct += stats["correct"]

    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    metrics_by_level["overall"] = {
        "total_tasks": overall_total,
        "correct": overall_correct,
        "accuracy": round(overall_accuracy, 2),
    }

    return metrics_by_level


def compute_token_summary(results: list) -> dict:
    total_meta_prompt = 0
    total_meta_completion = 0
    total_pipeline_prompt = 0
    total_pipeline_completion = 0

    for result in results:
        if "error" in result:
            continue
        tokens = result.get("tokens", {})
        total_meta_prompt += tokens.get("meta_prompt", 0)
        total_meta_completion += tokens.get("meta_completion", 0)
        total_pipeline_prompt += tokens.get("pipeline_prompt", 0)
        total_pipeline_completion += tokens.get("pipeline_completion", 0)

    return {
        "meta_agent": {
            "prompt_tokens": total_meta_prompt,
            "completion_tokens": total_meta_completion,
            "total_tokens": total_meta_prompt + total_meta_completion,
        },
        "pipeline": {
            "prompt_tokens": total_pipeline_prompt,
            "completion_tokens": total_pipeline_completion,
            "total_tokens": total_pipeline_prompt + total_pipeline_completion,
        },
        "grand_total": {
            "prompt_tokens": total_meta_prompt + total_pipeline_prompt,
            "completion_tokens": total_meta_completion + total_pipeline_completion,
            "total_tokens": (
                total_meta_prompt + total_meta_completion
                + total_pipeline_prompt + total_pipeline_completion
            ),
        },
    }


def print_score_by_level(metrics_by_level: dict) -> None:
    print("\n" + "=" * 50)
    print("GAIA Benchmark Results")
    print("=" * 50)

    for key in sorted(metrics_by_level.keys()):
        if key.startswith("level_"):
            level = key.split("_")[1]
            stats = metrics_by_level[key]
            print(f"Level {level}: {stats['accuracy']:.2f}%  ({stats['correct']}/{stats['total_tasks']})")

    overall = metrics_by_level["overall"]
    print(f"Overall:   {overall['accuracy']:.2f}%  ({overall['correct']}/{overall['total_tasks']})")
    print("=" * 50)


def print_token_summary(token_summary: dict) -> None:
    print("\n" + "=" * 50)
    print("Token Usage")
    print("=" * 50)
    meta = token_summary["meta_agent"]
    pipe = token_summary["pipeline"]
    grand = token_summary["grand_total"]
    print(f"Meta-agent:  {meta['total_tokens']:>10,}  (prompt: {meta['prompt_tokens']:,}, completion: {meta['completion_tokens']:,})")
    print(f"Pipeline:    {pipe['total_tokens']:>10,}  (prompt: {pipe['prompt_tokens']:,}, completion: {pipe['completion_tokens']:,})")
    print(f"Grand total: {grand['total_tokens']:>10,}  (prompt: {grand['prompt_tokens']:,}, completion: {grand['completion_tokens']:,})")
    print("=" * 50)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def process_task(
    task,
    gaia_benchmark: GaiaBenchmark,
    task_log_dir: Path,
) -> dict:
    """Process a single GAIA task using FEDOT.MAS MAW."""
    instruction = (
        "Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n"
        "For example: The answer to the question is <solution>42</solution>.\n\n"
    )

    query = instruction
    if task.file_path:
        query += f"File path: {task.file_path}\n"
    if task.file_name:
        query += f"File name: {task.file_name}\n"
    query += f"Question: {task.question}"

    maw = MAW(mcp_servers="all")
    state = await maw.run(query)

    answer = extract_answer_from_state(state)
    is_correct = gaia_benchmark.is_correct_answer(answer, task.ground_truth)

    pipeline_result = maw.last_result

    result = {
        "task_id": task.task_id,
        "question": task.question,
        "response": answer,
        "ground_truth": task.ground_truth,
        "difficulty": task.difficulty,
        "is_correct": is_correct,
        "session_state": {k: str(v) for k, v in state.items()},
        "tokens": {
            "meta_prompt": maw.meta_prompt_tokens,
            "meta_completion": maw.meta_completion_tokens,
            "pipeline_prompt": pipeline_result.total_prompt_tokens if pipeline_result else 0,
            "pipeline_completion": pipeline_result.total_completion_tokens if pipeline_result else 0,
            "total_prompt": maw.total_prompt_tokens,
            "total_completion": maw.total_completion_tokens,
        },
        "elapsed": maw.elapsed,
    }

    # Save per-task result
    task_log_dir.mkdir(parents=True, exist_ok=True)
    with open(task_log_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    leaderboard = {"task_id": task.task_id, "model_answer": answer}
    with open(task_log_dir / "leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

    return result


async def run_gaia(difficulty: str, split: str) -> Any:
    """Run GAIA benchmark using FEDOT.MAS MAW."""
    base_log_dir = Path(__file__).resolve().parent / "gaia_logs" / f"run_{RUN_ID}"
    base_log_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Logs will be saved to: {}", base_log_dir)
    _log.info("Loading GAIA benchmark (difficulty={}, split={})", difficulty, split)

    gaia = GaiaBenchmark({"difficulty": difficulty, "split": split})
    gaia.download()

    results = []

    for task in tqdm(gaia, desc="Processing GAIA tasks"):
        task_log_dir = base_log_dir / f"task_{task.task_id}"
        try:
            result = await process_task(task, gaia, task_log_dir)
            status = "CORRECT" if result["is_correct"] else "WRONG"
            _log.info(
                "[{}] task={} answer='{}' gt='{}'",
                status, task.task_id, result["response"][:60], task.ground_truth,
            )
        except Exception as e:
            _log.error("Failed task {} after all retries: {}", task.task_id, e)
            result = {
                "task_id": task.task_id,
                "question": task.question,
                "response": "",
                "ground_truth": task.ground_truth,
                "difficulty": task.difficulty,
                "is_correct": False,
                "error": str(e),
            }

        results.append(result)

    metrics_by_level = compute_metrics_by_level(results)
    token_summary = compute_token_summary(results)

    output_data = {
        "config": {
            "difficulty": difficulty,
            "split": split,
            "num_tasks": len(results),
        },
        "metrics": metrics_by_level,
        "token_summary": token_summary,
        "results": results,
    }

    with open(base_log_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print_score_by_level(metrics_by_level)
    print_token_summary(token_summary)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with FEDOT.MAS")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Dataset difficulty level (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation[:1]",
        help="Dataset split (default: validation[:1])",
    )
    args = parser.parse_args()

    asyncio.run(run_gaia(difficulty=args.difficulty, split=args.split))


if __name__ == "__main__":
    main()
