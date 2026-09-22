from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from fedotmas import MAS, MASConfig
from fedotmas._settings import ModelConfig

HERE = Path(__file__).resolve().parent
PREDICTOR_SERVER = "rubber-recipe-predictor"


def load_inputs(model: str) -> tuple[MASConfig, str]:
    raw_config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    raw_config["coordinator"]["model"] = model
    for worker in raw_config["workers"]:
        worker["model"] = model
    task = (HERE / "task.md").read_text(encoding="utf-8")
    return MASConfig.model_validate(raw_config), task


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rubber-recipe MAS")
    parser.add_argument(
        "--model",
        default=os.getenv("FEDOTMAS_DEFAULT_MODEL", "host/gpt-5.6-terra"),
        help="Provider-prefixed model name",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=HERE / "provider_result.json")
    args = parser.parse_args()

    if "/" not in args.model:
        parser.error("--model must include a provider prefix")
    if args.model.startswith("host/"):
        parser.error(
            "host/gpt-5.6-terra uses the Codex subscription, not an API endpoint; "
            "open the recorded run with serve_gui.py or pass a provider-backed model"
        )

    config, task = load_inputs(args.model)
    endpoint = ModelConfig(
        model=args.model,
        api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    mas = MAS(worker_models=[endpoint], mcp_servers=[PREDICTOR_SERVER])
    try:
        state = await mas.build_and_run(config, task, timeout=args.timeout)
    except Exception as exc:
        payload = {
            "status": "error",
            "config": config.model_dump(mode="json"),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    else:
        payload = {
            "status": "ok",
            "config": config.model_dump(mode="json"),
            "state": state,
            "metrics": {
                "prompt_tokens": mas.total_prompt_tokens,
                "completion_tokens": mas.total_completion_tokens,
                "elapsed_seconds": mas.elapsed,
                "truncated_agents": (
                    mas.last_result.truncated_agents if mas.last_result else []
                ),
            },
        }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
