from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fedotmas.core.runner import run_pipeline
from fedotmas.maw.builder import build
from fedotmas.maw.models import MAWConfig
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def render_request(llm_request: LlmRequest) -> tuple[str, dict[str, Any]]:
    system_instruction = llm_request.config.system_instruction or ""
    if not isinstance(system_instruction, str):
        system_instruction = str(system_instruction)

    conversation: list[dict[str, str]] = []
    for content in llm_request.contents:
        text = "\n".join(
            part.text for part in content.parts or [] if part.text is not None
        )
        if text:
            conversation.append({"role": content.role or "unknown", "text": text})

    rendered = "\n\n".join(
        f"[{item['role'].upper()}]\n{item['text']}" for item in conversation
    )
    prompt = (
        f"[SYSTEM INSTRUCTION]\n{system_instruction}\n\n"
        f"[CONVERSATION]\n{rendered}\n\n"
        "Return only the response requested for this worker."
    )
    return prompt, {
        "system_instruction": system_instruction,
        "conversation": conversation,
    }


class HostBridgeLlm(BaseLlm):
    agent_name: str
    run_dir: Path
    timeout_seconds: float = 900.0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if stream:
            raise ValueError("Host bridge does not support streaming")

        prompt, details = render_request(llm_request)
        request_id = f"{self.agent_name}-{uuid.uuid4().hex}"
        requests_dir = self.run_dir / "requests"
        responses_dir = self.run_dir / "responses"
        requests_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / f"{request_id}.json"
        response_path = responses_dir / f"{request_id}.json"
        request = {
            "request_id": request_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "prompt": prompt,
            "response_path": str(response_path.resolve()),
            **details,
        }
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"FEDOT_BRIDGE_REQUEST {request_path.resolve()}", flush=True)

        started = time.monotonic()
        while not response_path.exists():
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError(f"Timed out waiting for {response_path}")
            await asyncio.sleep(0.25)

        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") != request_id:
            raise ValueError(f"Response id mismatch in {response_path}")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        text = str(response.get("text", "")).strip()
        yield LlmResponse(
            model_version=self.model,
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=text)]
            ),
            finish_reason=types.FinishReason.STOP,
        )


def walk_llm_agents(root: BaseAgent):
    if isinstance(root, LlmAgent):
        yield root
    for child in root.sub_agents or []:
        yield from walk_llm_agents(child)


def load_task(args: argparse.Namespace) -> str:
    if args.task is not None:
        return args.task
    return args.task_file.read_text(encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    config = MAWConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    root = build(config)
    run_dir = args.queue_dir / args.run_id
    for agent in walk_llm_agents(root):
        configured = next(item for item in config.agents if item.name == agent.name)
        if not configured.model:
            raise ValueError(f"Bridge mode requires an explicit model for {agent.name}")
        agent.model = HostBridgeLlm(
            model=configured.model,
            agent_name=agent.name,
            run_dir=run_dir,
            timeout_seconds=args.request_timeout,
        )

    task = load_task(args)
    result = await run_pipeline(root, task, timeout=args.pipeline_timeout)
    payload = {
        "run_id": args.run_id,
        "state": {key: str(value) for key, value in result.state.items()},
        "elapsed_seconds": result.elapsed,
        "queue_dir": str(run_dir.resolve()),
        "config": config.model_dump(),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"FEDOT_BRIDGE_RESULT {args.result.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a FEDOT.MAS MAW through host-native subagents"
    )
    parser.add_argument("--config", type=Path, required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task")
    task_group.add_argument("--task-file", type=Path)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--pipeline-timeout", type=float, default=3600.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
