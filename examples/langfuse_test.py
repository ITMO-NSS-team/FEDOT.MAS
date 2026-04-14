"""Test LangfusePlugin with both handcrafted and auto-generated pipelines.

Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
and OPENROUTER_API_KEY in .env (or environment).
"""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

# Map OpenRouter key to the env vars fedotmas expects.
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

from fedotmas import MAW, ModelConfig
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.plugins import LangfusePlugin, LoggingPlugin

WORKER_MODEL = ModelConfig(
    model="google/gemini-2.0-flash-001",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)


async def handcrafted():
    """Run a handcrafted pipeline — traces pipeline execution only."""
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="researcher",
                instruction="Research the topic: {user_query}. Provide 3 key facts.",
                output_key="research",
                model="google/gemini-2.0-flash-001",
            ),
            MAWAgentConfig(
                name="writer",
                instruction=(
                    "Write a concise 2-paragraph summary based on:\n\n{research}"
                ),
                output_key="summary",
                model="google/gemini-2.0-flash-001",
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(type="agent", agent_name="researcher"),
                MAWStepConfig(type="agent", agent_name="writer"),
            ],
        ),
    )

    maw = MAW(
        worker_models=[WORKER_MODEL],
        plugins=[
            LoggingPlugin(),
            LangfusePlugin(trace_name="langfuse_test:handcrafted"),
        ],
    )

    print("=== Handcrafted pipeline (no meta-agent) ===")
    state = await maw.build_and_run(config, "What is WebAssembly?")
    maw._finalize_langfuse()

    print("\n--- Summary ---")
    print(state.get("summary", "(none)"))
    print(f"\nTokens: {maw.total_prompt_tokens} in / {maw.total_completion_tokens} out")


async def full_auto():
    """Run a full-auto pipeline — traces meta-agent generation + pipeline execution."""
    maw = MAW(
        meta_model=WORKER_MODEL,
        worker_models=[WORKER_MODEL],
        plugins=[
            LoggingPlugin(),
            LangfusePlugin(trace_name="langfuse_test:full_auto"),
        ],
    )

    print("\n=== Full-auto pipeline (with meta-agent) ===")
    state = await maw.run("Explain the difference between TCP and UDP in 3 sentences")

    print("\nResult keys:", list(state.keys()))
    print(f"Meta tokens: {maw.meta_prompt_tokens} in / {maw.meta_completion_tokens} out")
    print(f"Total tokens: {maw.total_prompt_tokens} in / {maw.total_completion_tokens} out")
    print(f"Elapsed: {maw.elapsed:.1f}s")


if __name__ == "__main__":
    # Run handcrafted first (pipeline only), then full-auto (meta-agent + pipeline)
    asyncio.run(handcrafted())
    asyncio.run(full_auto())
