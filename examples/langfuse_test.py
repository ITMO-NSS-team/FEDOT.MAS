"""Quick test of LangfusePlugin with a handcrafted MAW pipeline.

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


async def main():
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
        worker_models=[
            ModelConfig(
                model="google/gemini-2.0-flash-001",
                api_base="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENAI_API_KEY"],
            )
        ],
        plugins=[
            LoggingPlugin(),
            LangfusePlugin(trace_name="langfuse_integration_test"),
        ],
    )

    print("Running pipeline with Langfuse tracing...")
    state = await maw.build_and_run(config, "What is WebAssembly?")

    # Manually finalize trace since we're using build_and_run (not run)
    maw._finalize_langfuse()

    print("\n--- Research ---")
    print(state.get("research", "(none)"))
    print("\n--- Summary ---")
    print(state.get("summary", "(none)"))
    print("\n--- Token usage ---")
    print(f"  Prompt tokens:     {maw.total_prompt_tokens}")
    print(f"  Completion tokens: {maw.total_completion_tokens}")
    print(f"  Elapsed:           {maw.elapsed:.1f}s")
    print("\nCheck your Langfuse dashboard for the trace!")


if __name__ == "__main__":
    asyncio.run(main())
