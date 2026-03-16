import asyncio
from pathlib import Path

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_ROOT = Path(__file__).resolve().parents[2]

_log = get_logger("fedotmas.examples.tools.document")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="reader",
                instruction=(
                    "{user_query}\n" "Read the document and summarize key findings."
                ),
                model=MODEL,
                output_key="result",
                tools=["document"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="reader"),
    )

    maw = MAW(mcp_servers=["document"])
    readme = _ROOT / "README.md"
    state = await maw.build_and_run(config, f"Read and summarize {readme}")

    _log.info("Result: {}", str(state.get("result", ""))[:500])
    _log.info(
        "Tokens: prompt={} completion={} elapsed={:.1f}s",
        maw.total_prompt_tokens,
        maw.total_completion_tokens,
        maw.elapsed,
    )


if __name__ == "__main__":
    asyncio.run(main())
