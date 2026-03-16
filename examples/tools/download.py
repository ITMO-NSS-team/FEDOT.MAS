import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.download")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="fetcher",
                instruction=(
                    "{user_query}\n"
                    "Download the files and report file names and sizes."
                ),
                model=MODEL,
                output_key="result",
                tools=["download"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="fetcher"),
    )

    maw = MAW(mcp_servers=["download"])
    state = await maw.build_and_run(
        config,
        "Download the Python logo from https://www.python.org/static/community_logos/python-logo.png",
    )

    _log.info("Result: {}", str(state.get("result", ""))[:500])
    _log.info(
        "Tokens: prompt={} completion={} elapsed={:.1f}s",
        maw.total_prompt_tokens,
        maw.total_completion_tokens,
        maw.elapsed,
    )


if __name__ == "__main__":
    asyncio.run(main())
