import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.browser")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="browser",
                instruction=(
                    "{user_query}\n"
                    "Use the browser tools to navigate to the page, "
                    "extract content as markdown, and summarize the findings."
                ),
                model=MODEL,
                output_key="result",
                tools=["browser"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="browser"),
    )

    maw = MAW(mcp_servers=["browser"])
    state = await maw.build_and_run(
        config, "Go to https://lightpanda.io and summarize what the product does"
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
