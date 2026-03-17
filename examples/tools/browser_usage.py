import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.browser_usage")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="browser",
                instruction=(
                    "Go to {user_query} and extract the main heading and "
                    "first paragraph of content from the page."
                ),
                model=MODEL,
                output_key="result",
                tools=["browser-usage"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="browser"),
    )

    maw = MAW(mcp_servers=["browser-usage"])
    state = await maw.build_and_run(
        config, "https://en.wikipedia.org/wiki/Multi-agent_system"
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
