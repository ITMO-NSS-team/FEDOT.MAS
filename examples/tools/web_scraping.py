import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.web_scraping")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="scraper",
                instruction=(
                    "Navigate to {user_query} and extract the page content as markdown. "
                    "Then list all links found on the page."
                ),
                model=MODEL,
                output_key="result",
                tools=["web-scraping"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="scraper"),
    )

    maw = MAW(mcp_servers=["web-scraping"])
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
