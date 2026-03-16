import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.web")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="researcher",
                instruction=(
                    "Search the web for: {user_query}\n"
                    "Return a concise summary of the top results."
                ),
                model=MODEL,
                output_key="result",
                tools=["websearch-searxng"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="researcher"),
    )

    maw = MAW(mcp_servers=["websearch-searxng"])
    state = await maw.build_and_run(
        config, "Latest breakthroughs in multi-agent systems 2026"
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
