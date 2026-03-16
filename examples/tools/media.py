import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.media")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="analyst",
                instruction=(
                    "{user_query}\n" "Analyze the media and report your findings."
                ),
                model=MODEL,
                output_key="result",
                tools=["media"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="analyst"),
    )

    maw = MAW(mcp_servers=["media"])
    state = await maw.build_and_run(
        config,
        "Describe the image at https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png",
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
