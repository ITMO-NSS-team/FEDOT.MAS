import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.sequential_thinking")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="thinker",
                instruction=(
                    "Use the sequential_thinking tool to reason through "
                    "the following problem step by step:\n\n"
                    "{user_query}\n\n"
                    "Break it into small logical steps, revise if needed, "
                    "and provide a final answer."
                ),
                model=MODEL,
                output_key="result",
                tools=["sequential-thinking"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="thinker"),
    )

    maw = MAW(mcp_servers=["sequential-thinking"])
    state = await maw.build_and_run(
        config,
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
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
