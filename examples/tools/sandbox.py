import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.sandbox")

MODEL = "openai/gpt-4o-mini"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="coder",
                instruction=(
                    "Solve the task using Python code: {user_query}\n"
                    "Return the result."
                ),
                model=MODEL,
                output_key="result",
                tools=["sandbox-light"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="coder"),
    )

    maw = MAW(mcp_servers=["sandbox-light"])
    state = await maw.build_and_run(
        config,
        "Estimate π by computing the Leibniz series "
        "(π/4 = 1 - 1/3 + 1/5 - 1/7 + ...) with 1,000,000 terms. "
        "Return the estimate rounded to 4 decimal places.",
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
