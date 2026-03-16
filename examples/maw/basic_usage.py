import asyncio
import json

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.maw.basic_usage")


async def full_auto():
    maw = MAW(mcp_servers="all")
    state = await maw.run("Explain the difference between TCP and UDP in 3 sentences")
    _log.info("Result: {}", json.dumps(state, indent=2, default=str))


async def two_step():
    maw = MAW()

    config = await maw.generate_config("Compare Python and Rust for CLI tools")
    _log.info("Config: {}", config.model_dump_json(indent=2))

    state = await maw.build_and_run(config, "Compare Python and Rust for CLI tools")
    _log.info("Result: {}", json.dumps(state, indent=2, default=str))


async def handcrafted():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="researcher",
                instruction="Research the topic: {user_query}. Provide key facts.",
                output_key="research",
            ),
            MAWAgentConfig(
                name="writer",
                instruction=(
                    "Write a concise summary based on the research:\n\n{research}"
                ),
                output_key="summary",
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

    maw = MAW()
    state = await maw.build_and_run(config, "What is WebAssembly?")
    _log.info("Summary: {}", state.get("summary", "(no summary produced)"))


async def parallel_analysis():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="pros_analyst",
                instruction="List 3 key advantages of: {user_query}",
                output_key="pros",
            ),
            MAWAgentConfig(
                name="cons_analyst",
                instruction="List 3 key disadvantages of: {user_query}",
                output_key="cons",
            ),
            MAWAgentConfig(
                name="synthesizer",
                instruction=(
                    "Given these pros:\n{pros}\n\n"
                    "And these cons:\n{cons}\n\n"
                    "Write a balanced 2-paragraph verdict."
                ),
                output_key="verdict",
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(
                    type="parallel",
                    children=[
                        MAWStepConfig(type="agent", agent_name="pros_analyst"),
                        MAWStepConfig(type="agent", agent_name="cons_analyst"),
                    ],
                ),
                MAWStepConfig(type="agent", agent_name="synthesizer"),
            ],
        ),
    )

    maw = MAW()
    state = await maw.build_and_run(config, "Microservices architecture")
    _log.info("Verdict: {}", state.get("verdict", "(no verdict produced)"))


async def loop_with_critic():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="writer",
                instruction=(
                    "Write a short poem about: {user_query}.\n"
                    "If there is previous feedback, incorporate it: {feedback}"
                ),
                output_key="draft",
            ),
            MAWAgentConfig(
                name="critic",
                instruction=(
                    "Review this poem:\n{draft}\n\n"
                    "If the poem is good, call the exit_loop tool.\n"
                    "Otherwise, provide specific feedback for improvement."
                ),
                output_key="feedback",
            ),
        ],
        pipeline=MAWStepConfig(
            type="loop",
            max_iterations=3,
            children=[
                MAWStepConfig(type="agent", agent_name="writer"),
                MAWStepConfig(type="agent", agent_name="critic"),
            ],
        ),
    )

    maw = MAW()
    state = await maw.build_and_run(config, "the ocean at sunset")
    _log.info("Draft: {}", state.get("draft", "(no draft produced)"))


if __name__ == "__main__":
    # Pick one:
    # asyncio.run(handcrafted())  # yes
    asyncio.run(full_auto())  # yes
    # asyncio.run(two_step())  # yes
    # asyncio.run(parallel_analysis())  # yes
    # asyncio.run(loop_with_critic())  # yes
