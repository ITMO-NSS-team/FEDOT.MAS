import asyncio
import json

from fedotmas import MAS, AgentConfig, PipelineConfig, StepConfig


async def full_auto():
    mas = MAS(mcp_servers=[])
    state = await mas.run("Explain the difference between TCP and UDP in 3 sentences")
    print(json.dumps(state, indent=2, default=str))


async def two_step():
    mas = MAS(mcp_servers=[])

    # meta-agent generates the pipeline config.
    config = await mas.generate_config("Compare Python and Rust for CLI tools")
    print(config.model_dump_json(indent=2))

    # build the ADK agent tree and execute.
    state = await mas.build_and_run(config, "Compare Python and Rust for CLI tools")
    print(json.dumps(state, indent=2, default=str))


async def handcrafted():
    config = PipelineConfig(
        agents=[
            AgentConfig(
                name="researcher",
                instruction="Research the topic: {user_query}. Provide key facts.",
                output_key="research",
            ),
            AgentConfig(
                name="writer",
                instruction=(
                    "Write a concise summary based on the research:\n\n{research}"
                ),
                output_key="summary",
            ),
        ],
        pipeline=StepConfig(
            type="sequential",
            children=[
                StepConfig(type="agent", agent_name="researcher"),
                StepConfig(type="agent", agent_name="writer"),
            ],
        ),
    )

    mas = MAS(mcp_servers=[])
    state = await mas.build_and_run(config, "What is WebAssembly?")
    print(state.get("summary", "(no summary produced)"))


async def parallel_analysis():
    config = PipelineConfig(
        agents=[
            AgentConfig(
                name="pros_analyst",
                instruction="List 3 key advantages of: {user_query}",
                output_key="pros",
            ),
            AgentConfig(
                name="cons_analyst",
                instruction="List 3 key disadvantages of: {user_query}",
                output_key="cons",
            ),
            AgentConfig(
                name="synthesizer",
                instruction=(
                    "Given these pros:\n{pros}\n\n"
                    "And these cons:\n{cons}\n\n"
                    "Write a balanced 2-paragraph verdict."
                ),
                output_key="verdict",
            ),
        ],
        pipeline=StepConfig(
            type="sequential",
            children=[
                StepConfig(
                    type="parallel",
                    children=[
                        StepConfig(type="agent", agent_name="pros_analyst"),
                        StepConfig(type="agent", agent_name="cons_analyst"),
                    ],
                ),
                StepConfig(type="agent", agent_name="synthesizer"),
            ],
        ),
    )

    mas = MAS(mcp_servers=[])
    state = await mas.build_and_run(config, "Microservices architecture")
    print(state.get("verdict", "(no verdict produced)"))


async def loop_with_critic():
    config = PipelineConfig(
        agents=[
            AgentConfig(
                name="writer",
                instruction=(
                    "Write a short poem about: {user_query}.\n"
                    "If there is previous feedback, incorporate it: {feedback}"
                ),
                output_key="draft",
            ),
            AgentConfig(
                name="critic",
                instruction=(
                    "Review this poem:\n{draft}\n\n"
                    "If the poem is good, call the exit_loop tool.\n"
                    "Otherwise, provide specific feedback for improvement."
                ),
                output_key="feedback",
            ),
        ],
        pipeline=StepConfig(
            type="loop",
            max_iterations=3,
            children=[
                StepConfig(type="agent", agent_name="writer"),
                StepConfig(type="agent", agent_name="critic"),
            ],
        ),
    )

    mas = MAS(mcp_servers=[])
    state = await mas.build_and_run(config, "the ocean at sunset")
    print(state.get("draft", "(no draft produced)"))


if __name__ == "__main__":
    # Pick one:
    asyncio.run(handcrafted())  # yes
    # asyncio.run(full_auto())  # yes
    # asyncio.run(two_step())  # yes
    # asyncio.run(parallel_analysis())  # yes
    # asyncio.run(loop_with_critic())  # yes
