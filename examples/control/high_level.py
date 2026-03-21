import asyncio

from fedotmas import MAW
from fedotmas.control import Controller
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

RESEARCH_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction="Исследуй тему: {user_query}",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            instruction="Напиши отчёт на основе: {research}",
            output_key="report",
        ),
        MAWAgentConfig(
            name="reviewer",
            instruction="Оцени и улучши отчёт: {report}",
            output_key="review",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="writer"),
            MAWStepConfig(agent_name="reviewer"),
        ],
    ),
)


async def interactive_break():
    maw = MAW()
    ctrl = Controller(maw)

    async with ctrl.iter("Сравни Python и Rust для CLI", RESEARCH_CONFIG) as pipeline:
        async for step in pipeline:
            print(f"Step: {step.name}, state: {step.state}")
            if step.name == "writer":
                break
        result = await pipeline.finish()
        print(result.state)


async def interactive_debug():
    maw = MAW()
    ctrl = Controller(maw)

    async with ctrl.iter("Анализ рынка облачных сервисов", RESEARCH_CONFIG) as pipeline:
        async for step in pipeline:
            print(f"Step: {step.name}, state: {step.state}")
    print(pipeline.result)


async def auto_generated():
    """Config is generated automatically by MAW from the task description."""
    maw = MAW()
    ctrl = Controller(maw)

    async with ctrl.iter("Сравни Python и Rust для CLI") as pipeline:
        async for step in pipeline:
            if step.agent:
                print(f"Step {step.index}: {step.name}")
                print(f"  model: {step.agent.model}")
                print(f"  instruction: {step.agent.instruction}")
                print(f"  output_key: {step.agent.output_key}")
                print(f"  tools: {step.agent.tools}")
            else:
                # composite step (ParallelAgent, LoopAgent)
                print(f"Step {step.index}: {step.name} (composite)")
    print(pipeline.result)


async def auto_recovery():
    """Run with regex-based error detection (default)."""
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Проанализируй рынок электромобилей",
        max_retries=2,
    )
    print(f"Status: {result.status}")
    if result.status == "success":
        print(f"Result: {result.state}")
    else:
        print(f"Error: {result.error}")


async def auto_recovery_llm():
    """Run with LLM-based error classification."""
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Проанализируй рынок электромобилей",
        max_retries=2,
        llm_error_detection=True,
        error_hint="agent produces empty output or hallucinates tool names",
    )
    print(f"Status: {result.status}")
    if result.status == "success":
        print(f"Result: {result.state}")
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    # asyncio.run(interactive_debug())
    # asyncio.run(interactive_break())
    # asyncio.run(auto_generated())
    asyncio.run(auto_recovery())
