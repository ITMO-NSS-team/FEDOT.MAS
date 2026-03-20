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


async def interactive_pause():
    maw = MAW()
    ctrl = Controller(maw)

    async with ctrl.run_interactive(
        RESEARCH_CONFIG, "Сравни Python и Rust для CLI"
    ) as run:
        await run.wait_until("writer")
        print(run.state)

        result = await run.continue_()
        print(result.state)


async def interactive_multi_pause():
    maw = MAW()
    ctrl = Controller(maw)

    async with ctrl.run_interactive(
        RESEARCH_CONFIG, "Анализ рынка облачных сервисов"
    ) as run:
        await run.wait_until("writer")
        print(run.state)

        await run.wait_until("reviewer")
        print(run.state)

        result = await run.continue_()
        print(result.state)


if __name__ == "__main__":
    # asyncio.run(interactive_multi_pause())
    asyncio.run(interactive_pause())
