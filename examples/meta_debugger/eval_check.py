import asyncio

from fedotmas import MAW
from fedotmas.control import Controller
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="calculator",
            instruction="Верни в ответе только число 2.",
            output_key="number",
        ),
    ],
    pipeline=MAWStepConfig(agent_name="calculator"),
)


async def main():
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Верни число 5",
        config=CONFIG,
        max_retries=2,
        error_hint="Агент должен возвращать число 5, а не любое другое",
    )
    print(f"Status: {result.status}")
    print(f"Number: {result.state.get('number')}")


if __name__ == "__main__":
    asyncio.run(main())
