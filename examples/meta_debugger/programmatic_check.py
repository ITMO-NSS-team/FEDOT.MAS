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


def check_number(state: dict) -> str | None:
    """Return an error message if the output is wrong, None if OK."""
    output = str(state.get("number", ""))
    if "5" not in output:
        return f"Агент вернул {output!r}, но должен возвращать число 5. Исправь instruction."
    return None


async def main():
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Верни число 5",
        config=CONFIG,
        max_retries=2,
        checks={"calculator": check_number},
    )
    print(f"Status: {result.status}")
    print(f"Number: {result.state.get('number')}")


if __name__ == "__main__":
    asyncio.run(main())
