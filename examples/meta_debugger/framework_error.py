import asyncio

from fedotmas import MAW
from fedotmas.control import Controller
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction=(
                "Исследуй тему: {user_query}. "
                "Ответ ОБЯЗАТЕЛЬНО оформи как JSON объект с ключами 'title' и 'body'."
            ),
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            instruction="Напиши краткий отчёт на основе: {research}",
            output_key="report",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="writer"),
        ],
    ),
)


def check_research(state: dict) -> str | None:
    """Research must be plain text, not JSON."""
    research = str(state.get("research", ""))
    if "{" in research and "}" in research:
        return (
            "Агент вернул JSON вместо обычного текста. "
            "Убери из instruction требование формата JSON, "
            "пусть агент отвечает обычным текстом."
        )
    return None


async def main():
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Проанализируй рынок электромобилей",
        config=CONFIG,
        max_retries=2,
        checks={"researcher": check_research},
    )
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Result keys: {list(result.state.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
