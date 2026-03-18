import asyncio

from fedotmas import MAW
from fedotmas.control import Controller, Strategy
from fedotmas.maw.models import MAWAgentConfig


async def manual_fix_after_error():
    maw = MAW(mcp_servers="all")
    ctrl = Controller(maw)

    run = await ctrl.run("Проанализируй продажи за Q1")

    if run.status == "error":
        print(f"Упал агент: {run.error.agent_name}")
        print(f"Ошибка: {run.error.message}")
        print(f"Успешные: {[cp.agent_name for cp in run.checkpoints]}")

        new_config = run.config.replace_agent(
            "forecaster",
            MAWAgentConfig(
                name="forecaster_v2",
                instruction="Сделай прогноз на основе: {analysis}",
                output_key="forecast",
            ),
        )

        run = await ctrl.resume(new_config)

    print(run.result)


async def structural_change():
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Напиши отчёт по рынку")

    if run.status == "error":
        new_config = run.config.replace_agent(
            "writer",
            [
                MAWAgentConfig(
                    name="validator",
                    instruction="Проверь данные: {research}",
                    output_key="validated_data",
                ),
                MAWAgentConfig(
                    name="writer_v2",
                    instruction="Напиши отчёт на основе: {validated_data}",
                    output_key="report",
                ),
            ],
        )

        run = await ctrl.resume(new_config)

    print(run.result)


async def insert_agent():
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Анализ конкурентов")

    new_config = run.config.insert_after(
        "researcher",
        MAWAgentConfig(
            name="fact_checker",
            instruction="Проверь факты: {research}",
            output_key="verified_research",
        ),
    )

    run = await ctrl.resume(new_config)
    print(run.result)


async def remove_agent():
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Подготовь презентацию")

    new_config = run.config.remove_agent("unnecessary_step")
    run = await ctrl.resume(new_config)
    print(run.result)


async def explicit_strategy():
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Задача")

    new_config = run.config.replace_agent(
        "agent_a",
        MAWAgentConfig(name="agent_a", instruction="Новый промпт", output_key="a"),
    )

    run = await ctrl.resume(new_config, strategy=Strategy.RETRY_FAILED)

    run = await ctrl.resume(new_config, strategy=Strategy.RESTART_AFTER)

    run = await ctrl.resume(new_config, strategy=Strategy.RESTART_ALL)


if __name__ == "__main__":
    asyncio.run(manual_fix_after_error())
