import asyncio

from fedotmas import MAW
from fedotmas.control import Controller, Strategy
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

SALES_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="analyst",
            instruction="Проанализируй данные о продажах: {user_query}",
            output_key="analysis",
        ),
        MAWAgentConfig(
            name="forecaster",
            instruction="Сделай прогноз на основе: {analysis}",
            output_key="forecast",
        ),
        MAWAgentConfig(
            name="reporter",
            instruction="Сформируй отчёт: анализ={analysis}, прогноз={forecast}",
            output_key="report",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="analyst"),
            MAWStepConfig(agent_name="forecaster"),
            MAWStepConfig(agent_name="reporter"),
        ],
    ),
)


async def replace_agent():
    """1:1 замена агента (топология не меняется)."""
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Проанализируй продажи за Q1", config=SALES_CONFIG)

    if run.status == "error" and run.error:
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


async def replace_with_parallel():
    """Замена одного агента на параллельный бранч (replace_step)."""
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Проанализируй продажи за Q1", config=SALES_CONFIG)

    if run.status == "error":
        forecaster_stat = MAWAgentConfig(
            name="forecaster_stat",
            instruction="Статистический прогноз: {analysis}",
            output_key="forecast_stat",
        )
        forecaster_ml = MAWAgentConfig(
            name="forecaster_ml",
            instruction="ML прогноз: {analysis}",
            output_key="forecast_ml",
        )

        new_config = run.config.replace_step(
            "forecaster",
            step=MAWStepConfig(
                type="parallel",
                children=[
                    MAWStepConfig(agent_name="forecaster_stat"),
                    MAWStepConfig(agent_name="forecaster_ml"),
                ],
            ),
            agents=[forecaster_stat, forecaster_ml],
        )

        run = await ctrl.resume(new_config)

    print(run.result)


REPORT_CONFIG = MAWConfig(
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
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="writer"),
        ],
    ),
)


async def replace_with_loop():
    """Замена одного агента на цикл (replace_step)."""
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Напиши отчёт по рынку", config=REPORT_CONFIG)

    if run.status == "error":
        drafter = MAWAgentConfig(
            name="drafter",
            instruction="Напиши черновик на основе: {research}",
            output_key="draft",
        )
        reviewer = MAWAgentConfig(
            name="reviewer",
            instruction="Оцени черновик: {draft}",
            output_key="report",
        )

        new_config = run.config.replace_step(
            "writer",
            step=MAWStepConfig(
                type="loop",
                children=[
                    MAWStepConfig(agent_name="drafter"),
                    MAWStepConfig(agent_name="reviewer"),
                ],
                max_iterations=3,
            ),
            agents=[drafter, reviewer],
        )

        run = await ctrl.resume(new_config)

    print(run.result)


COMPETITORS_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction="Исследуй конкурентов: {user_query}",
            output_key="research",
        ),
        MAWAgentConfig(
            name="summarizer",
            instruction="Подведи итоги: {research}",
            output_key="summary",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="summarizer"),
        ],
    ),
)


async def insert_agent():
    """Вставка агента после существующего."""
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Анализ конкурентов", config=COMPETITORS_CONFIG)

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


PRESENTATION_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction="Исследуй тему: {user_query}",
            output_key="research",
        ),
        MAWAgentConfig(
            name="formatter",
            instruction="Отформатируй: {research}",
            output_key="formatted",
        ),
        MAWAgentConfig(
            name="presenter",
            instruction="Подготовь слайды: {formatted}",
            output_key="presentation",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="formatter"),
            MAWStepConfig(agent_name="presenter"),
        ],
    ),
)


async def remove_agent():
    """Удаление агента из pipeline."""
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Подготовь презентацию", config=PRESENTATION_CONFIG)

    new_config = run.config.remove_agent("formatter")
    run = await ctrl.resume(new_config)
    print(run.result)


async def explicit_strategy():
    """Явный выбор стратегии при resume."""
    maw = MAW()
    ctrl = Controller(maw)

    run = await ctrl.run("Проанализируй продажи за Q1", config=SALES_CONFIG)

    new_config = run.config.replace_agent(
        "forecaster",
        MAWAgentConfig(
            name="forecaster",
            instruction="Новый промпт для прогноза: {analysis}",
            output_key="forecast",
        ),
    )

    run = await ctrl.resume(new_config, strategy=Strategy.RETRY_FAILED)

    run = await ctrl.resume(new_config, strategy=Strategy.RESTART_AFTER)

    run = await ctrl.resume(new_config, strategy=Strategy.RESTART_ALL)


if __name__ == "__main__":
    asyncio.run(remove_agent())
