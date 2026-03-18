import asyncio

from fedotmas import MAW
from fedotmas.control import Controller


async def auto_recovery():
    maw = MAW(mcp_servers="all")
    ctrl = Controller(maw)
    state = await ctrl.run_with_recovery(
        "Проанализируй продажи за Q1",
        max_retries=2,
    )
    print(state["result"])


async def interactive_inspection():
    maw = MAW()
    ctrl = Controller(maw)

    config = await maw.generate_config("Сравни Python и Rust для CLI")
    print(config)

    async with ctrl.run_interactive(config, "Сравни Python и Rust для CLI") as run:
        await run.wait_until("researcher")
        print(run.state)

        await run.continue_()


async def interactive_with_restart():
    maw = MAW()
    ctrl = Controller(maw)

    config = await maw.generate_config("Напиши отчёт по рынку")
    print(config)

    async with ctrl.run_interactive(config, "Напиши отчёт по рынку") as run:
        await run.wait_until("data_prep")
        print(run.state)

        await run.restart_from("data_prep")


if __name__ == "__main__":
    asyncio.run(auto_recovery())
