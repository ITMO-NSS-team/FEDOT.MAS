from __future__ import annotations

import asyncio

import uvicorn
from fedotmas import MAS, serve
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.deployment.auto_generate_serve")

TASK = "Calculate factorial of 6."


async def main() -> None:
    mas = MAS()

    _log.info("Generating pipeline config for task …")
    config = await mas.generate_config(TASK)
    _log.info(
        "Config generated in {:.1f}s ({} agents)",
        mas.meta_elapsed,
        len(config.agents),
    )

    agent = mas.build(config)

    app = serve(
        {"agent": agent},
        web=True,
        host="0.0.0.0",
        port=8000,
    )

    _log.info("Starting API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
