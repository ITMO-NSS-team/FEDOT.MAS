from __future__ import annotations

import asyncio

import uvicorn
from fedotmas import MAS
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.deployment.auto_generate_serve")

TASK = "Sort the list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]."


async def generate():
    mas = MAS(mcp_servers=["light-sandbox"])

    _log.info("Generating pipeline config for task …")
    config = await mas.generate_config(TASK)
    _log.info(
        "Config generated in {:.1f}s ({} agents)",
        mas.meta_elapsed,
        len(config.agents),
    )

    return mas.serve(
        config,
        name="agent",
        web=True,
        host="0.0.0.0",
        port=8000,
        auto_create_session=True,
    )


if __name__ == "__main__":
    app = asyncio.run(generate())
    _log.info("Starting API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
