from __future__ import annotations

import asyncio

import uvicorn

from fedotmas import MAS
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.deployment.mas.auto_generate_serve")

TASK = "Handle customer support for an e-commerce platform"


async def generate():
    mas = MAS(mcp_servers=["light-sandbox"])

    _log.info("Generating routing config for task ...")
    config = await mas.generate_config(TASK)
    _log.info(
        "Config generated in {:.1f}s (coordinator={}, {} workers)",
        mas.meta_elapsed,
        config.coordinator.name,
        len(config.workers),
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
    _log.info("Starting MAS API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
