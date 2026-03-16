from __future__ import annotations

import os

import uvicorn

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.deployment.maw.handcrafted_serve")

config = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction=(
                "Research the topic provided by the user. "
                "Collect key facts and relevant data points. "
                "Store your findings in {research}."
            ),
            output_key="research",
            model=os.getenv("FEDOTMAS_DEFAULT_MODEL", "default_model"),
        ),
        MAWAgentConfig(
            name="writer",
            instruction=(
                "Using the research in {research?}, write a well-structured "
                "article. Store the final article in {article}."
            ),
            output_key="article",
            model=os.getenv("FEDOTMAS_DEFAULT_MODEL", "default_model"),
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="researcher"),
            MAWStepConfig(type="agent", agent_name="writer"),
        ],
    ),
)


def main() -> None:
    maw = MAW()

    app = maw.serve(
        config,
        name="research_writer",
        session_service_uri="sqlite:///sessions.db",
        web=True,
        host="0.0.0.0",
        port=8000,
        auto_create_session=True,
    )

    _log.info("Starting API server on http://0.0.0.0:8000")
    _log.info("Sessions persisted to sessions.db")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
