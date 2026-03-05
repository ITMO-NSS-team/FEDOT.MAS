from __future__ import annotations

import uvicorn

from fedotmas import MAS, AgentConfig, PipelineConfig, StepConfig, serve
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.deployment.handcrafted_serve")

config = PipelineConfig(
    agents=[
        AgentConfig(
            name="researcher",
            instruction=(
                "Research the topic provided by the user. "
                "Collect key facts and relevant data points. "
                "Store your findings in {research}."
            ),
            output_key="research",
        ),
        AgentConfig(
            name="writer",
            instruction=(
                "Using the research in {research?}, write a well-structured "
                "article. Store the final article in {article}."
            ),
            output_key="article",
        ),
    ],
    pipeline=StepConfig(
        type="sequential",
        children=[
            StepConfig(type="agent", agent_name="researcher"),
            StepConfig(type="agent", agent_name="writer"),
        ],
    ),
)


def main() -> None:
    mas = MAS()
    agent = mas.build(config)

    app = serve(
        {"research_writer": agent},
        session_service_uri="sqlite:///sessions.db",
        web=True,
        host="0.0.0.0",
        port=8000,
    )

    _log.info("Starting API server on http://0.0.0.0:8000")
    _log.info("Sessions persisted to sessions.db")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
