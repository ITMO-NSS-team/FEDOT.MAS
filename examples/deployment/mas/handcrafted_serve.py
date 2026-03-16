from __future__ import annotations

import os

import uvicorn

from fedotmas import MAS, MASConfig
from fedotmas.common.logging import get_logger
from fedotmas.mas.models import MASAgentConfig

_log = get_logger("fedotmas.examples.deployment.mas.handcrafted_serve")

config = MASConfig(
    coordinator=MASAgentConfig(
        name="router",
        description="Routes customer queries to the right specialist",
        instruction=(
            "You are a customer service router. Analyze the user's request "
            "and route it to the most appropriate specialist agent."
        ),
        model=os.getenv("FEDOTMAS_DEFAULT_MODEL", "openai/gpt-4o"),
    ),
    workers=[
        MASAgentConfig(
            name="billing_agent",
            description="Handles billing, invoices, and payment questions",
            instruction=(
                "You are a billing specialist. Help the customer with "
                "their billing inquiry. Be concise and helpful."
            ),
            model=os.getenv("FEDOTMAS_DEFAULT_MODEL", "openai/gpt-4o"),
            output_key="billing_response",
        ),
        MASAgentConfig(
            name="support_agent",
            description="Handles technical support and troubleshooting",
            instruction=(
                "You are a technical support specialist. Help the customer "
                "troubleshoot their issue. Provide step-by-step guidance."
            ),
            model=os.getenv("FEDOTMAS_DEFAULT_MODEL", "openai/gpt-4o"),
            output_key="support_response",
        ),
    ],
)


def main() -> None:
    mas = MAS()

    app = mas.serve(
        config,
        name="customer_support",
        session_service_uri="sqlite:///sessions.db",
        web=True,
        host="0.0.0.0",
        port=8000,
        auto_create_session=True,
    )

    _log.info("Starting MAS API server on http://0.0.0.0:8000")
    _log.info("Sessions persisted to sessions.db")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
