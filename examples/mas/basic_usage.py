import asyncio
import json

from fedotmas import MAS, MASConfig
from fedotmas.common.logging import get_logger
from fedotmas.mas.models import MASAgentConfig

_log = get_logger("fedotmas.examples.mas.basic_usage")


async def full_auto():
    """Fully automatic: MAS generates a routing config and executes it."""
    mas = MAS(mcp_servers="all")
    state = await mas.run("Handle customer support request about billing")
    _log.info("Result: {}", json.dumps(state, indent=2, default=str))


async def two_step():
    """Generate config first, inspect it, then execute."""
    mas = MAS()

    config = await mas.generate_config("Handle customer support request")
    _log.info("Config: {}", config.model_dump_json(indent=2))

    state = await mas.build_and_run(config, "I need help with my invoice")
    _log.info("Result: {}", json.dumps(state, indent=2, default=str))


async def handcrafted():
    """Manually define a coordinator + workers routing system."""
    config = MASConfig(
        coordinator=MASAgentConfig(
            name="router",
            description="Routes customer queries to the right specialist",
            instruction=(
                "You are a customer service router. Analyze the user's request "
                "and route it to the most appropriate specialist agent."
            ),
        ),
        workers=[
            MASAgentConfig(
                name="billing_agent",
                description="Handles billing, invoices, and payment questions",
                instruction=(
                    "You are a billing specialist. Help the customer with "
                    "their billing inquiry. Be concise and helpful."
                ),
                output_key="billing_response",
            ),
            MASAgentConfig(
                name="support_agent",
                description="Handles technical support and troubleshooting",
                instruction=(
                    "You are a technical support specialist. Help the customer "
                    "troubleshoot their issue. Provide step-by-step guidance."
                ),
                output_key="support_response",
            ),
        ],
    )

    mas = MAS()
    state = await mas.build_and_run(config, "Why was I charged twice?")
    _log.info("Result: {}", json.dumps(state, indent=2, default=str))


async def with_tools():
    """Workers with MCP tools assigned."""
    config = MASConfig(
        coordinator=MASAgentConfig(
            name="router",
            description="Routes tasks to specialized workers",
            instruction="Route the user request to the appropriate worker.",
        ),
        workers=[
            MASAgentConfig(
                name="coder",
                description="Writes and executes code",
                instruction="Write and run code to solve the user's problem.",
                tools=["light-sandbox"],
                output_key="code_result",
            ),
            MASAgentConfig(
                name="researcher",
                description="Searches the web for information",
                instruction="Search the web to answer the user's question.",
                tools=["web-search"],
                output_key="research_result",
            ),
        ],
    )

    mas = MAS(mcp_servers="all")
    state = await mas.build_and_run(config, "What is the fibonacci sequence?")
    _log.info("Result: {}", json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    # Pick one:
    asyncio.run(full_auto())
    # asyncio.run(two_step())
    # asyncio.run(handcrafted())
    # asyncio.run(with_tools())
