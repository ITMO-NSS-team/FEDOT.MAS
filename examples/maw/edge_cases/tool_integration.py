import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.maw.edge_cases.tool_integration")

META_MODEL = "openai/gpt-oss-120b"
MODEL = "openai/gpt-4o-mini"


async def mcp_all_servers():
    """MAS(mcp_servers="all") — meta-agent sees every registered MCP tool."""
    _log.info("------ Scenario: mcp_all_servers ------")
    try:
        maw = MAW(mcp_servers="all", meta_model=META_MODEL)
        config = await maw.generate_config(
            "Download the Python zen (import this) and summarize its key principles"
        )
        _log.info(
            "Generated config: {} agents, pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        for agent in config.agents:
            _log.info("  Agent '{}' tools={}", agent.name, agent.tools)
        state = await maw.build_and_run(
            config, "Download the Python zen and summarize its key principles"
        )
        _log.info("Result state keys: {}", list(state.keys()))
        for key in state:
            if key != "user_query":
                _log.info("Output [{}]: {}", key, str(state[key])[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def mcp_filtered_servers():
    """MAS(mcp_servers=["light-sandbox"]) — only sandbox tool available."""
    _log.info("------ Scenario: mcp_filtered_servers ------")
    try:
        maw = MAW(mcp_servers=["light-sandbox"], meta_model=META_MODEL)
        config = await maw.generate_config(
            "Calculate the first 10 Fibonacci numbers using Python code"
        )
        _log.info(
            "Generated config: {} agents, pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        for agent in config.agents:
            _log.info("  Agent '{}' tools={}", agent.name, agent.tools)
        state = await maw.build_and_run(
            config, "Calculate the first 10 Fibonacci numbers using Python code"
        )
        _log.info("Result state keys: {}", list(state.keys()))
        for key in state:
            if key != "user_query":
                _log.info("Output [{}]: {}", key, str(state[key])[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def mcp_handcrafted_with_tools():
    """Hand-crafted config with tools=["light-sandbox"] on an agent."""
    _log.info("------ Scenario: mcp_handcrafted_with_tools ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="coder",
                    instruction=(
                        "Write and execute Python code to solve: {user_query}\n"
                        "Use the sandbox tool to run the code and return the output."
                    ),
                    model=MODEL,
                    output_key="result",
                    tools=["light-sandbox"],
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="coder"),
        )
        maw = MAW(mcp_servers=["light-sandbox"])
        state = await maw.build_and_run(
            config, "Sort the list [5, 2, 8, 1, 9] and return the result"
        )
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", str(state.get("result", ""))[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def mcp_agent_decides_tool_use():
    """Agent receives a task it *can* solve with tools but doesn't have to."""
    _log.info("------ Scenario: mcp_agent_decides_tool_use ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="assistant",
                    instruction=(
                        "Answer the user's question: {user_query}\n"
                        "You have a code sandbox tool available. Use it only if "
                        "you need to verify something computationally. "
                        "Otherwise, answer directly from your knowledge."
                    ),
                    model=MODEL,
                    output_key="answer",
                    tools=["light-sandbox"],
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="assistant"),
        )
        maw = MAW(mcp_servers=["light-sandbox"])
        state = await maw.build_and_run(config, "What is the capital of Japan?")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", str(state.get("answer", ""))[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def mcp_sequential_tool_chain():
    """Sequential pipeline: first agent downloads data, second processes it in sandbox."""
    _log.info("------ Scenario: mcp_sequential_tool_chain ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="downloader",
                    instruction=(
                        "Download the content from this URL: "
                        "https://raw.githubusercontent.com/jikkujose/data/master/iris.csv\n"
                        "Return the raw CSV content."
                    ),
                    model=MODEL,
                    output_key="raw_data",
                    tools=["download-url-content"],
                ),
                MAWAgentConfig(
                    name="analyzer",
                    instruction=(
                        "You have the following CSV data:\n{raw_data}\n\n"
                        "Write and execute Python code to:\n"
                        "1. Parse the CSV\n"
                        "2. Count the number of rows\n"
                        "3. List the column names\n"
                        "Return a short summary of the dataset."
                    ),
                    model=MODEL,
                    output_key="analysis",
                    tools=["light-sandbox"],
                ),
            ],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="downloader"),
                    MAWStepConfig(type="agent", agent_name="analyzer"),
                ],
            ),
        )
        maw = MAW(mcp_servers=["download-url-content", "light-sandbox"])
        state = await maw.build_and_run(config, "Analyze the Iris dataset")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", str(state.get("analysis", ""))[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def mcp_unknown_server_error():
    """MAS(mcp_servers=["nonexistent"]) — should raise ValueError."""
    _log.info("------ Scenario: mcp_unknown_server_error ------")
    try:
        MAW(mcp_servers=["nonexistent"])
        _log.error("Expected ValueError was not raised!")
    except ValueError as e:
        _log.info("Correctly raised ValueError: {}", e)
    except Exception as e:
        _log.error("Unexpected error type {}: {}", type(e).__name__, e)


async def main():
    _log.info("Starting tool_integration edge cases")
    await mcp_all_servers()
    await mcp_filtered_servers()
    await mcp_handcrafted_with_tools()
    await mcp_agent_decides_tool_use()
    await mcp_sequential_tool_chain()
    await mcp_unknown_server_error()
    _log.info("All tool_integration scenarios completed")


if __name__ == "__main__":
    asyncio.run(main())
