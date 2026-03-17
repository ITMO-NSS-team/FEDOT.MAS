import asyncio
from pathlib import Path

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.sandbox_e2b")

MODEL = "openai/gpt-4o-mini"
README_PATH = str(Path(__file__).resolve().parents[2] / "README.md")


async def _run_scenario(name: str, instruction: str, query: str) -> None:
    _log.info("--- {} ---", name)
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="analyst",
                instruction=instruction,
                model=MODEL,
                output_key="result",
                tools=["sandbox"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="analyst"),
    )
    maw = MAW(mcp_servers=["sandbox"])
    state = await maw.build_and_run(config, query)
    _log.info("Result: {}", str(state.get("result", ""))[:500])
    _log.info(
        "Tokens: prompt={} completion={} elapsed={:.1f}s",
        maw.total_prompt_tokens,
        maw.total_completion_tokens,
        maw.elapsed,
    )


async def main():
    # 1. Run code with pip install
    await _run_scenario(
        "pip install + run code",
        (
            "Solve the task using Python code in the sandbox: {user_query}\n"
            "Install required packages with run_command('pip install ...') first.\n"
            "Return the result."
        ),
        "Install numpy, then generate 10,000 random samples from a standard normal "
        "distribution. Compute the mean, std, and the 95th percentile. "
        "Return the results as a formatted string.",
    )

    # 2. Upload a file to the sandbox
    await _run_scenario(
        "upload file",
        (
            "{user_query}\n"
            "Upload the file and verify it arrived by running 'cat' on it in the sandbox.\n"
            "Return the first 5 lines of the uploaded file."
        ),
        f"Upload the local file at {README_PATH} to the sandbox as 'README.md'.",
    )

    # 3. Download a file from the sandbox
    await _run_scenario(
        "download file",
        (
            "{user_query}\n"
            "First create a small text file in the sandbox using run_code, "
            "then download it to the local path provided.\n"
            "Return the content of the downloaded file."
        ),
        "Create a file called 'hello.txt' containing 'Hello from E2B sandbox!' "
        "inside the sandbox, then download it to /tmp/hello_from_sandbox.txt.",
    )


if __name__ == "__main__":
    asyncio.run(main())
