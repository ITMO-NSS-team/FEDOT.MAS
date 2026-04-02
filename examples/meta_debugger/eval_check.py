"""Meta-debugger: using error_hint for LLM-based evaluation.

The calculator agent is instructed to return the number 2, but the task
asks for 5. There is no programmatic check here. Instead, error_hint
tells the meta-debugger what the expected behavior is, and
llm_error_detection lets the LLM decide whether the output is correct.

If the LLM determines that the output does not match the hint, recovery
is triggered and the debugger rewrites the agent's instruction.
"""

import asyncio

from fedotmas import MAW
from fedotmas.control import Controller
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="calculator",
            instruction="Верни в ответе только число 2.",
            output_key="number",
        ),
    ],
    pipeline=MAWStepConfig(agent_name="calculator"),
)


async def main():
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Верни число 5",
        config=CONFIG,
        max_retries=2,
        error_hint="Агент должен возвращать число 5, а не любое другое",
    )
    print(f"Status: {result.status}")
    print(f"Number: {result.state.get('number')}")


if __name__ == "__main__":
    asyncio.run(main())
