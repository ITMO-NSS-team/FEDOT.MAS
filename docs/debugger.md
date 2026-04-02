# Meta-debugger

The meta-debugger is an LLM agent that fixes your pipeline configuration when something goes wrong. It receives the current config and a description of the error, then returns a corrected config. The pipeline is retried with the fix applied.

You do not need to write recovery logic yourself. The `Controller.run_with_recovery()` method handles the entire cycle: run the pipeline, catch the error, call the debugger, retry.

## Quick Start

```python
import asyncio

from fedotmas import MAW
from fedotmas.control import Controller
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction="Research the topic: {user_query}. Use data from {nonexistent}.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            instruction="Write a report based on: {research}",
            output_key="report",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="writer"),
        ],
    ),
)


async def main():
    maw = MAW()
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Analyze the electric vehicle market",
        config=CONFIG,
        max_retries=2,
    )
    print(result.status)


asyncio.run(main())
```

The researcher agent references `{nonexistent}`, a variable that does not exist in the pipeline state. This causes a RuntimeError during execution. The meta-debugger reads the error message, understands that the instruction contains a bad variable reference, and rewrites the instruction to remove it. The pipeline is then retried with the corrected config and completes successfully.

## Two Types of Errors

The meta-debugger handles two types of problems.

The first type is framework exceptions. These are Python errors that crash the pipeline, like a RuntimeError when an agent references a variable that does not exist, or when an API call fails with an unexpected response. The framework catches the exception automatically and passes the error message to the debugger.

The second type is logical errors. The pipeline completes without any exception, but the agent's output is wrong. For example, you asked the agent to return the number 5, but it returned 2. There is no crash, so the framework thinks everything is fine. To catch these problems, you write a check function that inspects the output and returns an error message if something is wrong.

## Programmatic Checks

A check function receives the full pipeline state as a dictionary and returns either an error message (a string describing what went wrong) or `None` if the output is correct. The error message is passed directly to the meta-debugger, so write it as an instruction that explains what needs to be fixed.

```python
def check_number(state: dict) -> str | None:
    output = str(state.get("number", ""))
    if "5" not in output:
        return f"Agent returned {output!r}, but should return 5. Fix the instruction."
    return None


result = await ctrl.run_with_recovery(
    "Return the number 5",
    config=CONFIG,
    max_retries=2,
    checks={"calculator": check_number},
)
```

The `checks` parameter is a dictionary where keys are agent names and values are check functions. When the agent named `calculator` finishes, the check function runs. If it returns a string, the pipeline stops and the meta-debugger receives that string as the problem description. The debugger then rewrites the agent's instruction and retries the pipeline.

You can add checks for multiple agents at once. Each check runs only after its corresponding agent completes, so if the first agent produces a bad output, the pipeline stops early without running the remaining agents.

## Error Hint

Sometimes the exception message alone is not enough for the debugger to understand what went wrong. The `error_hint` parameter lets you provide additional context.

```python
result = await ctrl.run_with_recovery(
    "Analyze the EV market",
    config=CONFIG,
    max_retries=2,
    error_hint="The researcher agent may reference variables that don't exist in state",
)
```

The hint is appended to the debugger's instruction as extra context. It does not replace checks. Checks detect the problem, the hint explains the context. You can use them together or separately.

## Custom Fix Tools

By default, the meta-debugger has one tool: `fix_instruction`, which rewrites the instruction of a specific agent. You can give the debugger additional tools through the `fix_tools` parameter.

A fix tool is an async function that reads and writes the config through `tool_context.state["config"]`. The debugger decides which tools to call based on the error description.

```python
async def fix_model(tool_context, agent_name: str, new_model: str) -> str:
    """Switch the agent to a different model."""
    config = MAWConfig.model_validate_json(tool_context.state["config"])
    agent = next((a for a in config.agents if a.name == agent_name), None)
    if agent is None:
        return f"Error: agent '{agent_name}' not found"
    updated = agent.model_copy(update={"model": new_model})
    config = config.replace_agent(agent_name, updated)
    tool_context.state["config"] = config.model_dump_json()
    return f"Switched {agent_name} to {new_model}"


result = await ctrl.run_with_recovery(
    "Analyze the EV market",
    config=CONFIG,
    fix_tools=[fix_model],
)
```

The default `fix_instruction` tool is replaced when you pass `fix_tools`. If you want to keep it alongside your custom tools, include it explicitly.

## When to Use the Optimizer Instead

The meta-debugger fixes specific errors. It takes one config, receives one error description, and produces one corrected config. This is the right tool when an agent crashes or fails a simple check.

If you need to evaluate quality on a scale, compare multiple config variants, or use an ensemble of judges, that is a different problem. The [Optimizer](optimizer.md) runs an evolutionary loop over many iterations, scores each candidate with a Scorer or LLMJudge, and converges on the best config across a training set of tasks.

Use the debugger for recovery from errors. Use the optimizer for improving quality.
