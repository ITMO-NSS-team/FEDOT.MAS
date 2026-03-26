# FEDOT.MAS

A framework for building and optimizing multi-agent systems. Built on top of [Google ADK](https://google.github.io/adk-docs/).

## Installation

Soon...

```bash
pip install fedotmas
```

## Two Orchestration Modes

FEDOT.MAS provides two ways to organize agents:

|                 | MAS                                             | MAW                                         |
| --------------- | ----------------------------------------------- | ------------------------------------------- |
| **Meaning**     | Multi-Agent System                              | Multi-Agent Workflow                        |
| **Control**     | LLM coordinator decides where to route the task | Fixed pipeline: sequential, parallel, loop  |
| **When to use** | Classification, routing, delegation             | Multi-step processes with predictable order |
| **Example**     | Support: router -> billing / tech / sales       | Research: researcher -> writer -> reviewer  |
| **Config**      | `MASConfig(coordinator, workers)`               | `MAWConfig(agents, pipeline)`               |

## Quick Start: MAW

MAW (Multi-Agent Workflow) runs agents in a predefined order. Each agent writes its result to a shared state under `output_key`, and the next agent can use it via `{output_key}` in its instruction.

### Fully Automatic Mode

MAW automatically generates the agent configuration and runs it:

```python
import asyncio
from fedotmas import MAW

async def main():
    maw = MAW()
    state = await maw.run("Explain the difference between TCP and UDP")
    print(state)

asyncio.run(main())
```

### Two-Step Mode

First generate the config, inspect it, then run:

```python
import asyncio
from fedotmas import MAW

async def main():
    maw = MAW()

    config = await maw.generate_config("Compare Python and Rust for CLI tools")
    print(config.model_dump_json(indent=2))  # inspect generated config

    state = await maw.build_and_run(config, "Compare Python and Rust for CLI tools")
    print(state)

asyncio.run(main())
```

### Manual Config

Full control: you define agents and the pipeline yourself.

```python
import asyncio
from fedotmas import MAW, MAWConfig
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

config = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            model="openai/gpt-4o-mini",
            instruction="Research the topic: {user_query}. Provide key facts.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            model="openai/gpt-4o-mini",
            instruction="Write a concise summary based on the research:\n\n{research}",
            output_key="summary",
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

async def main():
    maw = MAW()
    state = await maw.build_and_run(config, "What is WebAssembly?")
    print(state["summary"])

asyncio.run(main())
```

Agents run sequentially: `researcher` writes to `state["research"]`, and `writer` receives it via `{research}` in its instruction.

## Pipeline Types

### Sequential

Agents run one after another. Each sees the results of the previous ones.

```python
pipeline = MAWStepConfig(
    type="sequential",
    children=[
        MAWStepConfig(type="agent", agent_name="researcher"),
        MAWStepConfig(type="agent", agent_name="writer"),
    ],
)
```

### Parallel

Agents run simultaneously. Useful when parts of the task are independent.

```python
config = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="pros_analyst",
            model="openai/gpt-4o-mini",
            instruction="List 3 key advantages of: {user_query}",
            output_key="pros",
        ),
        MAWAgentConfig(
            name="cons_analyst",
            model="openai/gpt-4o-mini",
            instruction="List 3 key disadvantages of: {user_query}",
            output_key="cons",
        ),
        MAWAgentConfig(
            name="synthesizer",
            model="openai/gpt-4o-mini",
            instruction="Given pros:\n{pros}\n\nAnd cons:\n{cons}\n\nWrite a balanced verdict.",
            output_key="verdict",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(
                type="parallel",
                children=[
                    MAWStepConfig(type="agent", agent_name="pros_analyst"),
                    MAWStepConfig(type="agent", agent_name="cons_analyst"),
                ],
            ),
            MAWStepConfig(type="agent", agent_name="synthesizer"),
        ],
    ),
)
```

`pros_analyst` and `cons_analyst` run in parallel, then `synthesizer` combines their results.

### Loop

Agents repeat up to `max_iterations` times. The last LLM agent in the loop automatically gets access to the `exit_loop` tool for early stopping.

```python
config = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="writer",
            model="openai/gpt-4o-mini",
            instruction="Write a short poem about: {user_query}.\n"
                        "If there is feedback, incorporate it: {feedback}",
            output_key="draft",
        ),
        MAWAgentConfig(
            name="critic",
            model="openai/gpt-4o-mini",
            instruction="Review this poem:\n{draft}\n\n"
                        "If the poem is good, call the exit_loop tool.\n"
                        "Otherwise, provide specific feedback for improvement.",
            output_key="feedback",
        ),
    ],
    pipeline=MAWStepConfig(
        type="loop",
        max_iterations=3,
        children=[
            MAWStepConfig(type="agent", agent_name="writer"),
            MAWStepConfig(type="agent", agent_name="critic"),
        ],
    ),
)
```

## Quick Start: MAS

MAS (Multi-Agent System) uses a coordinator that dynamically decides which worker should handle the task.

```python
import asyncio
from fedotmas import MAS, MASConfig
from fedotmas.mas.models import MASAgentConfig

config = MASConfig(
    coordinator=MASAgentConfig(
        name="router",
        description="Routes customer queries to the right specialist",
        instruction="Analyze the user's request and route it to the appropriate specialist.",
    ),
    workers=[
        MASAgentConfig(
            name="billing_agent",
            description="Handles billing, invoices, and payment questions",
            instruction="Help the customer with their billing inquiry.",
            output_key="billing_response",
        ),
        MASAgentConfig(
            name="support_agent",
            description="Handles technical support and troubleshooting",
            instruction="Help the customer troubleshoot their issue.",
            output_key="support_response",
        ),
    ],
)

async def main():
    mas = MAS()
    state = await mas.build_and_run(config, "Why was I charged twice?")
    print(state)

asyncio.run(main())
```

The coordinator reads worker descriptions and decides that `billing_agent` is the best fit for a double charge issue.

## What’s Next

* [Optimizer](optimizer.md) - evolutionary optimization of agent prompts
