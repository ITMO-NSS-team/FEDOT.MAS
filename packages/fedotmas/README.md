# fedot-mas

Core library for FEDOT.MAS — automatic multi-agent pipeline generation and execution.

## What it does

1. **Meta-agent** analyses a plain-text task and generates a `PipelineConfig` (agent definitions + execution topology).
2. **Pipeline builder** turns the config into a Google ADK agent tree.
3. **Pipeline runner** executes the tree and returns the final state with token usage stats.

Supported pipeline topologies: `sequential`, `parallel`, `loop`.

## Installation

From the workspace root:

```
uv sync --all-packages
```

Or standalone:

```
uv pip install -e packages/fedotmas
```

## Usage

```python
import asyncio
from fedotmas import MAS

async def main():
    mas = MAS()
    state = await mas.run("Compare Python and Rust for CLI tools")
    print(state)

asyncio.run(main())
```

For finer control, split into two steps:

```python
config = await mas.generate_config("Compare Python and Rust for CLI tools")
# inspect / edit config ...
state = await mas.build_and_run(config, "Compare Python and Rust for CLI tools")
```

## Public API

- `MAS` — high-level facade (`.run()`, `.generate_config()`, `.build_and_run()`)
- `PipelineConfig`, `AgentConfig`, `StepConfig` — pipeline definition models
- `PipelineResult` — execution result with state and token usage
