# fedot-mas

Core library for FEDOT.MAS.

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
