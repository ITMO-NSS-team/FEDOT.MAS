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
