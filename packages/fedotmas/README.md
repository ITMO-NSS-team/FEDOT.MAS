# fedotmas

Core library for FEDOT.MAS.

## LLM Configuration

```env
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

See `.env.example` for all available options.

## Usage

```python
import asyncio
from fedotmas import MAW

async def main():
    maw = MAW()
    state = await maw.run("Compare Python and Rust for CLI tools")
    print(state)

asyncio.run(main())
```

For observability, plugins, optimizer, and more — see the [documentation](../../docs/index.md).
