# fedotmas

Core library for FEDOT.MAS.

## LLM Configuration

```env
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
FEDOTMAS_DEFAULT_MODEL=openai/gpt-5-mini
```

Set these values in `.env` in the repository root. `FEDOTMAS_DEFAULT_MODEL`
selects the model for generation and execution; without it, the meta-agent
defaults to a private model. See [`.env.example`](../../.env.example) for overrides.

## Usage

```python
import asyncio
from fedotmas import MAW

async def main():
    maw = MAW(mcp_servers=[])
    state = await maw.run("Compare Python and Rust for CLI tools")
    print(state)

asyncio.run(main())
```

For observability, plugins, optimizer, and more — see the [documentation](../../docs/index.md).
