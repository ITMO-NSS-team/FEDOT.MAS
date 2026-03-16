# fedotmas

Core library for FEDOT.MAS.

## LLM Configuration

FEDOT.MAS uses [LiteLLM](https://docs.litellm.ai/) under the hood via Google ADK. Model routing is determined by the model prefix (`openai/...`, `openrouter/...`, etc.) and standard environment variables:

```env
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

See `.env.example` for all available options.

## Bifrost (optional local proxy)

Run Bifrost locally via Docker to get load balancing, failover, and centralized key management:

```
just bifrost
```

Then point your base URL at the proxy:

```env
OPENAI_BASE_URL=http://localhost:9090/litellm
```

To stop:

```
just bifrost-stop
```

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
