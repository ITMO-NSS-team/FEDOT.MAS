# Langfuse Observability

FEDOT.MAS ships an optional `LangfusePlugin` that sends full execution traces to [Langfuse](https://langfuse.com). Every meta-agent call, pipeline agent, LLM generation, and tool invocation appears as a structured trace you can inspect in the Langfuse dashboard.

## Installation

```sh
uv pip install "fedotmas[langfuse]"
# or
pip install "fedotmas[langfuse]"
```

## Configuration

Set the standard Langfuse environment variables (or pass them to the plugin constructor):

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

## Usage

Add `LangfusePlugin` to the `plugins` list when creating a `MAW` or `MAS` instance:

```python
import asyncio
from fedotmas import MAW
from fedotmas.plugins import LangfusePlugin, LoggingPlugin

async def main():
    maw = MAW(
        plugins=[LoggingPlugin(), LangfusePlugin()],
    )
    state = await maw.run("Compare Python and Rust for CLI tools")
    print(state)

asyncio.run(main())
```

The plugin accepts optional constructor arguments for finer control:

```python
LangfusePlugin(
    trace_name="my-experiment",   # custom trace name (default: "fedotmas:<task>")
    user_id="user-123",           # attach a user ID to the trace
    session_id="sess-abc",        # attach a session ID
    tags=["experiment", "v2"],    # custom tags
    metadata={"env": "staging"},  # arbitrary metadata
)
```

## What gets traced

| Component | Langfuse observation type |
|-----------|--------------------------|
| Full `run()` / `build_and_run()` call | Trace |
| Meta-agent config generation | Agent span + Generation |
| Each pipeline agent | Agent span |
| LLM calls (with model, tokens, input/output) | Generation |
| Tool calls (with args and result) | Tool span |
| Errors (model or tool) | Error-level span |

Both the meta-agent phase (config generation) and the pipeline execution phase appear under a single trace, giving you end-to-end visibility.
