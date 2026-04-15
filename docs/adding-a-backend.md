# Adding a Backend to FEDOT.MAS

This guide walks you through implementing a custom agentic backend for FEDOT.MAS. A backend bridges the framework-neutral agent descriptions produced by FEDOT.MAS with a specific agentic library (LangChain, PydanticAI, CrewAI, AutoGen, etc.).

## Architecture Overview

```
MAW / MAS (user-facing)
    │
    ▼
Builders ──► AgentTree (framework-neutral descriptors)
    │
    ▼
Backend Registry ──► YourBackend.create_runner()
    │
    ▼
YourRunner.run_pipeline()  /  run_single_agent()
    │
    ▼
PipelineResult / SingleAgentResult
```

FEDOT.MAS never touches your framework directly. It produces an `AgentTree` — a tree of plain dataclasses — and hands it to your runner. Your runner translates those descriptors into your framework's native objects and executes them.

## What You Need to Implement

| Component | Purpose |
|-----------|---------|
| **Backend class** | Factory that creates runners (`BackendProtocol`) |
| **Runner class** | Executes agent trees and single-agent calls (`RunnerProtocol`) |
| **Builder functions** | Translate `AgentTree` descriptors into your framework's agents |
| **Middleware adapter** | Map `MiddlewareProtocol` hooks to your framework's lifecycle |

## Step 1: Create the Package Structure

```
packages/fedotmas/src/fedotmas/backends/mybackend/
├── __init__.py      # Backend class + registration
├── runner.py        # Runner implementation
├── builder.py       # Descriptor → framework object translation
└── middleware.py     # Middleware adapter (optional but recommended)
```

## Step 2: Implement the Backend Class

Your backend class must satisfy `BackendProtocol` — a single method that creates a runner:

```python
# backends/mybackend/__init__.py
from __future__ import annotations

from typing import Any

from fedotmas.interfaces.runner import RunnerProtocol

from .runner import MyRunner


class MyBackend:
    """MyFramework backend for FEDOT.MAS."""

    def create_runner(
        self,
        *,
        session_service: Any | None = None,
        memory_service: Any | None = None,
    ) -> RunnerProtocol:
        return MyRunner(
            session_service=session_service,
            memory_service=memory_service,
        )
```

## Step 3: Implement the Runner

The runner is the core of your backend. It implements two async methods:

### `run_pipeline` — Execute a Full Agent Tree

This is the main execution path. It receives a tree of agent descriptors, translates them into your framework's objects, executes them, and returns the final state.

```python
# backends/mybackend/runner.py
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from fedotmas._settings import ModelConfig
from fedotmas.interfaces.agent import AgentTree
from fedotmas.interfaces.middleware import MiddlewareProtocol
from fedotmas.interfaces.runner import PipelineResult, SingleAgentResult
from fedotmas.interfaces.tools import ToolDescriptor

from .builder import build_framework_tree


class MyRunner:
    def __init__(
        self,
        *,
        session_service: Any | None = None,
        memory_service: Any | None = None,
    ) -> None:
        self._session_service = session_service
        self._memory_service = memory_service

    async def run_pipeline(
        self,
        agent_tree: AgentTree,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        middlewares: list[MiddlewareProtocol] | None = None,
        backend_plugins: list[Any] | None = None,
    ) -> PipelineResult:
        # 1. Convert descriptors to your framework's agent objects
        native_agent = build_framework_tree(agent_tree)

        # 2. Set up state
        state: dict[str, Any] = {"user_query": user_query}
        if initial_state:
            state.update(initial_state)

        # 3. Wire up middleware (see Step 5)
        # ...

        # 4. Execute using your framework
        start = time.monotonic()
        total_prompt = 0
        total_completion = 0

        # ... your framework's execution logic here ...

        elapsed = time.monotonic() - start

        # 5. Return results
        return PipelineResult(
            state=state,  # final state dict with all agent outputs
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            elapsed=elapsed,
        )
```

### `run_single_agent` — Execute a Single LLM Call

Used by the meta-agent (config generation) and debugger (error recovery). It runs one agent with specific parameters and returns structured output.

```python
    async def run_single_agent(
        self,
        *,
        agent_name: str,
        instruction: str,
        user_message: str,
        model: str | ModelConfig,
        temperature: float,
        output_schema: type[BaseModel] | None = None,
        output_key: str,
        tools: list[ToolDescriptor] | None = None,
        after_tool_callback: Any | None = None,
        initial_state: dict[str, Any] | None = None,
        backend_plugins: list[Any] | None = None,
    ) -> SingleAgentResult:
        # 1. Resolve model (string or ModelConfig)
        # 2. Build tools from ToolDescriptors
        # 3. Create and run a single agent
        # 4. Extract raw_output from state[output_key]

        return SingleAgentResult(
            raw_output=raw_output,      # value from state[output_key]
            state=final_state,          # full state dict
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed=elapsed,
        )
```

**Important:** `raw_output` must not be `None` — the caller treats `None` as "agent did not produce output" and raises an error.

## Step 4: Implement the Builder

The builder translates the framework-neutral descriptor tree into your framework's native objects. You must handle all descriptor types:

```python
# backends/mybackend/builder.py
from __future__ import annotations

from fedotmas.interfaces.agent import (
    AgentDescriptor,
    AgentTree,
    LoopDescriptor,
    ParallelDescriptor,
    RoutingDescriptor,
    SequentialDescriptor,
)
from fedotmas.interfaces.tools import ToolDescriptor


def build_framework_tree(tree: AgentTree):
    """Convert a framework-neutral AgentTree into native agents."""
    if isinstance(tree, RoutingDescriptor):
        return _build_routing(tree)
    return _build_node(tree)


def _build_node(node: AgentTree):
    """Recursively build the agent tree."""
    match node:
        case AgentDescriptor():
            return _build_agent(node)

        case SequentialDescriptor():
            children = [_build_node(c) for c in node.children]
            # Create your framework's sequential execution wrapper
            # that runs children one after another
            return YourSequentialAgent(name=node.name, steps=children)

        case ParallelDescriptor():
            children = [_build_node(c) for c in node.children]
            # Create your framework's parallel execution wrapper
            # that runs children concurrently
            return YourParallelAgent(name=node.name, steps=children)

        case LoopDescriptor():
            children = [_build_node(c) for c in node.children]
            # Create a loop that runs children repeatedly
            # up to node.max_iterations times.
            # Provide an exit mechanism to the agent named
            # node.exit_loop_agent (or the last agent if None)
            return YourLoopAgent(
                name=node.name,
                steps=children,
                max_iterations=node.max_iterations,
            )

        case _:
            raise TypeError(f"Unknown node type: {type(node)}")


def _build_agent(desc: AgentDescriptor):
    """Create a single LLM agent from a descriptor."""
    tools = _build_tools(desc.tools)

    return YourLLMAgent(
        name=desc.name,
        instruction=desc.instruction,
        model=desc.model,            # str | ModelConfig | None
        tools=tools,
        output_key=desc.output_key,  # key to write output to in state
        description=desc.description,
        temperature=desc.temperature,
    )


def _build_routing(desc: RoutingDescriptor):
    """Build a routing/MAS system.

    The coordinator agent decides which worker to delegate to.
    """
    coordinator = _build_agent(desc.coordinator)
    workers = [_build_agent(w) for w in desc.workers]

    # Your framework's routing mechanism
    return YourRoutingAgent(
        coordinator=coordinator,
        workers=workers,
    )


def _build_tools(descriptors: list[ToolDescriptor]) -> list:
    """Convert ToolDescriptors to your framework's tool objects."""
    tools = []
    for desc in descriptors:
        if desc.mcp_server is not None:
            # MCP tool — connect to MCP server
            # desc.mcp_server is a StdioMCPServer or HttpMCPServer
            tools.append(create_mcp_tool(desc))
        elif desc.function is not None:
            # Plain Python function tool
            tools.append(create_function_tool(desc.function))
    return tools
```

### Descriptor Reference

Each descriptor carries these fields:

**`AgentDescriptor`** — A single LLM-powered agent:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique agent name |
| `instruction` | `str` | System prompt. May contain `{placeholders}` referencing state keys |
| `model` | `str \| ModelConfig \| None` | LLM model identifier |
| `tools` | `list[ToolDescriptor]` | Tools available to this agent |
| `output_key` | `str \| None` | State key where the agent writes its output |
| `description` | `str` | Short description (used by routing coordinators) |
| `temperature` | `float \| None` | LLM sampling temperature |

**`SequentialDescriptor`** — Run children in order. Fields: `name`, `children: list[AgentTree]`.

**`ParallelDescriptor`** — Run children concurrently. Fields: `name`, `children: list[AgentTree]`.

**`LoopDescriptor`** — Repeat children up to N times. Fields: `name`, `children: list[AgentTree]`, `max_iterations: int`, `exit_loop_agent: str | None`.

**`RoutingDescriptor`** — Coordinator dispatches to workers. Fields: `coordinator: AgentDescriptor`, `workers: list[AgentDescriptor]`.

**`ToolDescriptor`** — A tool available to an agent. Exactly one of `mcp_server` or `function` is set:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool name |
| `mcp_server_name` | `str \| None` | Name of the MCP server |
| `mcp_server` | `MCPServerConfig \| None` | `StdioMCPServer` or `HttpMCPServer` |
| `function` | `Callable \| None` | Python callable |
| `description` | `str` | Tool description |
| `after_tool_callback` | `Callable \| None` | Hook called after tool execution |

## Step 5: Adapt Middleware

FEDOT.MAS uses `MiddlewareProtocol` for cross-backend lifecycle hooks (logging, checkpointing, evaluation, skip-completed). Your runner must call these hooks at the right time.

The protocol is minimal:

```python
class MiddlewareProtocol(Protocol):
    async def before_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> dict[str, str] | None:
        """Called before agent executes.

        Return None to proceed normally.
        Return a dict to short-circuit (skip the agent, use dict as output).
        """
        ...

    async def after_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> None:
        """Called after agent finishes. Return value is ignored."""
        ...
```

### Integration Pattern

In your runner or execution loop, call middleware hooks around each agent:

```python
async def _execute_agent(self, agent, state, middlewares):
    # Before hooks
    for mw in middlewares or []:
        result = await mw.before_agent(agent.name, state)
        if result is not None:
            # Short-circuit: skip this agent
            return result

    # Run the agent
    output = await agent.run(state)
    state[agent.output_key] = output

    # After hooks
    for mw in middlewares or []:
        await mw.after_agent(agent.name, state)

    return None
```

**Convention:** Skip middleware calls for workflow nodes (names starting with `seq_`, `par_`, `loop_`). Only fire hooks for actual LLM agents.

The ADK backend implements this via a `MiddlewareAdapter` class that wraps the list of middlewares as a single ADK `BasePlugin`. If your framework has its own plugin/hook system, write a similar adapter.

## Step 6: Register the Backend

Add auto-registration so the backend is available when installed:

```python
# backends/mybackend/__init__.py  (append to the file)

def register():
    """Register this backend with FEDOT.MAS."""
    from fedotmas.backends import register_backend
    register_backend("mybackend", MyBackend)
```

Then add auto-registration in the backends registry:

```python
# backends/__init__.py — add alongside _auto_register_adk()

def _auto_register_mybackend() -> None:
    try:
        from fedotmas.backends.mybackend import MyBackend
        register_backend("mybackend", MyBackend)
    except ImportError:
        pass

_auto_register_mybackend()
```

Alternatively, for third-party backends distributed as separate packages, use a Python entry point:

```toml
# In your package's pyproject.toml
[project.entry-points."fedotmas.backends"]
mybackend = "my_package:MyBackend"
```

## Step 7: Add the Dependency as an Optional Extra

```toml
# packages/fedotmas/pyproject.toml
[project.optional-dependencies]
adk = ["google-adk>=1.26"]
mybackend = ["my-framework>=1.0"]
all = ["google-adk>=1.26", "my-framework>=1.0"]
```

## Usage

Once registered, users select your backend at construction time:

```python
from fedotmas import MAW

maw = MAW(backend="mybackend")
state = await maw.run("Summarize this document")
```

Or with explicit config:

```python
maw = MAW(backend="mybackend")
config = await maw.generate_config("Compare TCP and UDP")
state = await maw.build_and_run(config, "Compare TCP and UDP")
```

The `Controller` and all other high-level APIs automatically route through your backend.

## Testing Your Backend

### Minimal Smoke Test

```python
import asyncio
from fedotmas import MAW, MAWConfig
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

async def test_backend():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="greeter",
                instruction="Say hello to the user about: {user_query}",
                model="your-model-id",
                output_key="greeting",
            ),
        ],
        pipeline=MAWStepConfig(agent_name="greeter"),
    )

    maw = MAW(backend="mybackend")
    state = await maw.build_and_run(config, "WebAssembly")
    assert "greeting" in state
    print(state["greeting"])

asyncio.run(test_backend())
```

### What to Test

1. **Single agent** — one `AgentDescriptor`, verify output appears in state
2. **Sequential pipeline** — two agents in sequence, second reads first's output
3. **Parallel pipeline** — two agents run concurrently
4. **Loop** — agents repeat, exit condition triggers
5. **Routing** — coordinator dispatches to workers (MAS)
6. **Middleware** — `LoggingPlugin` fires, `CheckpointPlugin` records snapshots
7. **Tools** — function tools and MCP tools resolve and execute
8. **Single agent call** — `run_single_agent` works (needed for meta-agent/debugger)
9. **Error handling** — runtime errors propagate as `RuntimeError` with the format `"Agent '{name}' failed with error {code}: {message}"`

## Reference: ADK Backend

The Google ADK backend in `fedotmas/backends/adk/` is the reference implementation. Key files:

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~20 | Backend class, `create_runner()` |
| `runner.py` | ~286 | `ADKRunner` with `run_pipeline` and `run_single_agent` |
| `builder.py` | ~180 | Descriptor-to-ADK translation (`build_adk_tree`, `_build_node`, `_build_llm_agent`, `_build_tools`) |
| `middleware.py` | ~60 | `MiddlewareAdapter(BasePlugin)` wrapping `MiddlewareProtocol` |
| `serving.py` | ~50 | ADK-specific FastAPI serving (optional) |
