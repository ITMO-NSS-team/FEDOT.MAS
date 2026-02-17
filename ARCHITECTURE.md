# AutoMAS → fedot-mas Migration Plan (Google ADK)

## Context

AutoMAS (`automas-research/src/automas/`) is a framework for automatic generation and execution of multi-agent pipelines. It uses pydantic-ai for LLM agents and a custom DAG execution engine. The goal is to migrate to Google ADK (`google-adk`), leveraging its native workflow agents (SequentialAgent, ParallelAgent, LoopAgent), session state, and MCP integration. The new project lives in `src/fedotmas/`.

### Key Decisions
- **Pipeline config format**: Tree-based (meta-agent generates nested tree directly, not DAG)
- **LLM provider**: Configurable — Gemini (native), Claude (`anthropic` package), or any provider via LiteLLM (`provider/model` format). ADK supports all three natively via `google-adk[extensions]`.
- **MCP servers**: Copy only `download` server as reference implementation with clear interface/contract. Other servers (document, web, media, browser, sandbox) deferred — user will add them following the template.

---

## 1. Analysis of Old AutoMAS Architecture

### Core Data Flow
```
User query
    │
    ▼
┌──────────────────┐     ┌──────────────────┐
│  PoolGenerator   │────▶│  GraphGenerator   │
│  (meta-agent)    │     │  (meta-agent)     │
│                  │     │                   │
│  Output:         │     │  Input: AgentPool │
│  List[AgentSchema]│    │  Output: GraphDict│
│  → AgentPool     │     │  (adjacency list) │
└──────────────────┘     └──────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   PipelineBuilder    │
                    │                      │
                    │  create_from_pool()  │
                    │  → topological_sort  │
                    │  → Pipeline          │
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │      Pipeline        │
                    │                      │
                    │  execution_levels    │
                    │  (level-parallel)    │
                    │  NodeSession for     │
                    │  inter-node data     │
                    └──────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `AgentNode` | `pipeline/node.py` | DAG node wrapping pydantic-ai Agent (name, instructions, model, mcp_tools, children/parents) |
| `Pipeline` | `pipeline/pipeline.py` | Executes DAG level-by-level; nodes at same depth run concurrently via `asyncio.gather` |
| `PipelineBuilder` | `pipeline/pipeline_builder.py` | Builds DAG from adjacency list (`GraphDict = Dict[str, List[str]]`), validates acyclic, topological sort |
| `NodeSession` | `pipeline/node_session.py` | Manages inter-node data: entry nodes get raw query, downstream nodes get structured JSON with all parent results |
| `AgentPool` | `agent_pool.py` | Registry of AgentNode configs, iterable collection |
| `PoolGenerator` | `meta_agents/pool_gen.py` | LLM generates `List[AgentSchema]` (name, instructions, mcp_tools, model) |
| `GraphGenerator` | `meta_agents/graph_gen.py` | LLM generates `GraphDict` (adjacency list) given AgentPool |
| `UnifiedGenerator` | `meta_agents/unified_gen.py` | Generates both pool + graph in one LLM call |
| `BaseMetaAgent` | `meta_agents/base.py` | Abstract base for meta-agents, wraps pydantic-ai Agent with structured output |
| MCP Registry | `mcp/registry.py` | Maps server names → `MCPServerConfig`, creates `MCPServerStdio` toolsets |

### Data Passing Between Nodes (old)
- Entry nodes receive raw user query string
- Downstream nodes receive structured JSON via `NodeSession.get_input_for_node()`:
  ```json
  {
    "pipeline_context": {
      "original_query": "...",
      "parent_nodes": [{"node_name": "...", "output": "..."}],
      "child_nodes": [...]
    },
    "task": "Process the result from previous node..."
  }
  ```

---

## 2. Concept Mapping: AutoMAS → ADK

| AutoMAS Concept | ADK Equivalent | Notes |
|-----------------|---------------|-------|
| `AgentNode` | `LlmAgent` with `output_key` | Each agent writes result to `state[output_key]` |
| `Pipeline` (DAG, level-parallel) | Tree of `SequentialAgent` / `ParallelAgent` | DAG linearized into nested workflow agents |
| `PipelineBuilder.build()` | Builder function: config → ADK agent tree | Constructs nested SequentialAgent/ParallelAgent/LoopAgent |
| `NodeSession` (inter-node data) | `session.state` + `output_key` + `{key}` templates | ADK's native state mechanism replaces custom JSON passing |
| `AgentPool` (agent configs) | List of agent config dicts/dataclasses | Similar role, different backing implementation |
| `PoolGenerator` (meta-agent) | `LlmAgent` generating JSON config via structured output | ADK can use `output_schema` for Pydantic models |
| `GraphGenerator` (meta-agent) | Part of meta-agent, generates topology config | Topology is now a tree spec, not arbitrary DAG |
| `UnifiedGenerator` | Single `LlmAgent` meta-agent | Generates full pipeline config |
| `GraphDict` (adjacency list) | Tree-based pipeline config (see schema below) | New config format needed for tree structure |
| MCP Registry | `McpToolset` with `StdioConnectionParams` | ADK has native MCP support |
| Post-order traversal | `SequentialAgent` ordering | ADK handles execution order automatically |
| Level-parallel execution | `ParallelAgent` | Native parallel execution in ADK |
| Critic-actor loop | `LoopAgent` + `exit_loop` tool | `exit_loop` sets `escalate=True` to break loop |
| `AutoMAS.arun()` | Orchestrator: meta-agent → builder → `Runner.run()` | Same flow, different primitives |

### How Data Flows in ADK
```
                    ┌─────────────────────┐
                    │    session.state     │
                    │  (shared key-value)  │
                    └─────────────────────┘
                          ▲         │
           output_key     │         │  {state_key} in
           writes here    │         │  instruction
                          │         ▼
              ┌───────────┴───────────────────┐
              │         LlmAgent              │
              │  instruction: "... {prev}..." │
              │  output_key: "my_result"      │
              └───────────────────────────────┘
```

---

## 3. Pipeline Configuration Schema

The meta-agent generates this JSON. The builder constructs an ADK agent tree from it.

```python
# Pydantic models for pipeline configuration

class AgentConfig(BaseModel):
    """Configuration for a single LLM agent."""
    name: str                          # Unique agent name
    instruction: str                   # Agent instruction (can contain {state_key} refs)
    model: str | None = None           # LLM model (None = inherit from settings)
    output_key: str                    # Key in session.state for this agent's output
    tools: list[str] = []             # MCP server names from registry
    # Model format examples:
    #   "gemini-2.5-flash"                  — Gemini (native)
    #   "claude-sonnet-4-20250514"          — Claude (requires anthropic pkg)
    #   "openrouter/anthropic/claude-sonnet-4" — via LiteLLM

class PipelineNodeConfig(BaseModel):
    """A node in the pipeline tree."""
    type: Literal["agent", "sequential", "parallel", "loop"]

    # For type="agent": reference to an agent config
    agent_name: str | None = None

    # For type="sequential"/"parallel": ordered list of child nodes
    children: list[PipelineNodeConfig] = []

    # For type="loop": max iterations
    max_iterations: int | None = None

class PipelineConfig(BaseModel):
    """Complete pipeline configuration generated by meta-agent."""
    agents: list[AgentConfig]         # All agent definitions
    pipeline: PipelineNodeConfig      # Root of the pipeline tree
```

### Example: Research Pipeline with Critic
```json
{
  "agents": [
    {
      "name": "researcher",
      "instruction": "Research the topic: {user_query}. Provide detailed findings.",
      "output_key": "research_result",
      "tools": ["web-search"]
    },
    {
      "name": "writer",
      "instruction": "Write a report based on: {research_result}",
      "output_key": "draft",
      "tools": []
    },
    {
      "name": "critic",
      "instruction": "Review the draft: {draft}. If quality is good, call exit_loop. Otherwise provide feedback in your response.",
      "output_key": "feedback",
      "tools": []
    },
    {
      "name": "reviser",
      "instruction": "Revise the draft: {draft} based on feedback: {feedback}",
      "output_key": "draft",
      "tools": []
    }
  ],
  "pipeline": {
    "type": "sequential",
    "children": [
      {"type": "agent", "agent_name": "researcher"},
      {"type": "loop", "max_iterations": 3, "children": [
        {"type": "agent", "agent_name": "writer"},
        {"type": "agent", "agent_name": "critic"}
      ]}
    ]
  }
}
```

---

## 4. Architecture Diagrams

### Overall Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        fedot-mas                            │
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌───────────────────┐    │
│  │  meta/   │───▶│ pipeline/ │───▶│  Google ADK        │    │
│  │          │    │           │    │  Runtime            │    │
│  │ MetaAgent│    │ Builder   │    │                     │    │
│  │ (LLM)   │    │ config →  │    │ SequentialAgent     │    │
│  │          │    │ ADK tree  │    │ ParallelAgent       │    │
│  │ Output:  │    │           │    │ LoopAgent           │    │
│  │ Pipeline │    │           │    │ LlmAgent            │    │
│  │ Config   │    │           │    │                     │    │
│  └──────────┘    └───────────┘    └───────────────────┘    │
│       │                                     ▲               │
│       │          ┌───────────┐              │               │
│       │          │   mcp/    │──────────────┘               │
│       │          │           │                              │
│       │          │ Registry  │   McpToolset instances       │
│       │          │ of MCP    │   for agent tools            │
│       └─────────▶│ servers   │                              │
│                  └───────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: User Query → Pipeline Execution
```
User query: "Research quantum computing trends"
    │
    ▼
┌─────────────────────────────┐
│ 1. Meta-Agent (LlmAgent)   │
│    instruction: "Design a   │
│    pipeline for: {query}"   │
│    output_schema: Pipeline- │
│    Config (Pydantic)        │
│                             │
│    Output: PipelineConfig   │
│    (agents + tree topology) │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. Pipeline Builder         │
│                             │
│    PipelineConfig           │
│         │                   │
│    resolve agents           │
│    resolve MCP tools        │
│    build ADK tree:          │
│                             │
│    SequentialAgent(          │
│      sub_agents=[           │
│        LlmAgent(researcher),│
│        ParallelAgent(       │
│          sub_agents=[       │
│            LlmAgent(a1),    │
│            LlmAgent(a2)     │
│          ]                  │
│        ),                   │
│        LlmAgent(synthesizer)│
│      ]                      │
│    )                        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. ADK Runner.run()         │
│                             │
│    Sets state["user_query"] │
│    Executes agent tree      │
│    Each agent reads {keys}  │
│    and writes output_key    │
│    Result in final agent's  │
│    output_key               │
└─────────────────────────────┘
```

### Example: Generated Pipeline Tree for RAG + Critic
```
SequentialAgent("pipeline_root")
│
├── LlmAgent("researcher")
│   instruction: "Research: {user_query}"
│   output_key: "research_data"
│   tools: [McpToolset(web-search)]
│
├── ParallelAgent("parallel_analysis")
│   │
│   ├── LlmAgent("technical_analyst")
│   │   instruction: "Analyze technical aspects of: {research_data}"
│   │   output_key: "technical_analysis"
│   │
│   └── LlmAgent("business_analyst")
│       instruction: "Analyze business implications of: {research_data}"
│       output_key: "business_analysis"
│
├── LoopAgent("refinement_loop", max_iterations=3)
│   │
│   ├── LlmAgent("writer")
│   │   instruction: "Write report using {technical_analysis} and
│   │                 {business_analysis}. Previous feedback: {feedback?}"
│   │   output_key: "draft"
│   │
│   └── LlmAgent("critic")
│       instruction: "Review: {draft}. If satisfactory, call exit_loop.
│                     Otherwise provide feedback."
│       output_key: "feedback"
│       tools: [exit_loop]
│
└── LlmAgent("final_editor")
    instruction: "Polish the final draft: {draft}"
    output_key: "final_result"
```

### DAG → Tree Transformation
```
Old AutoMAS DAG:                    New ADK Tree:

    A ──┐                           SequentialAgent
    │   │                           ├── LlmAgent(A)
    ▼   │                           ├── ParallelAgent
    B   │                           │   ├── LlmAgent(B)
   ╱ ╲  │                          │   └── LlmAgent(C)
  C   D  │                          ├── LlmAgent(D)  ← reads B+C from state
  ╲  ╱  │                          └── LlmAgent(E)  ← reads A+D from state
   D    │
   │   │                           Data flow via session.state:
   ▼   ▼                           A.output_key="a_out"
    E                               B.output_key="b_out"
                                    C.output_key="c_out"
Level execution:                    D.instruction="...{b_out}...{c_out}..."
L0: [A]                             D.output_key="d_out"
L1: [B, C]  (parallel)             E.instruction="...{a_out}...{d_out}..."
L2: [D]     (fan-in from B,C)
L3: [E]     (fan-in from A,D)
```

---

## 5. Key Design Decisions

### Q1: Pipeline as ADK tree — how does PipelineBuilder translate?

The old `PipelineBuilder` creates a flat DAG with level-parallel execution. The new builder converts from DAG-like thinking to a **nested tree**:

1. **Linear chain** (A→B→C): `SequentialAgent(sub_agents=[A, B, C])`
2. **Fan-out** (A→[B,C]): `SequentialAgent([A, ParallelAgent([B, C])])`
3. **Fan-in** ([B,C]→D): `SequentialAgent([ParallelAgent([B, C]), D])` — D reads B and C outputs from state
4. **Diamond** (A→[B,C]→D): `SequentialAgent([A, ParallelAgent([B, C]), D])`

**Edge case: Cross-level dependencies** (e.g., A feeds both B and E directly). Handled via `output_key` — A writes to state, E reads it regardless of how many intermediate agents ran. No structural change needed.

The meta-agent generates the tree config directly (not a DAG), eliminating the need for DAG→tree conversion.

### Q2: output_key DAG — preserving data flow semantics

In the old system, `NodeSession.get_input_for_node()` builds a structured JSON with all parent results. In ADK:

- Each agent has a unique `output_key` (e.g., `"researcher_output"`)
- Downstream agents reference `{researcher_output}` in their instruction
- **Fan-in**: Agent instruction simply references multiple state keys: `"Synthesize {analysis_a} and {analysis_b}"`
- **No race conditions in ParallelAgent**: Each parallel sub-agent writes to its own unique output_key. The shared state is safe because keys don't collide.
- **Original query**: Set as `state["user_query"]` before execution. All agents can reference `{user_query}`.

### Q3: LoopAgent for critic-actor

ADK provides native critic-actor via LoopAgent:
```
LoopAgent(
    name="refinement",
    max_iterations=N,
    sub_agents=[
        LlmAgent(name="actor",  output_key="draft", ...),
        LlmAgent(name="critic", output_key="feedback",
                 tools=[exit_loop],
                 instruction="Review {draft}. If good, call exit_loop tool.")
    ]
)
```

The `exit_loop` tool (`google.adk.tools.exit_loop_tool.exit_loop`) sets `event.actions.escalate = True`, causing the LoopAgent to stop. The builder auto-injects `exit_loop` into the last sub-agent of any loop node.

### Q4: Session state safety

- **ParallelAgent**: Creates branch isolation per sub-agent, but they share the same `session.state`. Writes to distinct keys are safe; writes to the same key are a race condition — prevented by schema validation (unique `output_key` per agent).
- **LoopAgent**: Actor overwrites `draft` each iteration, critic overwrites `feedback`. This is intentional — each iteration refines the previous result.

---

## 6. `src/fedotmas/` Structure

```
src/fedotmas/
├── __init__.py                    # Package exports: FedotMAS, AgentConfig, PipelineConfig
├── main.py                       # FedotMAS orchestrator class
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Model defaults, env config (DEFAULT_MODEL, META_AGENT_MODEL, etc.)
│
├── pipeline/
│   ├── __init__.py
│   ├── models.py                 # Pydantic: AgentConfig, PipelineNodeConfig, PipelineConfig
│   ├── builder.py                # PipelineConfig → ADK agent tree
│   └── runner.py                 # Wraps ADK Runner, sets initial state, extracts final result
│
├── meta/
│   ├── __init__.py
│   ├── agent.py                  # Meta-agent: LlmAgent that generates PipelineConfig
│   └── prompts.py                # System prompts for meta-agent
│
└── mcp/
    ├── __init__.py
    ├── registry.py               # MCP server registry: name → McpToolset factory
    └── servers/
        └── download/
            ├── __init__.py
            └── server.py         # Reference MCP server implementation
```

### Module Descriptions

**`config/settings.py`** — Centralized configuration. Default model (configurable per provider), API keys from env vars, MCP server defaults. ADK model format:
- Gemini: `"gemini-2.5-flash"` (native, no extra deps)
- Claude: `"claude-sonnet-4-20250514"` (requires `pip install google-adk[extensions]` or `anthropic`)
- LiteLLM: `"openrouter/anthropic/claude-sonnet-4"` (requires `pip install google-adk[extensions]` or `litellm`)

**`pipeline/models.py`** — Pydantic models for pipeline configuration. These are the data structures that the meta-agent generates and the builder consumes. Includes validation (unique output_keys, valid agent references, valid tree structure).

**`pipeline/builder.py`** — Core builder function: takes `PipelineConfig` + MCP registry, returns ADK `BaseAgent` tree. Recursively builds:
- `"agent"` → `LlmAgent(name, instruction, output_key, tools=[McpToolset(...)])`
- `"sequential"` → `SequentialAgent(sub_agents=[build(child) for child in children])`
- `"parallel"` → `ParallelAgent(sub_agents=[build(child) for child in children])`
- `"loop"` → `LoopAgent(sub_agents=[build(child) for child in children], max_iterations=N)` — auto-injects `exit_loop` tool into last sub-agent

**`pipeline/runner.py`** — Thin wrapper around ADK's `Runner`. Sets `state["user_query"]`, runs the agent tree, extracts result from the final agent's `output_key`.

**`meta/agent.py`** — The meta-agent is itself an `LlmAgent` with `output_schema=PipelineConfig`. Given a user task description + available MCP tools, it generates the pipeline configuration.

**`meta/prompts.py`** — System prompt for the meta-agent. Describes available MCP tools, pipeline node types, and design principles.

**`mcp/registry.py`** — Maps server names to `McpToolset` factory functions. Produces ADK `McpToolset` instances with `StdioConnectionParams`.

**`mcp/servers/download/`** — Reference MCP server implementation. Demonstrates the contract for adding new MCP servers: a FastMCP server with `DESCRIPTION` constant, `@mcp.tool` decorated functions, and `mcp.run(transport="stdio")` entry point.

---

## 7. Human-in-the-Loop

### Level 1: Pipeline Config Review (between meta-agent and builder)

After the meta-agent generates `PipelineConfig`, execution pauses for human review:

```
User query → Meta-Agent → PipelineConfig
                              │
                         ┌────▼────┐
                         │  PAUSE  │  Human reviews/edits config
                         │  Review │  (add/remove agents, change
                         │         │   topology, edit instructions)
                         └────┬────┘
                              │
                    PipelineConfig (possibly edited)
                              │
                         Builder → ADK tree → Runner
```

Implementation in `main.py`:
- `FedotMAS.generate_config(query) → PipelineConfig` — runs meta-agent, returns config
- `FedotMAS.build_and_run(config) → result` — builds and executes
- `FedotMAS.run(query) → result` — convenience method that does both (no review)

### Level 2: Runtime Intervention (Phase 4 — future)

ADK provides `LongRunningFunctionTool` for pausing execution mid-pipeline. A `human_review` tool can be created to gate specific pipeline steps on human approval.

---

## 8. Implementation Phases

### Phase 1: Foundation (complete)
1. `config/settings.py` — Environment config, default model
2. `pipeline/models.py` — Pydantic models with validation
3. `mcp/registry.py` — MCP server registry
4. `mcp/servers/download/server.py` — Reference MCP server

### Phase 2: Builder + Runner (complete)
5. `pipeline/builder.py` — Config → ADK agent tree
6. `pipeline/runner.py` — Runner wrapper with state management

### Phase 3: Meta-Agent (complete)
7. `meta/prompts.py` — Meta-agent system prompts
8. `meta/agent.py` — Meta-agent implementation
9. `main.py` — FedotMAS orchestrator

### Phase 4: Advanced Features (future)
10. Human-in-the-loop runtime intervention via `LongRunningFunctionTool`
11. Enhanced validation and error handling
12. Tracing/observability and cost tracking

### Dependencies
```
Phase 1: models.py ← settings.py
         registry.py ← settings.py
         download/server.py (standalone)
Phase 2: builder.py ← models.py, registry.py
         runner.py ← builder.py
Phase 3: prompts.py ← models.py
         agent.py ← prompts.py, models.py
         main.py ← agent.py, builder.py, runner.py
Phase 4: depends on all above
```
