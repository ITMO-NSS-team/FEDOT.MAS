# Meta-debugger

The meta-debugger is an LLM agent that fixes your pipeline configuration when something goes wrong. It receives the current config and a description of the error, then returns a corrected config. The pipeline is retried with the fix applied.

You do not need to write recovery logic yourself. The `Controller.run_with_recovery()` method handles the entire cycle: run the pipeline, catch the error, call the debugger, retry.

## Quick Start

```python
import asyncio

from fedotmas import MAW
from fedotmas.control import Controller
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction="Research the topic: {user_query}. Use data from {nonexistent}.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            instruction="Write a report based on: {research}",
            output_key="report",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(agent_name="researcher"),
            MAWStepConfig(agent_name="writer"),
        ],
    ),
)


async def main():
    maw = MAW(mcp_servers=[])
    ctrl = Controller(maw)
    result = await ctrl.run_with_recovery(
        "Analyze the electric vehicle market",
        config=CONFIG,
        max_retries=2,
    )
    print(result.status)


asyncio.run(main())
```

The researcher agent references `{nonexistent}`, a variable that does not exist in the pipeline state. This causes a RuntimeError during execution. The meta-debugger reads the error message, understands that the instruction contains a bad variable reference, and rewrites the instruction to remove it. The pipeline is then retried with the corrected config and completes successfully.

## Two Types of Errors

The meta-debugger handles two types of problems.

The first type is framework exceptions. These are Python errors that crash the pipeline, like a RuntimeError when an agent references a variable that does not exist, or when an API call fails with an unexpected response. The framework catches the exception automatically and passes the error message to the debugger.

The second type is logical errors. The pipeline completes without any exception, but the agent's output is wrong. For example, you asked the agent to return the number 5, but it returned 2. There is no crash, so the framework thinks everything is fine. To catch these problems, you write a check function that inspects the output and returns an error message if something is wrong.

## Recovery from stalled retrieval

Research convergence uses new retrieved content as a progress signal. Nonempty
web/DOM reads count even without a search candidate entry; identical extracted
text is deduplicated across tools and URLs, with at most eight fragment credits
per agent. Successful nonempty file downloads also count once per source.
Navigation messages, empty results, and tool error responses do not earn these
credits. Retrieved content still needs factual verification.

After consecutive research turns without progress, broad search and repeated
reading methods are paused. A worker can try an unvisited known candidate or a
different backend for an attempted source, passing the target explicitly. This
allowance is limited to two alternative routes and two distinct tool methods per
route. Changing only a selector or query does not create another method. The
allowance persists in session state and consumes the existing turn/time budget;
it does not override `discovery_only` or other tool restrictions.

File retrieval requires the appropriate tools in the configured MCP server list.
Include `download` alongside `document` when tasks require CSV/ZIP resources:
download the file, then read or extract its local path. Browser navigation alone
does not provide a downloaded file. Recovery cannot create an unavailable tool.

A terminal JSON object with empty result fields and an explicit refusal in its
notes/reason/explanation is `incomplete`, even if its schema uses task-specific
answer keys. Evidence URLs and excerpts do not turn that refusal into an answer.
A later concrete answer can resolve the incomplete status.

## Programmatic Checks

### Preserving identities across handoffs

When an agent changes a collection name, declare how its input identity paths
map to its output. For example, an inspector can consume `people[].id` and
produce `web_counts[].id`:

```python
from fedotmas import ArtifactContract, ArtifactRequirement, IdentityFieldMapping

requirement = ArtifactRequirement(
    source_key="people",
    required_fields=["people"],
    identity_fields=["people[].id"],
    identity_field_mapping=[
        IdentityFieldMapping(
            source_path="people[].id", output_path="web_counts[].id"
        )
    ],
)
output_contract = ArtifactContract(
    required_fields=["web_counts"], identity_fields=["web_counts[].id"]
)
```

The inspector's `input_requirements` contains `requirement`; its
`output_contract` is the contract above. IDs must remain in their records and
retain their original order. The runtime checks every ID and never fills
missing record IDs by list position. A mapping changes paths, not values or
cardinality. Without a mapping, identity paths retain their original names.

`repair_agent_configs` preserves these routes. Relocating an output path during
recovery requires a matching `output_field_renames` entry, updated output
contract, and updated input mapping. Input-path renames remain separate and
require `input_field_renames`. Adding inspection tools also requires changing
`research_mode` when the role was `discovery_only`; tools alone do not override
that mode.

Every MAW worker receives the original caller task at model-request time,
including its requested units and answer scope. `user_query` is reserved for
that task and cannot be an agent output key; a checkpoint's `user_query` cannot
override the current request. This preserves task context, but does not prove
that a model's answer or chosen evidence is correct. Use task-specific checks
for those properties.

### Source obligations during generation and recovery

An extraction agent can declare `evidence_requirements` separately from its
editable instruction. For example:

```python
from fedotmas import EvidenceRequirement

equation_source = EvidenceRequirement(
    task_quote="Use the equation from https://example.org/paper",
    source="https://example.org/paper",
    source_kind="url",
    source_field="source_url",
    required_fields=["formula", "units", "variable_bindings"],
)
```

Put this object in the reader's `evidence_requirements`. Its
`output_contract.required_fields` must include `source_url`, `formula`, `units`
and `variable_bindings`. Generation checks that the quote occurs in the original
task, the reader has inspection tools, and downstream consumers explicitly
require these fields after the reader executes. Explicit `Host file path:` task
metadata must also have a file evidence requirement. Archive contents should be
inventoried before choosing their roles or assuming filenames and XML tags.

The runtime checks that the artifact's `source_field` points to an executed,
nonempty read. A literal URL/file obligation must match that source, including
recorded download-to-file and archive-to-member links. Navigation, search
snippets, errors and tool calls written in model prose do not fulfill the
obligation. Missing source evidence replaces the output with an unresolved
marker and stops dependent agents. A later successful producer run can resolve
the dependency. Keep structured extraction separate from final answer formatting.

Instruction repair retains these obligations; replacing the agent/configuration
cannot remove them or drop the required fields and reading capability. They are
also appended at model-request time independently of the repaired instruction.
For `source_kind="reference"`, `source` is the task's named source and
`source_field` contains its resolved URL/path. The runtime checks that locator
was read; it does **not** establish the semantic identity of the publication,
correctness of a formula, units or every extracted value. Arbitrary named source
requirements must be declared by the planner; they are not inferred reliably
from all natural-language task descriptions. Existing configurations with no
`evidence_requirements` remain compatible and do not acquire them automatically.

During generation, a missing `identity_field_mapping` can use one targeted retry
within the existing retry budget. It may only add the missing mappings; the
instructions, tools, contracts and pipeline must remain unchanged. The runtime
does not guess that two identities are equivalent from their leaf names. An
invalid or broader rewrite fails validation instead of weakening the contract.

### Checking result values

A check function receives the full pipeline state as a dictionary and returns either an error message (a string describing what went wrong) or `None` if the output is correct. The error message is passed directly to the meta-debugger, so write it as an instruction that explains what needs to be fixed.

```python
def check_number(state: dict) -> str | None:
    output = str(state.get("number", ""))
    if "5" not in output:
        return f"Agent returned {output!r}, but should return 5. Fix the instruction."
    return None


result = await ctrl.run_with_recovery(
    "Return the number 5",
    config=CONFIG,
    max_retries=2,
    checks={"calculator": check_number},
)
```

The `checks` parameter is a dictionary where keys are agent names and values are check functions. When the agent named `calculator` finishes, the check function runs. If it returns a string, the pipeline stops and the meta-debugger receives that string as the problem description. The debugger then rewrites the agent's instruction and retries the pipeline.

You can add checks for multiple agents at once. Each check runs only after its corresponding agent completes, so if the first agent produces a bad output, the pipeline stops early without running the remaining agents.

## Error Hint

Sometimes the exception message alone is not enough for the debugger to understand what went wrong. The `error_hint` parameter lets you provide additional context.

```python
result = await ctrl.run_with_recovery(
    "Analyze the EV market",
    config=CONFIG,
    max_retries=2,
    error_hint="The researcher agent may reference variables that don't exist in state",
)
```

The hint is appended to the debugger's instruction as extra context. It does not replace checks. Checks detect the problem, the hint explains the context. You can use them together or separately.

## Custom Fix Tools

By default, the meta-debugger has one tool: `fix_instruction`, which rewrites the instruction of a specific agent. You can give the debugger additional tools through the `fix_tools` parameter.

A fix tool is an async function that reads and writes the config through `tool_context.state["config"]`. The debugger decides which tools to call based on the error description.

```python
async def fix_model(tool_context, agent_name: str, new_model: str) -> str:
    """Switch the agent to a different model."""
    config = MAWConfig.model_validate_json(tool_context.state["config"])
    agent = next((a for a in config.agents if a.name == agent_name), None)
    if agent is None:
        return f"Error: agent '{agent_name}' not found"
    updated = agent.model_copy(update={"model": new_model})
    config = config.replace_agent(agent_name, updated)
    tool_context.state["config"] = config.model_dump_json()
    return f"Switched {agent_name} to {new_model}"


result = await ctrl.run_with_recovery(
    "Analyze the EV market",
    config=CONFIG,
    fix_tools=[fix_model],
)
```

The default `fix_instruction` tool is replaced when you pass `fix_tools`. If you want to keep it alongside your custom tools, include it explicitly.

## Final answers and explicit abstention

When a caller supplies `final_answer_contract`, a terminal agent without tools
answers the original user query using its referenced upstream artifacts. The
runtime supplies this answer instruction instead of the generated presentation
plan: a plan requesting a country and a person's name must not add a country to
an answer when the user requested only the person. References and declared input
requirements still supply evidence, and missing inputs still produce diagnostics.
Input identities select the evidence; they do not mandate extra output fields.
This does not add a model call or modify the saved configuration. Without a caller
contract, the configured instruction is used as before. Terminal agents that
still need tools retain their working instructions.

A terminal `<abstain>reason</abstain>` is recognized even after explanatory prose,
or inside an erroneous `<solution>` wrapper. It makes the pipeline `incomplete`
and remains available to recovery; it is not scored as a successful answer. A
later concrete `<solution>` can supersede an earlier abstention. These checks do
not establish factual correctness or guarantee that a model follows the requested
format; evaluate that separately on a new run.

## When to Use the Optimizer Instead

The meta-debugger fixes specific errors. It takes one config, receives one error description, and produces one corrected config. This is the right tool when an agent crashes or fails a simple check.

If you need to evaluate quality on a scale, compare multiple config variants, or use an ensemble of judges, that is a different problem. The [Optimizer](optimizer.md) runs an evolutionary loop over many iterations, scores each candidate with a Scorer or LLMJudge, and converges on the best config across a training set of tasks.

Use the debugger for recovery from errors. Use the optimizer for improving quality.
