# Synapse export

A generated configuration does not have to run here. `fedotmas.export` turns a `MAWConfig` into the JSON that the Synapse platform imports through its configuration bundle endpoint, so the agents, prompts, tool assignments and the workflow between them are handed over as an artifact and executed by their engine.

Nothing in this module builds or runs anything. Generation and execution are separate steps here, and the export replaces the second one with their runtime.

## Quick start

```python
import asyncio

from fedotmas import MAW
from fedotmas.export import to_synapse_bundle

CATALOG = {"urbanprojects.getprojectbyid": "Get a Prostor project by id"}


async def main():
    maw = MAW(tool_catalog=CATALOG)
    config = await maw.generate_config("Assess school provision in a district")

    export = to_synapse_bundle(
        config,
        workflow_id="school_provisioning",
        workflow_name="School provisioning",
        tool_catalog=CATALOG,
    )
    print(export.bundle["items"]["workflows"][0]["nodes"])
    print(export.linearized_branches, export.unresolved_tools)


asyncio.run(main())
```

`tool_catalog` on `MAW` is what generation offers the meta-agent; the same catalogue passed to `to_synapse_bundle` is what the export is checked against. Give it their tenant's tool ids and descriptions, and the agents come out referencing tools that exist on the other side. Leave it out and the agents come out with no tools at all.

An instance built with a `tool_catalog` cannot `build()` the config it produced: the tools belong to another runtime, so the configuration is for export only.

## Agents

Their agent record and `MAWAgentConfig` line up almost field for field.

| FEDOT.MAS | Synapse | Note |
| --- | --- | --- |
| `name` | `_id`, `name`, `type` | Identity is the wire name, matching `^[a-z][a-z0-9_]{1,63}$`. `type` is an auction kind, unused here because every node names its agent. |
| — | `display_name` | The original agent name, when it differs from the wire name. |
| `instruction` | `system_prompt` | The optional-state marker `{key?}` is normalized to `{key}`. |
| `model` | `model` | Their identifiers (Bifrost). Ours require a provider prefix, which theirs already have. |
| `tools` | `allowed_tools` | Ids outside the catalogue are dropped and reported in `unresolved_tools`. |
| `output_key` | `output_save_key` | Required here, nullable there. |
| — | `description` | Derived from the instruction; their UI shows it and we have no field for it. |
| — | `agent_class` | Always `GenericAgent`. |
| — | `temperature` | Ours is set per meta-agent call, not per agent. Defaults to `1.0`, overridable. |
| — | `allowed_phases` | The `phase_label` of the node the agent sits on. |
| — | `eval_keywords` | Empty: only an auction reads them. |
| — | `allowed_delegation_targets` | `null`. The graph is explicit, so no agent delegates. |
| `max_output_tokens` | — | No counterpart; dropped. |

## Topology

Ours is a tree of `agent` / `sequential` / `parallel` / `loop`. Theirs is a flat `nodes` plus `edges` list with types `start`, `phase`, `approval_gate`, `execution`, `deploy`, `a2a_agent`, `validator`, `end`.

| FEDOT.MAS | Synapse | Note |
| --- | --- | --- |
| `agent` | `phase` | `agent_selection: "direct"` with `agent_type` naming the agent, and `writes` carrying its `output_key`. |
| `sequential` | edges | A chain, plus the single `start` and `end` their validation requires. |
| `loop` | `validator` + back edge | The body becomes phase nodes, followed by a validator that sends `approved` forward and `rejected` back to the first node of the body. |
| `max_iterations` | `max_reject_retries` | Ours counts passes, theirs rejections after the first. Capped at 10. |
| `parallel` | — | Not expressible. Branches are pulled into the chain and counted in `linearized_branches`. |

Their engine keeps one current node and picks exactly one successor, so a fan-out that rejoins has no representation at all. The export flattens it and counts what it flattened rather than failing, but a bundle with a non-zero `linearized_branches` is running a different plan from the one that was generated.

Their `validator` is not an LLM. Its checks are structural (presence, type, length) or a JSON schema, so an LLM critic stays an ordinary phase that writes its verdict into the state, and the validator gates on that verdict's key. Converting a critic straight into a validator gives a workflow that cannot work.

## Data flow

In a `MAWConfig` the connection between agents is implicit: an agent writes under its `output_key`, and the next one reads it by mentioning `{output_key}` in its instruction. Synapse declares reads on the node instead. Filling their `reads` accurately would mean mining those references out of prompt text, so the export sets `default_reads: ["*"]` on the workflow, which is what their own hand-built bundle does.

## Reusing agents the platform already has

When the config was generated over a caller-supplied pool (`MAW.generate_config(task, existing_agents=...)`), pass that pool to the export as well. An entry's `id` is the identifier the agent already has on their side, and the import matches on it, so the record is updated rather than duplicated:

```python
export = to_synapse_bundle(config, workflow_id="flow", existing_agents=pool)
export.reused_agents   # ids carried over verbatim
export.renamed_ids     # ids their format cannot carry, as (given, emitted)
```

An id that does not match their wire-name pattern cannot be kept. It is slugified so the bundle still imports, and reported in `renamed_ids` with a warning: such an agent arrives as a new record rather than an update.

`id` is the caller's field alone. `AgentPoolConfig` doubles as the meta-agent's output schema, so a generated pool can come back with the field filled in; `PoolGenerator` clears it, since an invented id would point the import at somebody's existing record.

## Checking a bundle without their tenant

Synapse validates a workflow when it is saved: one `start` and one `end`, every edge endpoint present, `agent_type` set on direct phase nodes, a `rejected` edge out of every validator, and no cycles once those back edges are excluded. Those rules are ported into `packages/fedotmas/tests/export/test_synapse_bundle_rules.py`, so an emitted bundle can be checked here.

What the port cannot check is whether tool ids and model names exist in a particular tenant. That needs their catalogue, which is also what decides whether the exported agents can do anything at all.
