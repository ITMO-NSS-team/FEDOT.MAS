# MAW config patterns

All model names include a provider prefix because FEDOT.MAS validates that
invariant. `host/...` is a descriptive label for bridge mode, not an API
provider.

## Sequential specialist and finalizer

```json
{
  "agents": [
    {
      "name": "specialist",
      "instruction": "Analyze {user_query?} and record the evidence.",
      "model": "host/model-id",
      "output_key": "evidence",
      "tools": []
    },
    {
      "name": "finalizer",
      "instruction": "Answer {user_query?} using {evidence?}. Verify the requested output format.",
      "model": "host/model-id",
      "output_key": "final_answer",
      "tools": []
    }
  ],
  "pipeline": {
    "type": "sequential",
    "children": [
      {"type": "agent", "agent_name": "specialist"},
      {"type": "agent", "agent_name": "finalizer"}
    ]
  }
}
```

## Parallel independent analyses and synthesis

Use a sequential root whose first child is parallel and whose second child is a
finalizer. Parallel workers must have different output keys.

```json
{
  "agents": [
    {
      "name": "analyst_a",
      "instruction": "Solve {user_query?} independently and cite decisive evidence.",
      "model": "host/model-id",
      "output_key": "analysis_a",
      "tools": []
    },
    {
      "name": "analyst_b",
      "instruction": "Look for counterexamples and solve {user_query?} independently.",
      "model": "host/model-id",
      "output_key": "analysis_b",
      "tools": []
    },
    {
      "name": "synthesizer",
      "instruction": "Resolve {analysis_a?} and {analysis_b?}; return the verified answer to {user_query?}.",
      "model": "host/model-id",
      "output_key": "final_answer",
      "tools": []
    }
  ],
  "pipeline": {
    "type": "sequential",
    "children": [
      {
        "type": "parallel",
        "children": [
          {"type": "agent", "agent_name": "analyst_a"},
          {"type": "agent", "agent_name": "analyst_b"}
        ]
      },
      {"type": "agent", "agent_name": "synthesizer"}
    ]
  }
}
```

Use loops only when an agent has an objective exit condition. Set
`max_iterations` even when the reviewer is instructed to stop early.
