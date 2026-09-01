# Host bridge

Use this mode only when FEDOT.MAS cannot call the requested model through an API,
but the current Codex or Claude host can create a subagent with that exact model.
The bridge supports `MAW` pipelines; use native provider execution for dynamic
`MAS` routing that depends on model-generated function calls.

## Protocol

1. Start `scripts/fedot_host_bridge.py` with a validated config, task, queue
   directory, and result path. Keep the process running.
2. When it prints `FEDOT_BRIDGE_REQUEST`, read the named request JSON. Its
   `prompt` field is the complete worker input; `agent_name` and `model` identify
   the role and requested model label.
3. Spawn one isolated host-native subagent with that prompt. In Codex, use the
   available subagent delegation primitive with no inherited conversation when
   possible. In Claude, use the available Agent/Task primitive. Pass the exact
   requested model when the host exposes model selection.
4. Write a response JSON to the `response_path` named in the request:

   ```json
   {
     "request_id": "same id as the request",
     "text": "raw subagent response",
     "elapsed_seconds": 12.34
   }
   ```

5. Continue until the bridge exits and writes its result JSON. Parallel FEDOT
   branches may emit several requests; dispatch them concurrently only when they
   do not share mutable files or other conflicting state.

Use an isolated working directory for benchmark arms and coding workers. A
worker may read the request file itself only when the file contains no hidden
answer or unrelated artifacts; otherwise pass the `prompt` text directly.

## Failure handling

- If a subagent fails, write `{"request_id":"...","error":"..."}` so the
  bridge fails explicitly instead of waiting forever.
- Do not fabricate a response after timeout. Preserve the queue and report the
  incomplete stage.
- Cap loops and total calls before launch. Stop if the bridge requests a model
  different from the one the user selected.
