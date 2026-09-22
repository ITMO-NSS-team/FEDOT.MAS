# Codex subscription run

This directory contains the completed host-native execution using exactly
`gpt-5.6-sol` with `xhigh` reasoning.

The FEDOT.MAS `config.json` remains the workflow authority: the first
`rubber_recipe_master` turn selected workers from that config, each worker
produced its configured state key, and the final master turn synthesized
`final_report`. This is the `MAS` master-orchestrator pattern, not a fixed
`MAW` workflow.

Authentication uses the Codex/ChatGPT subscription available to the host.
It is not an OpenAI API key and therefore cannot be injected into FEDOT.MAS's
LiteLLM client. The execution is accurately labeled `Codex host-native
subscription`; `native_litellm_api_run` is `false` in the manifest.

Files:

- `001_routing.json`: coordinator routing decision and worker requests.
- `*_output.md`: worker outputs mapped to the FEDOT.MAS state keys.
- `final_report.md`: final coordinator output (`final_report`).
- `final_state.json`: assembled FEDOT-style session state.
- `run_manifest.json`: ordered calls, exact model, reasoning level, and
  character counts.
- `verification.json`: structural and content verification result.

