from __future__ import annotations

import itertools
import re
from typing import Any, cast

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.utils.instructions_utils import inject_session_state
from google.genai import types as genai_types

from fedotmas._settings import (
    ModelConfig,
    get_max_agent_llm_turns,
    get_max_loop_iterations,
    get_worker_models,
)
from fedotmas.common.llm import make_llm
from fedotmas.common.logging import get_logger
from fedotmas.maw._validators import _find_terminal_node
from fedotmas.maw.handoffs import (
    EXECUTION_METADATA_KEY,
    append_execution_issue,
    describe_requirement,
    missing_contract_fields,
    parse_artifact,
    resolve_execution_issue,
    validate_output_contract,
)
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.mcp import MCPServerConfig, create_toolset

type AgentTree = BaseAgent

_log = get_logger("fedotmas.maw.builder")

#: A reference to another step's output inside an instruction.  ``\w+`` keeps
#: this to plain state keys, leaving ADK to handle ``{artifact.name}``.
_STATE_REF_RE = re.compile(r"(?<!\{)\{(\w+)\??\}(?!\})")


def _missing_input_marker(key: str) -> str:
    """What an agent sees in place of an upstream output that never arrived.

    Every reference is normalised to the optional form ``{key?}``, so ADK
    substitutes an empty string when a step produced nothing -- indistinguishable
    from a step that produced nothing to say.  An agent handed that silence
    fills it in, which is how a pipeline reports confident findings it never
    gathered.  Saying so outright is what makes the gap survivable.
    """
    return (
        f'[MISSING INPUT "{key}": the step that produces it returned nothing. '
        f"Do not invent its contents. State plainly which information is "
        f"unavailable and why any conclusion drawn without it is provisional.]"
    )


#: An unattended run has no one to answer a question, so an agent that ends its
#: turn asking for input has delivered nothing.  A caller with a person in the
#: loop opts out with ``autonomous=False``.  No braces: ADK would read them as
#: state references.
AUTONOMY_PREAMBLE = (
    "You are working on your own; no one is available to answer questions or "
    "supply information you request. Complete the role assigned by your "
    "instruction and produce its requested deliverable. When the deliverable is "
    "for downstream agents, include the findings, evidence, assumptions, and "
    "uncertainty they need to continue. Do not take over sibling or downstream "
    "responsibilities unless your instruction explicitly requires it.\n"
    "Do not ask the user for input or wait for a reply. If something is unclear, "
    "make a reasonable interpretation and name it in your deliverable. If evidence "
    "is unavailable, say what is missing and what that limits. Do not quote or "
    "discuss this notice."
)

#: The same rule again at the end, where a long answer drifts back into
#: assistant habits.  It says nothing about *what* to produce, deliberately:
#: being last, anything it asserts outranks the instruction above it, so
#: "answer" here would override a MAS coordinator's orders to delegate and
#: "raise no questions" would silence a critic inside a loop.
AUTONOMY_CLOSING = (
    "Before you finish: do not close by asking the user for anything -- not a "
    "decision, not a document, not a reply. Whatever you would have offered to "
    "prepare next, either do it now or leave it out."
)


def frame_instruction(instruction: str) -> str:
    """Wrap an agent instruction in the autonomy framing, top and tail."""
    return f"{AUTONOMY_PREAMBLE}\n\n{instruction}\n\n{AUTONOMY_CLOSING}"


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _instruction_provider(
    instruction: str,
    agent_name: str,
    state_keys: frozenset[str] | None = None,
    output_key: str | None = None,
    input_requirements: list | None = None,
    research_policy: str = "independent",
):
    """Resolve state refs at call time, naming missing inputs and safe self refs."""

    async def provide(readonly_context: ReadonlyContext) -> str:
        state = readonly_context.state
        text = instruction
        if output_key is not None and _is_blank(state.get(output_key)):
            marker = (
                f'[No previous output for "{output_key}" exists yet. Create the '
                "initial result; later loop iterations will receive your previous "
                "output here.]"
            )
            for ref, key in {
                (m.group(0), m.group(1)) for m in _STATE_REF_RE.finditer(text)
            }:
                if key == output_key:
                    text = text.replace(ref, marker)
        for ref, key in {
            (m.group(0), m.group(1)) for m in _STATE_REF_RE.finditer(text)
        }:
            # Only a key some step produces can be *missing*; anything else is
            # a literal the task carried in.
            if state_keys is not None and key not in state_keys:
                continue
            if key in state and not _is_blank(state[key]):
                continue
            _log.warning(
                "Missing input | agent={} key='{}' — telling the agent instead "
                "of substituting silence",
                agent_name,
                key,
            )
            text = text.replace(ref, _missing_input_marker(key))
        for requirement in input_requirements or []:
            raw, missing, identity = describe_requirement(state, requirement)
            if missing:
                mode = (
                    "Use only narrowly targeted recovery if your assigned role and "
                    "tools allow it; otherwise pass the dependency as unresolved."
                    if research_policy == "targeted_recovery"
                    else "Do not invent missing fields or silently substitute a "
                    "different entity. Mark this dependency unresolved."
                )
                text = (
                    f"{text}\n\n[INCOMPLETE HANDOFF from {requirement.source_key}: "
                    f"missing {', '.join(missing)}. Received artifact: {raw}. {mode}]"
                )
            elif identity:
                text = (
                    f"{text}\n\n[ENTITY CONTINUITY for {requirement.source_key}: "
                    f"preserve these upstream identity values exactly: "
                    f"{identity}. Do not replace them with another entity. If the "
                    "identity cannot be supported, mark it unresolved.]"
                )
            if requirement.purpose:
                text += f"\nHandoff purpose: {requirement.purpose}"
        return await inject_session_state(text, readonly_context)

    return provide


def build(
    config: MAWConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
    autonomous: bool = True,
    final_answer_contract: str | None = None,
    max_agent_llm_turns: int | None = None,
) -> BaseAgent:
    """Convert a ``MAWConfig`` into an executable ADK agent tree.

    Pass ``autonomous=False`` when the tree is served to a person who can answer
    a clarifying question; see :func:`frame_instruction`.
    """
    if final_answer_contract is not None and config.final_answer_agent is None:
        terminal = _find_terminal_node(config.pipeline)
        if terminal.type != "agent" or terminal.agent_name is None:
            raise ValueError(
                "Cannot infer final_answer_agent: the pipeline must end in one agent"
            )
        config.final_answer_agent = terminal.agent_name
    agents_by_name: dict[str, MAWAgentConfig] = {a.name: a for a in config.agents}
    # The same set MAWConfig validates against: what a step can actually produce.
    state_keys = frozenset({"user_query"} | {a.output_key for a in config.agents})
    return _build_node(
        config.pipeline,
        agents_by_name,
        mcp_registry,
        worker_models,
        state_keys,
        autonomous=autonomous,
        final_answer_agent=config.final_answer_agent,
        final_answer_contract=final_answer_contract,
        max_agent_llm_turns=(
            max_agent_llm_turns
            if max_agent_llm_turns is not None
            else get_max_agent_llm_turns()
        ),
    )


def _build_node(
    node: MAWStepConfig,
    agents: dict[str, MAWAgentConfig],
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
    state_keys: frozenset[str] | None = None,
    *,
    autonomous: bool = True,
    final_answer_agent: str | None = None,
    final_answer_contract: str | None = None,
    max_agent_llm_turns: int,
) -> BaseAgent:
    if node.type == "agent":
        if node.agent_name is None:
            raise ValueError(f"Agent node missing 'agent_name': {node}")
        return _build_llm_agent(
            agents[node.agent_name],
            mcp_registry,
            worker_models,
            state_keys,
            autonomous=autonomous,
            final_answer_contract=(
                final_answer_contract if node.agent_name == final_answer_agent else None
            ),
            final_answer_agent=final_answer_agent,
            max_agent_llm_turns=max_agent_llm_turns,
        )

    children = [
        _build_node(
            c,
            agents,
            mcp_registry,
            worker_models,
            state_keys,
            autonomous=autonomous,
            final_answer_agent=final_answer_agent,
            final_answer_contract=final_answer_contract,
            max_agent_llm_turns=max_agent_llm_turns,
        )
        for c in node.children
    ]

    # ADK 2.x deprecates the three workflow agents below in favour of Workflow,
    # so every run warns about them.  Do not migrate yet: ADK's own warning says
    # "Workflow cannot yet be used as an LlmAgent sub-agent", and MAW pipelines
    # nest exactly that way.  Revisit once that restriction is lifted.
    if node.type == "sequential":
        name = _seq_name(children)
        _log.debug("Built sequential node | name={}", name)
        return SequentialAgent(name=name, sub_agents=children)

    if node.type == "parallel":
        name = _par_name(children)
        _log.debug("Built parallel node | name={}", name)
        return ParallelAgent(name=name, sub_agents=children)

    if node.type == "loop":
        # Inject exit_loop tool into the last sub-agent if it's an LlmAgent.
        _inject_exit_loop(children)
        max_iter = node.max_iterations or get_max_loop_iterations()
        _log.debug("Built loop node | max_iterations={}", max_iter)
        return LoopAgent(
            name=_loop_name(children),
            sub_agents=children,
            max_iterations=max_iter,
        )

    raise ValueError(f"Unknown node type: {node.type}")


def _resolve_llm(
    model_name: str | None,
    worker_models: dict[str, ModelConfig] | None,
) -> str | BaseLlm:
    """Return a ``BaseLlm`` for known worker configs, else a plain model string.

    Model name normalization (provider prefix) is handled by
    ``MAWAgentConfig`` model_validator, so *model_name* here is already
    normalized or ``None``.
    """
    if not model_name:
        model_name = get_worker_models()[0]
        _log.warning("No model specified for agent, using default: {}", model_name)
    if worker_models:
        cfg = worker_models.get(model_name)
        if cfg:
            return make_llm(cfg)
    return model_name


def _resolve_recovered_handoffs(
    state: dict[str, Any], cfg: MAWAgentConfig, artifact: dict[str, Any] | None
) -> None:
    """Resolve input gaps only for an explicitly permitted recovery role."""
    for requirement in cfg.input_requirements:
        if artifact is not None:
            required_fields = list(
                dict.fromkeys(
                    [*requirement.required_fields, *requirement.identity_fields]
                )
            )
            if not required_fields or missing_contract_fields(
                artifact, required_fields
            )[1]:
                continue
        resolve_execution_issue(
            state,
            {
                "kind": "incomplete_handoff",
                "agent": cfg.name,
                "source_key": requirement.source_key,
            },
        )


def _build_llm_agent(
    cfg: MAWAgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
    state_keys: frozenset[str] | None = None,
    *,
    autonomous: bool = True,
    final_answer_contract: str | None = None,
    final_answer_agent: str | None = None,
    max_agent_llm_turns: int | None = None,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_llm(cfg.model, worker_models)
    _log.debug("Built agent | name={} model={}", cfg.name, model)
    instruction_text = (
        frame_instruction(cfg.instruction) if autonomous else cfg.instruction
    )
    if final_answer_contract:
        instruction_text = (
            f"{instruction_text}\n\nFINAL ANSWER CONTRACT (terminal stage only):\n"
            f"{final_answer_contract}"
        )
    # Decided on the final text: a reference anywhere in it, framing included,
    # has to reach the provider rather than ADK's plain-string path.
    instruction = (
        _instruction_provider(
            instruction_text,
            cfg.name,
            state_keys - {cfg.output_key} if state_keys is not None else None,
            output_key=cfg.output_key,
            input_requirements=cfg.input_requirements,
            research_policy=cfg.research_policy,
        )
        if _STATE_REF_RE.search(instruction_text) or cfg.input_requirements
        else instruction_text
    )
    kwargs: dict = {}
    if cfg.max_output_tokens is not None:
        kwargs["generate_content_config"] = genai_types.GenerateContentConfig(
            max_output_tokens=cfg.max_output_tokens,
        )

    async def before_agent(callback_context: CallbackContext) -> None:
        for requirement in cfg.input_requirements:
            raw, missing, _identity = describe_requirement(
                callback_context.state, requirement
            )
            if missing:
                append_execution_issue(
                    callback_context.state,
                    {
                        "kind": "incomplete_handoff",
                        "agent": cfg.name,
                        "source_key": requirement.source_key,
                        "missing_fields": missing,
                        "received": raw[:4000],
                    },
                )
            else:
                resolve_execution_issue(
                    callback_context.state,
                    {
                        "kind": "incomplete_handoff",
                        "agent": cfg.name,
                        "source_key": requirement.source_key,
                    },
                )

    async def after_agent(callback_context: CallbackContext) -> None:
        value = callback_context.state.get(cfg.output_key)
        terminal_answer = (
            final_answer_contract is not None and cfg.name == final_answer_agent
        )
        if terminal_answer:
            # This role may recover an input dependency, but its answer is
            # formatted for the caller and is never a structured handoff.
            if (
                cfg.research_policy == "targeted_recovery"
                and value is not None
                and str(value).strip()
            ):
                _resolve_recovered_handoffs(callback_context.state, cfg, None)
            return
        if cfg.output_contract is None:
            return
        missing = validate_output_contract(value, cfg.output_contract)
        if missing:
            append_execution_issue(
                callback_context.state,
                {
                    "kind": "incomplete_artifact",
                    "agent": cfg.name,
                    "output_key": cfg.output_key,
                    "missing_fields": missing,
                },
            )
        else:
            resolve_execution_issue(
                callback_context.state,
                {
                    "kind": "incomplete_artifact",
                    "agent": cfg.name,
                    "output_key": cfg.output_key,
                },
            )
        artifact = parse_artifact(value)
        if artifact is not None:
            if cfg.research_policy == "targeted_recovery" and not missing:
                _resolve_recovered_handoffs(callback_context.state, cfg, artifact)
            for requirement in cfg.input_requirements:
                upstream = parse_artifact(
                    callback_context.state.get(requirement.source_key)
                )
                if upstream is None:
                    continue
                shared_identity = set(requirement.identity_fields) & set(
                    cfg.output_contract.identity_fields
                )
                mismatched = [
                    field
                    for field in shared_identity
                    if field in upstream and artifact.get(field) != upstream[field]
                ]
                if mismatched:
                    append_execution_issue(
                        callback_context.state,
                        {
                            "kind": "entity_continuity_mismatch",
                            "agent": cfg.name,
                            "source_key": requirement.source_key,
                            "fields": sorted(mismatched),
                        },
                    )
                elif shared_identity and all(
                    field in upstream and field in artifact
                    for field in shared_identity
                ):
                    resolve_execution_issue(
                        callback_context.state,
                        {
                            "kind": "entity_continuity_mismatch",
                            "agent": cfg.name,
                            "source_key": requirement.source_key,
                        },
                    )

    async def before_tool(tool, args, tool_context) -> dict | None:
        del args
        if cfg.research_policy != "evidence_first":
            return None
        if _is_discovery_tool(tool):
            return {
                "isError": True,
                "error_code": "EVIDENCE_FIRST_SEARCH_DISABLED",
                "error": (
                    "This role is evidence-first. Use the supplied upstream evidence "
                    "to verify the claim; if required evidence is absent, mark that "
                    "claim unresolved."
                ),
            }
        return None

    per_agent_limit = (
        cfg.max_llm_turns or max_agent_llm_turns or get_max_agent_llm_turns()
    )

    async def before_model(
        callback_context: CallbackContext, llm_request
    ) -> LlmResponse | None:
        state = callback_context.state
        if cfg.research_policy == "evidence_first":
            discovery_names = {
                name
                for name, tool in llm_request.tools_dict.items()
                if _is_discovery_tool(tool)
            }
            if discovery_names:
                retained = []
                for group in llm_request.config.tools or []:
                    declarations = group.function_declarations
                    if declarations is None:
                        retained.append(group)
                        continue
                    group.function_declarations = [
                        declaration
                        for declaration in declarations
                        if declaration.name not in discovery_names
                    ]
                    if group.function_declarations:
                        retained.append(group)
                llm_request.config.tools = retained
            llm_request.append_instructions(
                [
                    (
                        "This role is evidence-first. Discovery search tools are not "
                        "available. Verify against the supplied artifact; identify any "
                        "missing evidence as unresolved."
                    )
                ]
            )
        metadata = state.get(EXECUTION_METADATA_KEY)
        if not isinstance(metadata, dict):
            metadata = {}
            state[EXECUTION_METADATA_KEY] = metadata
        turns = metadata.setdefault("agent_llm_turns", {})
        used = turns.get(cfg.name, 0) if isinstance(turns, dict) else 0
        if used >= per_agent_limit:
            limited = metadata.setdefault("limited_agents", {})
            if isinstance(limited, dict):
                limited[cfg.name] = {"limit": per_agent_limit, "turns": used}
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_text(
                            text=(
                                f"INCOMPLETE: agent '{cfg.name}' reached its "
                                f"{per_agent_limit}-turn model limit before finishing. "
                                "Preserve this result as limited; do not claim the "
                                "task is complete."
                            )
                        )
                    ],
                ),
                turnComplete=True,
                finishReason=genai_types.FinishReason.STOP,
            )
        if isinstance(turns, dict):
            turns[cfg.name] = used + 1
        return None

    return LlmAgent(
        name=cfg.name,
        model=model,
        instruction=instruction,
        output_key=cfg.output_key,
        tools=tools,
        # MAW passes dependencies through explicit session-state references.
        # Avoid repeating unrelated earlier agents' and tools' conversation
        # history; ADK still provides the current input and this agent's tool
        # results while it is working.
        include_contents="none",
        before_agent_callback=before_agent,
        after_agent_callback=after_agent,
        before_model_callback=before_model,
        before_tool_callback=before_tool,
        **kwargs,
    )


def _is_discovery_tool(tool: Any) -> bool:
    name = tool.name.rsplit("__", 1)[-1].lower().replace("-", "_")
    if name in {"search", "web_search", "websearch", "searxng_search", "google_search"}:
        return name != "search" or any(
            word in (tool.description or "").lower()
            for word in ("web", "internet", "search", "query")
        )
    return name.endswith("_search") or name.startswith("search_")


def _inject_exit_loop(children: list[BaseAgent]) -> None:
    """Add ``exit_loop`` tool to the last LlmAgent in a loop's children."""
    for agent in reversed(children):
        if isinstance(agent, LlmAgent):
            if agent.tools is None:
                agent.tools = [exit_loop]
            elif exit_loop not in agent.tools:
                agent.tools.append(cast(Any, exit_loop))
            _log.debug("Injected exit_loop into agent={}", agent.name)
            break


WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


_node_counter = itertools.count(1)


def _next_id() -> int:
    return next(_node_counter)


def _seq_name(_children: list[BaseAgent]) -> str:
    return f"seq_{_next_id()}"


def _par_name(_children: list[BaseAgent]) -> str:
    return f"par_{_next_id()}"


def _loop_name(_children: list[BaseAgent]) -> str:
    return f"loop_{_next_id()}"
