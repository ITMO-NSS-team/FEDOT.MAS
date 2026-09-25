from __future__ import annotations

import itertools
import json
import re
from typing import Any, cast

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LLMRegistry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
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
from fedotmas.maw._validators import _find_terminal_node, agent_executes_in_loop
from fedotmas.maw.handoffs import (
    ABSTENTION_STATE_KEY,
    EXECUTION_METADATA_KEY,
    append_execution_issue,
    describe_requirement,
    field_value,
    field_values,
    is_explicit_abstention,
    missing_contract_fields,
    parse_artifact,
    resolve_execution_issue,
    validate_output_contract,
)
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.mcp import MCPServerConfig, create_toolset
from fedotmas.mcp.capabilities import (
    ToolCapability,
    normalize_tool_name,
    tool_capability,
)
from fedotmas.plugins._research_telemetry import (
    RESEARCH_CANDIDATE_LEDGER_KEY,
    RESEARCH_GATE_STATE_KEY,
    RESEARCH_MODE_STATE_KEY,
    RESEARCH_POLICY_STATE_KEY,
    RESEARCH_PROGRESS_STATE_KEY,
    RESEARCH_TURN_STATE_KEY,
)

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

TERMINAL_COMPLETION_PROTOCOL = (
    "COMPLETION PROTOCOL: If you establish a concrete answer supported by the "
    "available evidence, produce the normal requested answer. If you cannot "
    "establish a supported answer, emit exactly <abstain> followed by a concise "
    "reason and </abstain>. Do not put an abstention inside a success wrapper, "
    "and do not fabricate a solution."
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


def _contract_instruction(cfg: MAWAgentConfig) -> str:
    contract = cfg.output_contract
    if contract is None:
        return ""
    required = list(dict.fromkeys(contract.required_fields))
    identity = list(dict.fromkeys(contract.identity_fields))
    return (
        "\n\nRUNTIME HANDOFF CONTRACT (required for this nonterminal agent):\n"
        "Return exactly one JSON object. Required field paths are "
        f"{json.dumps(required)}. Identity paths are {json.dumps(identity)}. "
        "A path containing [] refers to every record in that repeated list; keep "
        "each identity inside its own record and never move one to the top level. "
        "Preserve upstream identities and do not invent missing values. "
        "Additional evidence and provenance fields are allowed."
    )


def _record_contract_repair(state: dict[str, Any], agent: str, event: dict) -> None:
    metadata = state.get(EXECUTION_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
        state[EXECUTION_METADATA_KEY] = metadata
    repairs = metadata.setdefault("contract_repairs", {})
    if not isinstance(repairs, dict):
        return
    repairs.setdefault(agent, []).append(event)


def _upstream_identity_values(
    state: dict[str, Any], cfg: MAWAgentConfig
) -> dict[str, dict[str, Any]]:
    values = {}
    for requirement in cfg.input_requirements:
        artifact = parse_artifact(state.get(requirement.source_key))
        if artifact is None:
            continue
        identity = {
            field: field_value(artifact, field)
            for field in requirement.identity_fields
            if not missing_contract_fields(artifact, [field])[1]
        }
        if identity:
            values[requirement.source_key] = identity
    return values


async def _repair_contract_once(
    model: str | BaseLlm,
    cfg: MAWAgentConfig,
    previous: Any,
    missing: list[str],
    upstream_identity_values: dict[str, Any],
) -> tuple[Any, dict[str, int]]:
    """Make one tools-free formatting repair using only the existing artifact."""
    llm = LLMRegistry.new_llm(model) if isinstance(model, str) else model
    prompt = (
        "Reformat the previous output to satisfy this handoff contract. Return "
        "exactly one JSON object. Required field paths: "
        f"{json.dumps(list(dict.fromkeys([*cfg.output_contract.required_fields, *cfg.output_contract.identity_fields])))}. "
        f"Identity paths to preserve exactly: {json.dumps(cfg.output_contract.identity_fields)}. "
        "Exact upstream identity values: "
        f"{json.dumps(upstream_identity_values, ensure_ascii=False, default=str)}. "
        f"Keys currently missing or invalid: {json.dumps(missing)}.\n"
        "Preserve repeated-list nesting and map only information explicitly present in the previous output. "
        "Do not invent facts, evidence, sources, or identity values. "
        "Never move an identity from a repeated record to a top-level field. "
        "If a required value is absent, leave it absent or null. Additional fields are allowed.\n"
        "Previous output:\n"
        f"{previous}"
    )
    request = LlmRequest(
        contents=[
            genai_types.Content(
                role="user", parts=[genai_types.Part.from_text(text=prompt)]
            )
        ],
        config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=cfg.max_output_tokens or 8192,
        ),
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    async for response in llm.generate_content_async(request):
        if response.usage_metadata is not None:
            usage["prompt_tokens"] = (
                response.usage_metadata.prompt_token_count or 0
            )
            usage["completion_tokens"] = (
                response.usage_metadata.candidates_token_count or 0
            )
        parts = response.content.parts if response.content else []
        text = "".join(part.text or "" for part in parts if part.text)
        if text:
            return text, usage
    return None, usage


def _repair_values_supported(
    previous: Any,
    repaired: Any,
    cfg: MAWAgentConfig,
    upstream: dict[str, dict[str, Any]],
) -> list[str]:
    """Reject repaired contract values that cannot be traced to prior output."""
    source = parse_artifact(previous)
    result = parse_artifact(repaired)
    if source is None or result is None or cfg.output_contract is None:
        return ["<invalid_repair_artifact>"]

    aliases = (
        {"answer", "solution", "result"},
        {"valid", "correct", "answer", "solution"},
        {"source", "citation", "reference", "provenance", "url"},
        {"quote", "excerpt", "transcript"},
    )

    def key_tokens(key: str) -> set[str]:
        tokens = {
            token.rstrip("s") for token in re.findall(r"[a-z0-9]+", key.casefold())
        }
        for group in aliases:
            if tokens & group:
                tokens.update(group)
        return tokens

    invalid = []
    fields = dict.fromkeys(
        [*cfg.output_contract.required_fields, *cfg.output_contract.identity_fields]
    )
    # Identity values in repaired output must also match any available upstream value.
    upstream_by_field: dict[str, list[Any]] = {}
    for identity in upstream.values():
        for field, value in identity.items():
            upstream_by_field.setdefault(field, []).append(value)
    for field in fields:
        source_values, source_path_valid = field_values(source, field)
        result_values, result_path_valid = field_values(result, field)
        is_identity = field in cfg.output_contract.identity_fields
        if result_path_valid and source_path_valid:
            if result_values != source_values:
                invalid.append(field)
            continue
        if is_identity:
            if not result_path_valid:
                invalid.append(field)
                continue
            expected = upstream_by_field.get(field, [])
            value = field_value(result, field)
            if not expected or not any(value == item for item in expected):
                invalid.append(field)
            continue
        if field in source and not _is_blank(source[field]) and result.get(field) != source[field]:
            invalid.append(field)
            continue
        if field not in result or _is_blank(result[field]):
            continue
        if field in source and source[field] == result[field]:
            source_fields = [field]
        else:
            source_fields = [
                old_field
                for old_field, old_value in source.items()
                if old_value == result[field]
                and (
                    old_field == field
                    or (
                        field not in cfg.output_contract.identity_fields
                        and key_tokens(old_field) & key_tokens(field)
                    )
                )
            ]
            if not source_fields and field not in cfg.output_contract.identity_fields:
                nested = []

                def collect(value: Any, semantic_field: str, found: list[Any]) -> None:
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if (
                                isinstance(key, str)
                                and key_tokens(key) & key_tokens(semantic_field)
                            ):
                                found.append(item)
                            collect(item, semantic_field, found)
                    elif isinstance(value, list):
                        for item in value:
                            collect(item, semantic_field, found)

                collect(source, field, nested)
                if (
                    nested
                    and all(item == nested[0] for item in nested)
                    and nested[0] == result[field]
                ):
                    source_fields = [field]
        if not source_fields:
            invalid.append(field)
            continue
        if (
            field in cfg.output_contract.identity_fields
            and field in upstream_by_field
            and any(
                result[field] != upstream_value
                for upstream_value in upstream_by_field[field]
            )
        ):
            invalid.append(field)
    return invalid


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
    terminal = _find_terminal_node(config.pipeline)
    final_answer_agent = config.final_answer_agent or (
        terminal.agent_name if terminal.type == "agent" else None
    )
    if final_answer_contract is not None:
        if final_answer_agent is None:
            raise ValueError(
                "Cannot infer final_answer_agent: the pipeline must end in one agent"
            )
        if agent_executes_in_loop(config.pipeline, final_answer_agent):
            raise ValueError(
                "final_answer_agent cannot execute inside a loop when a "
                "final_answer_contract is active; add a post-loop finalizer"
            )
    if final_answer_agent is not None and not agent_executes_in_loop(
        config.pipeline, final_answer_agent
    ):
        config.final_answer_agent = final_answer_agent
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
    instruction_text = cfg.instruction
    terminal_boundary = (
        final_answer_contract is not None and cfg.name == final_answer_agent
    )
    if cfg.output_contract is not None and not terminal_boundary:
        instruction_text += _contract_instruction(cfg)
    if final_answer_contract:
        instruction_text = (
            f"{instruction_text}\n\nFINAL ANSWER CONTRACT (terminal stage only):\n"
            f"{final_answer_contract}"
        )
    if cfg.name == final_answer_agent:
        instruction_text += f"\n\n{TERMINAL_COMPLETION_PROTOCOL}"
    research_guidance = {
        "discovery_only": (
            "Research mode: discovery_only. Find and select a small set of the best "
            "candidate sources, then hand off each source's identity, URL, title, "
            "and concise relevance evidence. A single exact authoritative source "
            "can be sufficient when the task asks for one. Do not inspect full "
            "source contents; a downstream extractor owns that work. Once suitable "
            "candidates are found, stop broad searching and hand them off."
        ),
        "mixed": (
            "Research mode: mixed. Discover candidates, inspect at least one "
            "relevant candidate, then search again only if a specific evidence gap "
            "remains. Reuse the persistent candidate list instead of rediscovering "
            "known sources."
        ),
        "inspection_only": (
            "Research mode: inspection_only. Inspect the supplied source URLs and "
            "identities and extract the requested evidence. Do not repeat broad "
            "discovery. If a supplied source fails, report the failure and pass the "
            "best honest incomplete result."
        ),
    }[cfg.research_mode]
    research_agent = _is_research_agent(cfg)
    if research_agent:
        instruction_text += f"\n\n{research_guidance}"
    if autonomous:
        instruction_text = frame_instruction(instruction_text)
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
        modes = callback_context.state.get(RESEARCH_MODE_STATE_KEY)
        if not isinstance(modes, dict):
            modes = {}
        modes[cfg.name] = cfg.research_mode
        callback_context.state[RESEARCH_MODE_STATE_KEY] = modes
        effective_policy = cfg.research_policy
        if (
            cfg.input_requirements
            and _is_verifier_role(cfg)
            and not _requests_independent_research(cfg)
        ):
            has_missing_evidence = any(
                describe_requirement(callback_context.state, item)[1]
                for item in cfg.input_requirements
            )
            if not has_missing_evidence:
                effective_policy = "evidence_first"
            elif effective_policy != "evidence_first":
                effective_policy = "targeted_recovery"
        policies = callback_context.state.get(RESEARCH_POLICY_STATE_KEY)
        if not isinstance(policies, dict):
            policies = {}
        policies[cfg.name] = effective_policy
        callback_context.state[RESEARCH_POLICY_STATE_KEY] = policies
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
        if cfg.name == final_answer_agent and is_explicit_abstention(value):
            callback_context.state[ABSTENTION_STATE_KEY] = {
                "status": "abstained",
                "reason": _abstention_reason(value),
                "agent": cfg.name,
            }
            return
        terminal_answer = terminal_boundary
        if terminal_answer:
            # This role may recover an input dependency, but its answer is
            # formatted for the caller and is never a structured handoff.
            if is_explicit_abstention(value):
                callback_context.state[ABSTENTION_STATE_KEY] = {
                    "status": "abstained",
                    "reason": _abstention_reason(value),
                    "agent": cfg.name,
                }
            return
        if cfg.output_contract is None:
            return
        missing = validate_output_contract(value, cfg.output_contract)
        if missing:
            _record_contract_repair(
                callback_context.state,
                cfg.name,
                {"status": "initial_contract_failure", "missing_fields": missing},
            )
            repaired = None
            repaired_missing = missing
            if not _is_blank(value):
                try:
                    repaired, usage = await _repair_contract_once(
                        model,
                        cfg,
                        value,
                        missing,
                        _upstream_identity_values(callback_context.state, cfg),
                    )
                    metadata = callback_context.state.setdefault(
                        EXECUTION_METADATA_KEY, {}
                    )
                    repair_tokens = metadata.setdefault(
                        "contract_repair_tokens",
                        {"prompt_tokens": 0, "completion_tokens": 0},
                    )
                    if isinstance(repair_tokens, dict):
                        for key, count in usage.items():
                            repair_tokens[key] = repair_tokens.get(key, 0) + count
                    repaired_missing = validate_output_contract(
                        repaired, cfg.output_contract
                    )
                    upstream_identity = _upstream_identity_values(
                        callback_context.state, cfg
                    )
                    unsupported = _repair_values_supported(
                        value,
                        repaired,
                        cfg,
                        upstream_identity,
                    )
                    repaired_missing = list(
                        dict.fromkeys([*repaired_missing, *unsupported])
                    )
                except Exception as exc:  # noqa: BLE001 - one bounded repair is best-effort
                    _record_contract_repair(
                        callback_context.state,
                        cfg.name,
                        {"status": "repair_failed", "error": str(exc)[:300]},
                    )
                    _log.warning(
                        "Contract repair failed | agent={} error={}", cfg.name, exc
                    )
            if repaired is not None and not repaired_missing:
                callback_context.state[cfg.output_key] = repaired
                value = repaired
                missing = []
                _record_contract_repair(
                    callback_context.state,
                    cfg.name,
                    {"status": "format_repair_succeeded"},
                )
            else:
                _record_contract_repair(
                    callback_context.state,
                    cfg.name,
                    {
                        "status": "repair_missing_semantic_fields",
                        "missing_fields": repaired_missing or missing,
                    },
                )
                missing = repaired_missing or missing
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
            contract_fields = (
                [
                    (field, "handoff_field")
                    for field in cfg.output_contract.required_fields
                ]
                + [
                    (field, "resolved_identity")
                    for field in cfg.output_contract.identity_fields
                ]
                if cfg.output_contract is not None
                else []
            )
            for field, kind in contract_fields:
                if not missing_contract_fields(artifact, [field])[1]:
                    _record_semantic_progress(
                        callback_context.state, cfg.name, f"{kind}:{field}"
                    )
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
                    if not missing_contract_fields(upstream, [field])[1]
                    and field_value(artifact, field) != field_value(upstream, field)
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
                    not missing_contract_fields(upstream, [field])[1]
                    and not missing_contract_fields(artifact, [field])[1]
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
        capability = _runtime_tool_capability(tool)
        blocked: tuple[str, str] | None = None
        if cfg.research_mode == "discovery_only" and capability in {
            ToolCapability.URL_INSPECTION,
            ToolCapability.DOCUMENT_INSPECTION,
            ToolCapability.MEDIA_INSPECTION,
        }:
            blocked = (
                "DISCOVERY_ONLY_INSPECTION_DISABLED",
                (
                    "This source-finding role selects candidate sources and returns their "
                    "identity, URLs, titles, and snippets. Pass full extraction to the "
                    "downstream extractor."
                ),
            )
        if cfg.research_mode == "discovery_only" and capability == ToolCapability.MEDIA_INSPECTION and normalize_tool_name(tool.name) == "get_video_info":
            blocked = None
        if (
            cfg.research_mode == "discovery_only"
            and capability == ToolCapability.MEDIA_INSPECTION
            and normalize_tool_name(tool.name) == "get_video_info"
        ):
            blocked = None
        if capability == ToolCapability.BROWSER_NAVIGATION:
            candidates_for_browser = _candidate_context(tool_context.state, cfg.name)
            args_text = json.dumps(args, ensure_ascii=False).casefold()
            anchored = any(str(item.get("url", "")).casefold() in args_text for item in candidates_for_browser)
            turn_root = tool_context.state.get(RESEARCH_TURN_STATE_KEY)
            current_turn = turn_root.get(cfg.name) if isinstance(turn_root, dict) else None
            if not anchored:
                blocked = ("BROWSER_DISCOVERY_POLICY", "Browser exploration must be anchored to a known candidate; use the gated discovery tool for open-ended search.")
            elif isinstance(current_turn, dict) and current_turn.get("force_converge"):
                blocked = ("RESEARCH_CONVERGENCE_REQUIRED", "Inspect a known candidate or hand off; browser exploration is paused.")
        if normalize_tool_name(tool.name) in {"get_next_action", "research_controller_get_next_action"}:
            turn_root = tool_context.state.get(RESEARCH_TURN_STATE_KEY)
            current_turn = turn_root.get(cfg.name) if isinstance(turn_root, dict) else None
            if isinstance(current_turn, dict) and current_turn.get("force_converge"):
                blocked = ("RESEARCH_CONVERGENCE_REQUIRED", "Controller polling is paused after repeated turns without progress.")
        if capability == ToolCapability.DISCOVERY:
            effective_policy = _effective_research_policy(tool_context.state, cfg)
            gate_root = tool_context.state.get(RESEARCH_GATE_STATE_KEY)
            gate = gate_root.get(cfg.name) if isinstance(gate_root, dict) else None
            if effective_policy == "evidence_first":
                blocked = (
                    "EVIDENCE_FIRST_SEARCH_DISABLED",
                    (
                        "This role is evidence-first. Verify using supplied evidence; "
                        "mark missing claims unresolved."
                    ),
                )
            elif cfg.research_mode == "inspection_only" and effective_policy != "targeted_recovery":
                blocked = (
                    "INSPECTION_ONLY_DISCOVERY_DISABLED",
                    (
                        "This role inspects supplied sources. Do not repeat broad discovery; "
                        "report the missing source as unresolved."
                    ),
                )
            elif cfg.research_mode == "mixed" and isinstance(gate, dict) and (
                gate.get("phase") == "inspect" or gate.get("gated") is True
            ):
                candidates = _candidate_context(tool_context.state, cfg.name)
                blocked = (
                    "INSPECT_CANDIDATES_FIRST",
                    "Discovery is paused. Inspect at least one pending candidate first: "
                    + json.dumps(candidates, ensure_ascii=False),
                )
            candidates = _candidate_context(tool_context.state, cfg.name)
            turns_root = tool_context.state.get(RESEARCH_TURN_STATE_KEY)
            turns = turns_root.get(cfg.name) if isinstance(turns_root, dict) else None
            if isinstance(turns, dict):
                if turns.get("force_converge") is True:
                    blocked = (
                        "RESEARCH_CONVERGENCE_REQUIRED",
                        (
                            "Broad discovery is paused after repeated turns without new "
                            "evidence. Change evidence method, inspect/select existing "
                            "candidates, or provide the best honest incomplete handoff."
                        ),
                    )
                elif turns.get("discovery_calls", 0) >= 1:
                    blocked = (
                        "DISCOVERY_FANOUT_LIMIT",
                        (
                            "Only one broad discovery call is allowed in this model turn. "
                            "Use its results and inspect a candidate before another search."
                        ),
                    )
                else:
                    turns["discovery_calls"] = 1
                    if cfg.research_mode == "inspection_only" and effective_policy == "targeted_recovery":
                        progress_root = tool_context.state.get(RESEARCH_PROGRESS_STATE_KEY, {})
                        progress = progress_root.get(cfg.name, {}) if isinstance(progress_root, dict) else {}
                        query = str(args.get("query", "")).casefold()
                        candidate_text = " ".join(str(item.get("title", "")) + " " + str(item.get("url", "")) for item in _candidate_context(tool_context.state, cfg.name)).casefold()
                        if not query or not any(token in candidate_text for token in query.split() if len(token) >= 5):
                            blocked = ("TARGETED_RECOVERY_REQUIRES_IDENTITY", "Recovery search must reference a supplied title, DOI, domain, or URL.")
                        elif isinstance(progress, dict) and progress.get("targeted_recovery_used"):
                            blocked = ("TARGETED_RECOVERY_EXHAUSTED", "The one targeted recovery search has already been used.")
                        else:
                            progress["targeted_recovery_used"] = True
                            progress_root[cfg.name] = progress
                            tool_context.state[RESEARCH_PROGRESS_STATE_KEY] = progress_root
                    if isinstance(turns_root, dict):
                        turns_root[cfg.name] = turns
                        tool_context.state[RESEARCH_TURN_STATE_KEY] = turns_root
        if blocked is not None:
            _record_turn_tool_call(
                tool_context.state,
                cfg.name,
                tool.name,
                blocked_reason=blocked[0],
                call_id=getattr(tool_context, "function_call_id", None),
            )
            return {
                "isError": True,
                "error_code": blocked[0],
                "error": blocked[1],
            }
        _record_turn_tool_call(
            tool_context.state,
            cfg.name,
            tool.name,
            call_id=getattr(tool_context, "function_call_id", None),
        )
        return None

    per_agent_limit = (
        cfg.max_llm_turns or max_agent_llm_turns or get_max_agent_llm_turns()
    )

    async def before_model(
        callback_context: CallbackContext, llm_request
    ) -> LlmResponse | None:
        state = callback_context.state
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
        turn_index = used + 1
        if isinstance(turns, dict):
            turns[cfg.name] = turn_index
        progress_root = state.get(RESEARCH_PROGRESS_STATE_KEY)
        progress = progress_root.get(cfg.name) if isinstance(progress_root, dict) else None
        if not isinstance(progress, dict):
            progress = {"version": 0, "last_turn_version": 0, "no_progress_turns": 0}
        version = progress.get("version", 0)
        version = version if isinstance(version, int) and not isinstance(version, bool) else 0
        previous_version = progress.get("last_turn_version", version)
        previous_version = (
            previous_version
            if isinstance(previous_version, int) and not isinstance(previous_version, bool)
            else version
        )
        no_progress = progress.get("no_progress_turns", 0)
        no_progress = no_progress if isinstance(no_progress, int) and not isinstance(no_progress, bool) else 0
        recorded_progress = progress.get("progress_events")
        recorded_progress = recorded_progress if isinstance(recorded_progress, list) else []
        progress_delta = max(0, version - previous_version)
        semantic_progress_events = [
            str(signal)[:160] for signal in recorded_progress[-progress_delta:]
        ] if progress_delta else []
        if turn_index > 1:
            no_progress = no_progress + 1 if version <= previous_version else 0
        else:
            no_progress = 0
        progress["last_turn_version"] = version
        progress["no_progress_turns"] = no_progress
        if not isinstance(progress_root, dict):
            progress_root = {}
        progress_root[cfg.name] = progress
        state[RESEARCH_PROGRESS_STATE_KEY] = progress_root
        if no_progress:
            progress_metadata = metadata.setdefault("research_progress", {})
            if isinstance(progress_metadata, dict):
                progress_metadata[cfg.name] = {
                    "no_progress_turns": no_progress,
                    "version": version,
                }

        turn_root = state.get(RESEARCH_TURN_STATE_KEY)
        if not isinstance(turn_root, dict):
            turn_root = {}
        turn_state = {
            "turn_index": turn_index,
            "discovery_calls": 0,
            "force_converge": no_progress >= 2,
        }
        turn_root[cfg.name] = turn_state
        state[RESEARCH_TURN_STATE_KEY] = turn_root

        all_tool_names = set(llm_request.tools_dict)
        removed: dict[str, str] = {}
        for name, tool in llm_request.tools_dict.items():
            if _runtime_tool_capability(tool) == ToolCapability.DIAGNOSTIC:
                removed[name] = "diagnostic/control tools are internal-only"
        discovery_names = {
            name
            for name, tool in llm_request.tools_dict.items()
            if _runtime_tool_capability(tool) == ToolCapability.DISCOVERY
        }
        inspection_names = {
            name
            for name, tool in llm_request.tools_dict.items()
            if _runtime_tool_capability(tool)
            in {
                ToolCapability.URL_INSPECTION,
                ToolCapability.DOCUMENT_INSPECTION,
                ToolCapability.MEDIA_INSPECTION,
            }
        }
        gate_root = state.get(RESEARCH_GATE_STATE_KEY)
        gate = gate_root.get(cfg.name) if isinstance(gate_root, dict) else None
        candidates = _candidate_context(state, cfg.name)
        hide_discovery_reason = None
        effective_policy = _effective_research_policy(state, cfg)
        if effective_policy == "evidence_first":
            hide_discovery_reason = "evidence_first policy"
        elif cfg.research_mode == "inspection_only" and not (
            effective_policy == "targeted_recovery" and not _targeted_recovery_used(state, cfg.name)
        ):
            hide_discovery_reason = "inspection_only research mode"
        elif cfg.research_mode == "mixed" and isinstance(gate, dict) and gate.get("phase") == "inspect":
            hide_discovery_reason = "candidate inspection required"
        elif cfg.research_mode == "discovery_only" and _discovery_waves_used(state, cfg.name) >= 3:
            hide_discovery_reason = "discovery wave allowance exhausted"
        elif turn_state["force_converge"]:
            hide_discovery_reason = "repeated turns without semantic progress"
        if hide_discovery_reason:
            removed.update({name: hide_discovery_reason for name in discovery_names})
        if cfg.research_mode == "discovery_only":
            removed.update(
                {
                    name: "discovery_only source selection role"
                    for name in inspection_names
                    if normalize_tool_name(llm_request.tools_dict[name].name) != "get_video_info"
                    if normalize_tool_name(llm_request.tools_dict[name].name) not in {"get_video_info"}
                }
            )

        surface_filter_names = set()
        for toolset in tools:
            filtered = getattr(toolset, "_fedotmas_filtered_diagnostic_tools", set())
            if isinstance(filtered, set):
                surface_filter_names.update(filtered)
        for name in surface_filter_names:
            removed[name] = "diagnostic/control tools are internal-only"
        for name in all_tool_names:
            if _runtime_tool_capability(llm_request.tools_dict[name]) == ToolCapability.DIAGNOSTIC:
                removed[name] = "diagnostic/control tools are internal-only"
        for name in discovery_names:
            if hide_discovery_reason:
                removed[name] = hide_discovery_reason
        if removed:
            declarations_removed = set(removed)
            retained = []
            for group in llm_request.config.tools or []:
                declarations = group.function_declarations
                if declarations is None:
                    retained.append(group)
                    continue
                group.function_declarations = [
                    declaration
                    for declaration in declarations
                    if declaration.name not in declarations_removed
                ]
                if group.function_declarations:
                    retained.append(group)
            llm_request.config.tools = retained

        if effective_policy == "evidence_first":
            llm_request.append_instructions(
                ["This role is evidence-first. Verify against supplied evidence; identify missing evidence as unresolved."]
            )
        if cfg.research_mode == "mixed" and isinstance(gate, dict) and gate.get("phase") == "inspect":
            llm_request.append_instructions(
                ["Inspect at least one pending candidate before another broad search. Available candidates: " + json.dumps(candidates, ensure_ascii=False)]
            )
        if cfg.research_mode == "discovery_only" and _discovery_waves_used(state, cfg.name) >= 3:
            llm_request.append_instructions(
                ["Select the best small set from these candidates and return the source handoff now: " + json.dumps(candidates, ensure_ascii=False)]
            )
        elif candidates and research_agent:
            llm_request.append_instructions(
                ["Persistent candidate ledger (reuse these titles, snippets, and URLs; do not search for known candidates again): " + json.dumps(candidates, ensure_ascii=False)]
            )
        if research_agent or discovery_names:
            llm_request.append_instructions(
                ["Per-turn research budget: make at most one broad discovery call during this model turn. Known-URL inspections and computations are not subject to this limit."]
            )
        if turn_state["force_converge"] and (research_agent or discovery_names):
            llm_request.append_instructions(
                ["Several consecutive turns produced no semantic progress. Change strategy once, inspect/select existing evidence, or provide the best honest incomplete handoff. Do not repeat a low-value search."]
            )

        _record_turn_observability(
            state,
            cfg.name,
            turn_index=turn_index,
            visible_tools=sorted(
                name for name in all_tool_names if name not in removed
            ),
            visible_tool_declarations=_visible_tool_declarations(
                llm_request, removed
            ),
            removed_tools=[{"name": name, "reason": reason} for name, reason in sorted(removed.items())],
            research_mode=cfg.research_mode,
            gate_state=(gate.get("phase") if isinstance(gate, dict) else "discover"),
            candidate_count=len(candidates),
            candidate_ledger_summary=candidates[:4],
            controller_recommendation=_controller_recommendation(state, cfg.name),
            semantic_progress_version=version,
            semantic_progress_events=semantic_progress_events,
            no_progress_turns=no_progress,
        )
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
    return tool_capability(tool.name) == ToolCapability.DISCOVERY


def _is_research_agent(cfg: MAWAgentConfig) -> bool:
    if "research_mode" in cfg.model_fields_set:
        return True
    if re.search(
        r"\b(research\w*|source[_ -]?finder|structured[_ -]?extractor|verif\w*|fact[ -]?check\w*)\b",
        f"{cfg.name} {cfg.instruction}".casefold(),
    ):
        return True
    return any(
        tool_capability(tool)
        in {
            ToolCapability.DISCOVERY,
            ToolCapability.URL_INSPECTION,
            ToolCapability.DOCUMENT_INSPECTION,
            ToolCapability.MEDIA_INSPECTION,
            ToolCapability.BROWSER_NAVIGATION,
        }
        for tool in cfg.tools
    )


def _is_verifier_role(cfg: MAWAgentConfig) -> bool:
    text = f"{cfg.name} {cfg.instruction}".casefold()
    return bool(re.search(r"\b(verif\w*|fact[ -]?check\w*)\b", text))


def _requests_independent_research(cfg: MAWAgentConfig) -> bool:
    return bool(
        re.search(
            r"\bindependent(?:ly)?\s+(?:research|search|gather|verify|evidence|sources?)\b",
            cfg.instruction.casefold(),
        )
    )


def _effective_research_policy(state: Any, cfg: MAWAgentConfig) -> str:
    root = state.get(RESEARCH_POLICY_STATE_KEY) if hasattr(state, "get") else None
    policy = root.get(cfg.name) if isinstance(root, dict) else None
    return policy if policy in {"independent", "evidence_first", "targeted_recovery"} else cfg.research_policy


def _candidate_context(state: Any, agent: str) -> list[dict[str, Any]]:
    root = state.get(RESEARCH_CANDIDATE_LEDGER_KEY) if hasattr(state, "get") else None
    ledger = root.get(agent) if isinstance(root, dict) else None
    if not isinstance(ledger, list):
        return []
    selected = ledger[-8:]
    return [
        {
            "url": item.get("url"),
            "title": str(item.get("title") or "")[:160],
            "snippet": str(item.get("snippet") or "")[:320],
            "source_tool": str(item.get("source_tool") or "")[:60],
            "inspected": item.get("inspected") is True,
            "inspection_status": str(item.get("inspection_status") or "uninspected")[:20],
            "inspection_tool": str(item.get("inspection_tool") or "")[:60],
            "inspection_error": str(item.get("inspection_error") or "")[:160],
        }
        for item in selected
        if isinstance(item, dict) and isinstance(item.get("url"), str)
    ]


def _record_turn_observability(
    state: dict[str, Any],
    agent: str,
    *,
    turn_index: int,
    visible_tools: list[str],
    visible_tool_declarations: list[dict[str, Any]],
    removed_tools: list[dict[str, str]],
    research_mode: str,
    gate_state: str,
    candidate_count: int,
    candidate_ledger_summary: list[dict[str, Any]],
    controller_recommendation: str | None,
    semantic_progress_version: int,
    semantic_progress_events: list[str],
    no_progress_turns: int,
) -> None:
    metadata = state.setdefault(EXECUTION_METADATA_KEY, {})
    if not isinstance(metadata, dict):
        return
    traces = metadata.setdefault("turn_observability", {})
    if not isinstance(traces, dict):
        return
    events = traces.setdefault(agent, [])
    if not isinstance(events, list):
        return
    events.append(
        {
            "agent": agent,
            "turn_index": turn_index,
            "visible_tools": visible_tools[:80],
            "visible_tool_declarations": visible_tool_declarations[:20],
            "removed_tools": removed_tools[:80],
            "research_mode": research_mode,
            "discovery_gate_state": gate_state,
            "candidate_count": candidate_count,
            "candidate_ledger_summary": candidate_ledger_summary[:4],
            "controller_recommendation": controller_recommendation,
            "semantic_progress_version": semantic_progress_version,
            "semantic_progress_events": semantic_progress_events[-8:],
            "no_progress_turns": no_progress_turns,
            "tool_calls_selected": [],
            "calls_blocked": [],
        }
    )
    del events[:-20]


def _visible_tool_declarations(
    llm_request: LlmRequest, removed: dict[str, str]
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for group in llm_request.config.tools or []:
        for declaration in group.function_declarations or []:
            name = str(declaration.name or "")
            if name in removed:
                continue
            summary: dict[str, Any] = {
                "name": name[:120],
                "description": str(declaration.description or "")[:200],
            }
            parameters = getattr(declaration, "parameters", None)
            if parameters is not None:
                if hasattr(parameters, "model_dump"):
                    parameters = parameters.model_dump(mode="json", exclude_none=True)
                encoded = json.dumps(parameters, ensure_ascii=False, default=str)
                summary["parameters"] = encoded[:600]
                summary["parameters_truncated"] = len(encoded) > 600
            declarations.append(summary)
            if len(declarations) >= 20:
                break
        if len(declarations) >= 20:
            break
    return declarations


def _record_turn_tool_call(
    state: Any,
    agent: str,
    tool_name: str,
    *,
    blocked_reason: str | None = None,
    call_id: str | None = None,
) -> None:
    metadata = state.get(EXECUTION_METADATA_KEY) if hasattr(state, "get") else None
    traces = metadata.get("turn_observability") if isinstance(metadata, dict) else None
    events = traces.get(agent) if isinstance(traces, dict) else None
    if not isinstance(events, list) or not events:
        return
    event = events[-1]
    target = "calls_blocked" if blocked_reason else "tool_calls_selected"
    calls = event.setdefault(target, [])
    if isinstance(calls, list) and len(calls) < 40:
        record = {"tool": tool_name[:120]}
        if isinstance(call_id, str):
            record["call_id"] = call_id[:120]
        if blocked_reason:
            record["reason"] = blocked_reason
        calls.append(record)


def _record_semantic_progress(state: dict[str, Any], agent: str, signal: str) -> bool:
    root = state.get(RESEARCH_PROGRESS_STATE_KEY)
    if not isinstance(root, dict):
        root = {}
    progress = root.get(agent)
    if not isinstance(progress, dict):
        progress = {"version": 0, "progress_events": [], "seen_signals": []}
    seen = progress.get("seen_signals")
    seen = seen if isinstance(seen, list) else []
    if signal in seen:
        return False
    seen.append(signal)
    progress["seen_signals"] = seen[-80:]
    progress["version"] = max(0, int(progress.get("version", 0))) + 1
    events = progress.get("progress_events")
    events = events if isinstance(events, list) else []
    events.append(signal[:160])
    progress["progress_events"] = events[-20:]
    root[agent] = progress
    state[RESEARCH_PROGRESS_STATE_KEY] = root
    return True


def _controller_recommendation(state: dict[str, Any], agent: str) -> str | None:
    metadata = state.get(EXECUTION_METADATA_KEY)
    recommendations = (
        metadata.get("controller_recommendations")
        if isinstance(metadata, dict)
        else None
    )
    if isinstance(recommendations, dict):
        latest = recommendations.get(agent)
        if latest in {"continue_search", "change_strategy", "strategy_blocked", "synthesize"}:
            return latest
    return None


def _abstention_reason(value: Any) -> str:
    if isinstance(value, dict):
        reason = value.get("reason")
        return str(reason)[:500] if isinstance(reason, str) else "No supported answer was established."
    if isinstance(value, str):
        text = value.strip()
        start = text.find("<abstain>") + len("<abstain>")
        end = text.find("</abstain>", start)
        if end >= 0:
            return text[start:end].strip()[:500]
    return "No supported answer was established."


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


def _runtime_tool_capability(tool: Any) -> ToolCapability:
    return tool_capability(tool.name, description=getattr(tool, "description", "") or "")

def _discovery_waves_used(state: Any, agent: str) -> int:
    progress = state.get(RESEARCH_PROGRESS_STATE_KEY, {}) if hasattr(state, "get") else {}
    item = progress.get(agent, {}) if isinstance(progress, dict) else {}
    return int(item.get("productive_discovery_waves", 0)) if isinstance(item, dict) else 0

def _targeted_recovery_used(state: Any, agent: str) -> bool:
    root = state.get(RESEARCH_PROGRESS_STATE_KEY, {}) if hasattr(state, "get") else {}
    item = root.get(agent, {}) if isinstance(root, dict) else {}
    return bool(item.get("targeted_recovery_used")) if isinstance(item, dict) else False
