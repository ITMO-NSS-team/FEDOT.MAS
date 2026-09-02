from __future__ import annotations

import itertools
import re
from typing import Any, TypeAlias, cast

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.utils.instructions_utils import inject_session_state
from google.genai import types as genai_types

from fedotmas._settings import (
    ModelConfig,
    get_max_loop_iterations,
    get_worker_models,
)
from fedotmas.common.llm import make_llm
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.mcp import MCPServerConfig, create_toolset

AgentTree: TypeAlias = BaseAgent

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


#: Prepended to every agent instruction, in this builder and in ``mas.builder``.
#: An unattended run has no one to answer a question, so an agent that ends its
#: turn asking for input has delivered nothing.  This is a fact about the runtime
#: rather than a property of any one generated design, which is why it lives in
#: the builders and not in the generation prompts -- and why a caller that *does*
#: have a person in the loop turns it off with ``autonomous=False``.  Braces are
#: avoided on purpose: ADK would read them as state references.
AUTONOMY_PREAMBLE = (
    "You are working on your own. No one is reading along to answer a question, "
    "pick between options, or supply a document you ask for; a request for input "
    "reaches nobody and ends the run with nothing delivered.\n"
    "So do not ask the user for anything and do not end your turn waiting for a "
    "reply. Where the task leaves something open, take the most reasonable "
    "reading, name it in one line, and carry it through. Where something is "
    "genuinely unavailable, say what is missing and what it would change, then "
    "give the best answer the available evidence supports. If your own task is to "
    "raise questions or lay out options, write them as your answer -- just do not "
    "hand them over as a decision for someone else to make. A provisional answer "
    "with its assumptions named is the deliverable; a request for input is not.\n"
    "These are your working conditions, not your subject. Write the answer "
    "itself, and do not quote or discuss this notice."
)


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
):
    """Resolve state refs at call time, naming the ones that came back empty."""

    async def provide(readonly_context: ReadonlyContext) -> str:
        state = readonly_context.state
        text = instruction
        for ref, key in {
            (m.group(0), m.group(1)) for m in _STATE_REF_RE.finditer(instruction)
        }:
            # Only a key some step actually produces can be *missing*; anything
            # else is a literal the task carried in, and claiming a step failed
            # for it would be its own fabrication.
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
        return await inject_session_state(text, readonly_context)

    return provide


def build(
    config: MAWConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
    autonomous: bool = True,
) -> BaseAgent:
    """Convert a ``MAWConfig`` into an executable ADK agent tree.

    Pass ``autonomous=False`` when the tree is served to a person who can answer
    a clarifying question; see :data:`AUTONOMY_PREAMBLE`.
    """
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
    )


def _build_node(
    node: MAWStepConfig,
    agents: dict[str, MAWAgentConfig],
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
    state_keys: frozenset[str] | None = None,
    *,
    autonomous: bool = True,
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
        )

    children = [
        _build_node(
            c, agents, mcp_registry, worker_models, state_keys, autonomous=autonomous
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


def _build_llm_agent(
    cfg: MAWAgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
    state_keys: frozenset[str] | None = None,
    *,
    autonomous: bool = True,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_llm(cfg.model, worker_models)
    _log.debug("Built agent | name={} model={}", cfg.name, model)
    instruction_text = (
        f"{AUTONOMY_PREAMBLE}\n\n{cfg.instruction}" if autonomous else cfg.instruction
    )
    # Decided on the final text: a state reference anywhere in it, preamble
    # included, has to reach the provider rather than ADK's plain-string path.
    instruction = (
        _instruction_provider(instruction_text, cfg.name, state_keys)
        if _STATE_REF_RE.search(instruction_text)
        else instruction_text
    )
    kwargs: dict = {}
    if cfg.max_output_tokens is not None:
        kwargs["generate_content_config"] = genai_types.GenerateContentConfig(
            max_output_tokens=cfg.max_output_tokens,
        )
    return LlmAgent(
        name=cfg.name,
        model=model,
        instruction=instruction,
        output_key=cfg.output_key,
        tools=tools,
        **kwargs,
    )


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
