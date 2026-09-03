from __future__ import annotations

import time
from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.genai import types

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.logging")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


def _is_workflow_node(name: str) -> bool:
    return name.startswith(_WORKFLOW_PREFIXES)


def _session_key(
    context: CallbackContext | InvocationContext, agent_name: str
) -> tuple[str, str]:
    """Scope per-agent bookkeeping to one session.

    One callback writes a bucket and another reads it, so both must derive the
    id the same way -- ``session.id``, which both context types expose.  There
    is deliberately no fallback: were only one of the two to lose its session,
    a fallback would send the writer and the reader to different buckets and
    report every agent as silent.  Failing here is the louder, truer signal.
    """
    return (context.session.id, agent_name)


class LoggingPlugin(BasePlugin):
    """Default FEDOT.MAS plugin that logs agent lifecycle and event details.

    Consolidates logging previously scattered across ``_ppline_utils``,
    ``builder``, and ``runner``.  Always returns ``None`` so it never
    short-circuits other plugins.
    """

    def __init__(self) -> None:
        super().__init__(name="fedotmas_logging")
        self._agent_start: dict[tuple[str, str], float] = {}
        # Both keyed by (session, agent): benchmarks and the optimizer drive
        # several sessions through one plugin instance, and a bare agent name
        # would let one run's state writes silence another run's "No output"
        # warning, and one run's start time distort another run's elapsed.
        self._written_keys: dict[tuple[str, str], set[str]] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        # A run that aborts mid-agent -- a pipeline timeout, an open circuit,
        # a search limit -- never reaches after_agent_callback, so its entries
        # would sit here forever now that the key carries a session id.  Drop
        # only this session's, the way the sibling plugins do: clearing outright
        # would erase a concurrently running run's bookkeeping.
        session_id = invocation_context.session.id
        self._agent_start = {
            key: value
            for key, value in self._agent_start.items()
            if key[0] != session_id
        }
        self._written_keys = {
            key: value
            for key, value in self._written_keys.items()
            if key[0] != session_id
        }
        return None

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if not _is_workflow_node(agent.name):
            _log.info("Agent started | name={}", agent.name)
        key = _session_key(callback_context, agent.name)
        self._agent_start[key] = time.monotonic()
        self._written_keys.pop(key, None)
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        key = _session_key(callback_context, agent.name)
        t0 = self._agent_start.pop(key, None)
        if t0 is not None and not _is_workflow_node(agent.name):
            _log.info(
                "Agent done | name={} elapsed={:.1f}s",
                agent.name,
                time.monotonic() - t0,
            )

        # An agent that writes nothing at all is otherwise invisible: the
        # "Empty output" warning below only fires once a key has been written.
        # Downstream steps then interpolate a missing key and carry on, which
        # is how a pipeline reaches the end having researched nothing.
        # Against the agent's own key, not any state write: a tool that
        # stashes something in state would otherwise mask a silent agent.
        output_key = getattr(agent, "output_key", None)
        written = self._written_keys.get(key, set())
        if (
            output_key
            and not _is_workflow_node(agent.name)
            and output_key not in written
        ):
            _log.warning(
                "No output | agent={} key='{}' — nothing was written to state",
                agent.name,
                output_key,
            )
        self._written_keys.pop(key, None)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """Record the shape of each model turn.

        Without this, a turn that returns only reasoning and no content is
        indistinguishable from one that never happened.
        """
        parts = (llm_response.content.parts if llm_response.content else None) or []
        # Thought and answer counted apart: ADK writes output_key only from
        # parts that are *not* thoughts (llm_agent.py, __handle_output_key), so
        # a turn that is all reasoning silently leaves state untouched.
        answer = sum(len(p.text) for p in parts if p.text and not p.thought)
        thought = sum(len(p.text) for p in parts if p.text and p.thought)
        calls = [p.function_call.name for p in parts if p.function_call]
        _log.debug(
            "Model turn | agent={} finish={} parts={} answer_chars={} "
            "thought_chars={} calls={}",
            callback_context.agent_name,
            llm_response.finish_reason,
            len(parts),
            answer,
            thought,
            calls or None,
        )
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        if event.partial:
            return None

        # Tool calls
        for fc in event.get_function_calls():
            _log.info(
                "Tool call | agent={} tool={} args={}",
                event.author,
                fc.name,
                fc.args,
            )

        # Tool responses
        for fr in event.get_function_responses():
            resp_str = str(fr.response)[:200] if fr.response else ""
            is_error = (
                isinstance(fr.response, dict) and fr.response.get("isError") is True
            )
            if is_error:
                _log.warning(
                    "Tool error | agent={} tool={} response={}",
                    event.author,
                    fr.name,
                    resp_str,
                )
            else:
                _log.info(
                    "Tool result | agent={} tool={}",
                    event.author,
                    fr.name,
                )

        # Token usage
        if event.usage_metadata:
            um = event.usage_metadata
            prompt = um.prompt_token_count or 0
            completion = um.candidates_token_count or 0
            if prompt or completion:
                _log.info(
                    "Tokens | agent={} prompt={} completion={}",
                    event.author,
                    prompt,
                    completion,
                )

        # Text response (no function calls)
        if event.content and event.content.parts and not event.get_function_calls():
            texts = [p.text for p in event.content.parts if p.text]
            if texts:
                preview = texts[0][:200]
                _log.trace("Response | agent={} text={}", event.author, preview)

        # State changes
        if event.actions.state_delta:
            if event.author:
                self._written_keys.setdefault(
                    _session_key(invocation_context, event.author), set()
                ).update(event.actions.state_delta)
            for key, value in event.actions.state_delta.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    _log.warning(
                        "Empty output | agent={} key='{}'",
                        event.author,
                        key,
                    )
            _log.info(
                "State update | agent={} keys={}",
                event.author,
                list(event.actions.state_delta.keys()),
            )

        return None
