"""Bifrost model gates for ADK LlmRequest.

Re-implements CodeSynapse's ``LLMClient._apply_model_gates()`` logic
for the ADK ``LlmRequest`` type.  Cannot reuse the original directly
because CodeSynapse operates on ``(model: str, temperature: float)``
while ADK operates on ``LlmRequest`` with ``config.temperature``.

See INCOMPATIBILITIES.md §2 for the gate-duplication tension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest

_log = get_logger("fedotmas_synapse.model_gates")

_GPT5_PREFIX = "gpt-5-"
_GPT5_EXCEPTIONS = frozenset({"gpt-5-chat-latest"})
_GPT5_FORCED_TEMPERATURE = 1.0


class BifrostModelGates:
    """Enforce model-specific temperature gates on ADK ``LlmRequest``.

    Stateless — instantiate once and call :meth:`enforce` from
    ``SynapsePlugin.before_model_callback``.
    """

    def enforce(self, llm_request: LlmRequest) -> None:
        """Mutate *llm_request*.config.temperature in-place.

        Currently applies:
        - GPT-5 temperature gate (see :meth:`_apply_gpt5_temperature_gate`)

        Args:
            llm_request: The ADK LLM request to gate.
        """
        raise NotImplementedError("BifrostModelGates.enforce")

    @staticmethod
    def _extract_base_model(model: str) -> str:
        """Strip provider prefix: ``'openai/gpt-5-mini'`` → ``'gpt-5-mini'``."""
        return model.rsplit("/", 1)[-1] if "/" in model else model

    @staticmethod
    def _apply_gpt5_temperature_gate(
        llm_request: LlmRequest, base_model: str
    ) -> None:
        """Force ``temperature=1.0`` for ``gpt-5-*`` models.

        Exception: ``gpt-5-chat-latest`` allows custom temperature.

        Mirrors ``LLMClient._apply_model_gates()`` from
        ``src/llm/client.py`` in CodeSynapse.
        """
        raise NotImplementedError(
            "BifrostModelGates._apply_gpt5_temperature_gate"
        )
