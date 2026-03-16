from __future__ import annotations

from typing import TYPE_CHECKING

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest

_log = get_logger("fedotmas_synapse.model_gates")


class BifrostModelGates:
    """Placeholder for model validation rules (allowlists, token budgets).

    Currently a no-op.  Future versions will enforce model access policies
    defined by the CodeSynapse platform (e.g. restrict models per project,
    enforce per-request token budgets).
    """

    def enforce(self, llm_request: LlmRequest) -> None:
        """Validate *llm_request* against gating rules.

        Currently a no-op — all requests are allowed through.
        """
