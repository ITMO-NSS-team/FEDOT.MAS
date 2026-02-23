from __future__ import annotations

from google.adk.agents.callback_context import CallbackContext


class CheckpointCallback:
    """Before/after agent callbacks that snapshot session state."""

    def __init__(self, storage_uri: str = "") -> None:
        self.storage_uri = storage_uri

    def before_agent(self, callback_context: CallbackContext) -> None:
        raise NotImplementedError("CheckpointCallback.before_agent")

    def after_agent(self, callback_context: CallbackContext) -> None:
        raise NotImplementedError("CheckpointCallback.after_agent")
