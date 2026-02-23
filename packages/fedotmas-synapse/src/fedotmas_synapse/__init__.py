"""Synapse integration for FEDOT.MAS."""

from fedotmas_synapse.checkpoint import CheckpointCallback
from fedotmas_synapse.otel import OtelEventCallback, configure_otel
from fedotmas_synapse.plugin import SynapsePlugin
from fedotmas_synapse.session import MongoSessionService

__all__ = [
    "CheckpointCallback",
    "MongoSessionService",
    "OtelEventCallback",
    "SynapsePlugin",
    "configure_otel",
]
