"""Synapse integration for FEDOT.MAS."""

from fedotmas_synapse.bridge import MASEventBridge
from fedotmas_synapse.checkpoint import CheckpointCallback
from fedotmas_synapse.memory import SynapseMemoryServiceAdapter
from fedotmas_synapse.model_gates import BifrostModelGates
from fedotmas_synapse.otel import OtelEventCallback, configure_otel
from fedotmas_synapse.plugin import SynapsePlugin
from fedotmas_synapse.session import MongoSessionService

__all__ = [
    "BifrostModelGates",
    "CheckpointCallback",
    "MASEventBridge",
    "MongoSessionService",
    "OtelEventCallback",
    "SynapseMemoryServiceAdapter",
    "SynapsePlugin",
    "configure_otel",
]
