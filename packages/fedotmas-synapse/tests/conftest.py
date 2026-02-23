from __future__ import annotations

import pytest

from fedotmas_synapse.checkpoint import CheckpointCallback
from fedotmas_synapse.otel import OtelEventCallback
from fedotmas_synapse.plugin import SynapsePlugin
from fedotmas_synapse.session import MongoSessionService

MONGO_URI = "mongodb://localhost:27017"


@pytest.fixture
def mongo_uri() -> str:
    return MONGO_URI


@pytest.fixture
def session_service(mongo_uri: str) -> MongoSessionService:
    return MongoSessionService(mongo_uri)


@pytest.fixture
def checkpoint(mongo_uri: str) -> CheckpointCallback:
    return CheckpointCallback(storage_uri=mongo_uri)


@pytest.fixture
def otel_callback() -> OtelEventCallback:
    return OtelEventCallback()


@pytest.fixture
def plugin(mongo_uri: str) -> SynapsePlugin:
    return SynapsePlugin(mongo_uri)
