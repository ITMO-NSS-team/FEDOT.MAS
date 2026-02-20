import warnings

from fedotmas.main import MAS
from fedotmas.pipeline.models import AgentConfig, PipelineConfig, StepConfig
from fedotmas.pipeline.runner import PipelineResult

# litellm's Message.__init__ deletes None-valued attributes from instances,
# causing Pydantic to warn about missing fields during serialization.
# The warning is cosmetic, serialization works correctly.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module=r"pydantic\.main",
)

# ADK marks MCP tools as experimental; the warning is cosmetic.
warnings.filterwarnings(
    "ignore",
    message=r".*EXPERIMENTAL.*BASE_AUTHENTICATED_TOOL.*",
    category=UserWarning,
)

# LiteLLM fires an async callback outside a running event loop;
# the unawaited coroutine warning is harmless.
warnings.filterwarnings(
    "ignore",
    message=r".*async_success_handler.*never awaited",
    category=RuntimeWarning,
)

__all__ = [
    "MAS",
    "AgentConfig",
    "PipelineConfig",
    "PipelineResult",
    "StepConfig",
]
