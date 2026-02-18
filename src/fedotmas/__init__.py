import warnings

from fedotmas.main import MAS
from fedotmas.pipeline.models import AgentConfig, PipelineConfig, StepConfig

# litellm's Message.__init__ deletes None-valued attributes from instances,
# causing Pydantic to warn about missing fields during serialization.
# The warning is cosmetic, serialization works correctly.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module=r"pydantic\.main",
)

__all__ = [
    "MAS",
    "AgentConfig",
    "PipelineConfig",
    "StepConfig",
]
