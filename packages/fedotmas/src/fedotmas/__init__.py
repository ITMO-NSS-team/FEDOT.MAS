import warnings

from fedotmas._settings import ModelConfig
from fedotmas.mas.mas import MAS
from fedotmas.mas.models import MASConfig
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.mcp._config import (
    HttpMCPServer,
    MCPServerConfig,
    StdioMCPServer,
    directory_server,
    http_server,
    npx_server,
    uvx_server,
    workspace_server,
)


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
    "MAW",
    "MASConfig",
    "MAWConfig",
    "ModelConfig",
    "HttpMCPServer",
    "StdioMCPServer",
    "MCPServerConfig",
    "http_server",
    "directory_server",
    "workspace_server",
    "npx_server",
    "uvx_server",
]
