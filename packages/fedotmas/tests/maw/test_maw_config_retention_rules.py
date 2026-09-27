from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.meta._result import MetaAgentResult


def _config() -> MAWConfig:
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "researcher",
                    "instruction": "Research the task",
                    "output_key": "findings",
                    "tools": ["websearch-searxng"],
                    "model": "openai/gpt-4o",
                }
            ],
            "pipeline": {"type": "agent", "agent_name": "researcher"},
        }
    )


@pytest.mark.asyncio
async def test_generate_config_is_retained_on_maw():
    config = _config()
    result = MetaAgentResult(config=config)

    with (
        patch("fedotmas.core.base.setup_logging"),
        patch(
            "fedotmas.maw.maw.generate_pipeline_config",
            new=AsyncMock(return_value=result),
        ),
    ):
        maw = MAW(two_stage=False, mcp_servers=[])
        generated = await maw.generate_config("research this")

    assert generated is config
    assert maw.generated_config is generated


@pytest.mark.asyncio
async def test_build_and_run_retains_caller_supplied_config():
    config = _config()
    with (
        patch("fedotmas.core.base.setup_logging"),
        patch(
            "fedotmas.core.base.BaseMAS.build_and_run",
            new=AsyncMock(return_value={"findings": "done"}),
        ),
    ):
        maw = MAW(mcp_servers=[])
        assert maw.generated_config is None
        result = await maw.build_and_run(config, "task")

    assert result == {"findings": "done"}
    assert maw.generated_config is config


@pytest.mark.asyncio
async def test_failed_meta_generation_usage_is_retained_on_maw():
    error = RuntimeError("invalid structured response")
    error.prompt_tokens = 11
    error.completion_tokens = 7
    error.elapsed = 2.5
    with (
        patch("fedotmas.core.base.setup_logging"),
        patch(
            "fedotmas.maw.maw.generate_pipeline_config",
            new=AsyncMock(side_effect=error),
        ),
    ):
        maw = MAW(two_stage=False, mcp_servers=[])
        with pytest.raises(RuntimeError, match="invalid structured response"):
            await maw.generate_config("task")

    assert maw.meta_prompt_tokens == 11
    assert maw.meta_completion_tokens == 7
    assert maw.meta_elapsed == 2.5


@pytest.mark.asyncio
async def test_generate_config_preserves_explicit_discovery_only_with_video_tools():
    config = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "source_finder",
                    "instruction": "Find and identify video sources",
                    "output_key": "sources",
                    "tools": ["websearch-tavily", "youtube-transcript"],
                    "research_mode": "discovery_only",
                }
            ],
            "pipeline": {"type": "agent", "agent_name": "source_finder"},
        }
    )
    with (
        patch("fedotmas.core.base.setup_logging"),
        patch(
            "fedotmas.maw.maw.generate_pipeline_config",
            new=AsyncMock(return_value=MetaAgentResult(config=config)),
        ),
    ):
        maw = MAW(two_stage=False, mcp_servers=[])
        generated = await maw.generate_config("Find a video source")

    assert generated.agents[0].research_mode == "discovery_only"
