"""Two-stage generation rules — pool↔pipeline validation, allowed_models passthrough."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas.config.settings import ModelConfig
from fedotmas.meta._adk_runner import LLMCallResult
from fedotmas.maw.models import AgentPoolConfig, MAWConfig


# ---------------------------------------------------------------------------
# Pool → Pipeline validation
# ---------------------------------------------------------------------------


class TestPoolPipelineExtraAgent:
    """Rule 1: _validate_against_pool rejects agents not in pool."""

    def test_extra_agent_rejected(self):
        from fedotmas.meta.pipeline_gen import PipelineGenerator

        pool = AgentPoolConfig(
            agents=[
                {"name": "a", "instruction": "do A"},
            ]
        )
        config = MAWConfig(
            agents=[
                {"name": "a", "instruction": "do A", "output_key": "ka"},
                {"name": "b", "instruction": "do B", "output_key": "kb"},
            ],
            pipeline={
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "a"},
                    {"type": "agent", "agent_name": "b"},
                ],
            },
        )
        with pytest.raises(ValueError, match="not in pool"):
            PipelineGenerator._validate_against_pool(config, pool)


class TestPoolPipelineSubsetOk:
    """Rule 2: subset of pool agents is valid."""

    def test_subset_accepted(self):
        from fedotmas.meta.pipeline_gen import PipelineGenerator

        pool = AgentPoolConfig(
            agents=[
                {"name": "a", "instruction": "do A"},
                {"name": "b", "instruction": "do B"},
                {"name": "c", "instruction": "do C"},
            ]
        )
        config = MAWConfig(
            agents=[
                {"name": "a", "instruction": "do A", "output_key": "ka"},
            ],
            pipeline={"type": "agent", "agent_name": "a"},
        )
        # Should not raise
        PipelineGenerator._validate_against_pool(config, pool)


# ---------------------------------------------------------------------------
# allowed_models passthrough
# ---------------------------------------------------------------------------


def _make_llm_result(output_key: str, raw_output: dict) -> LLMCallResult:
    return LLMCallResult(
        raw_output=raw_output,
        prompt_tokens=10,
        completion_tokens=20,
        elapsed=1.0,
    )


class TestAgentPassesAllowedModels:
    """Rule 3: agent.py passes allowed_models to run_meta_agent_call."""

    async def test_allowed_models_passed(self, two_agent_data):
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _make_llm_result("pipeline_config", two_agent_data)

        with (
            patch("fedotmas.meta.agent.run_meta_agent_call", side_effect=_capture),
            patch("fedotmas.meta.agent.get_server_descriptions", return_value={}),
        ):
            from fedotmas.meta.agent import generate_pipeline_config

            await generate_pipeline_config(
                "test task",
                meta_model=ModelConfig(model="openai/gpt-4o"),
                worker_models=[
                    ModelConfig(model="openai/gpt-4o"),
                    ModelConfig(model="openai/gpt-4o-mini"),
                ],
            )
            assert captured["allowed_models"] == ["openai/gpt-4o", "openai/gpt-4o-mini"]


class TestPoolGenPassesAllowedModels:
    """Rule 4: pool_gen.py passes allowed_models to run_meta_agent_call."""

    async def test_allowed_models_passed(self):
        pool_data = {
            "agents": [{"name": "a", "instruction": "x"}],
        }
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _make_llm_result("agent_pool", pool_data)

        with (
            patch("fedotmas.meta.pool_gen.run_meta_agent_call", side_effect=_capture),
            patch("fedotmas.meta.pool_gen.get_server_descriptions", return_value={}),
        ):
            from fedotmas.meta.pool_gen import PoolGenerator

            gen = PoolGenerator(
                meta_model=ModelConfig(model="openai/gpt-4o"),
                worker_models=[
                    ModelConfig(model="openai/gpt-4o"),
                    ModelConfig(model="openai/gpt-4o-mini"),
                ],
            )
            await gen.generate("test task")
            assert captured["allowed_models"] == ["openai/gpt-4o", "openai/gpt-4o-mini"]


class TestPipelineGenPassesAllowedModels:
    """Rule 5: pipeline_gen.py passes allowed_models to run_meta_agent_call."""

    async def test_allowed_models_passed(self, two_agent_data):
        pool = AgentPoolConfig(
            agents=[
                {"name": "researcher", "instruction": "Research"},
                {"name": "writer", "instruction": "Write"},
            ]
        )
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return _make_llm_result("pipeline_config", two_agent_data)

        with (
            patch(
                "fedotmas.meta.pipeline_gen.run_meta_agent_call", side_effect=_capture
            ),
            patch(
                "fedotmas.meta.pipeline_gen.get_server_descriptions", return_value={}
            ),
        ):
            from fedotmas.meta.pipeline_gen import PipelineGenerator

            gen = PipelineGenerator(
                meta_model=ModelConfig(model="openai/gpt-4o"),
                worker_models=[
                    ModelConfig(model="openai/gpt-4o"),
                    ModelConfig(model="openai/gpt-4o-mini"),
                ],
            )
            await gen.generate("test task", pool)
            assert captured["allowed_models"] == ["openai/gpt-4o", "openai/gpt-4o-mini"]
