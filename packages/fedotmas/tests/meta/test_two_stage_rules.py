"""Two-stage generation rules — pool↔pipeline validation, allowed_models passthrough."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas._settings import ModelConfig
from fedotmas.meta._adk_runner import LLMCallResult
from fedotmas.meta._helpers import validate_allowed_models
from fedotmas.maw.models import AgentPoolConfig, MAWConfig


# ---------------------------------------------------------------------------
# Pool → Pipeline validation
# ---------------------------------------------------------------------------


class TestPoolPipelineExtraAgent:
    """Rule 1: _validate_against_pool rejects agents not in pool."""

    def test_extra_agent_rejected(self):
        from fedotmas.meta.maw_pipeline_stage import PipelineGenerator

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
        from fedotmas.meta.maw_pipeline_stage import PipelineGenerator

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
            patch("fedotmas.meta.maw_single_stage.run_meta_agent_call", side_effect=_capture),
            patch("fedotmas.meta.maw_single_stage.get_server_descriptions", return_value={}),
        ):
            from fedotmas.meta.maw_single_stage import generate_pipeline_config

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
            patch("fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_capture),
            patch("fedotmas.meta.maw_pool_stage.get_server_descriptions", return_value={}),
        ):
            from fedotmas.meta.maw_pool_stage import PoolGenerator

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
    """Rule 5: maw_pipeline_stage.py passes allowed_models to run_meta_agent_call."""

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
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call", side_effect=_capture
            ),
            patch(
                "fedotmas.meta.maw_pipeline_stage.get_server_descriptions", return_value={}
            ),
        ):
            from fedotmas.meta.maw_pipeline_stage import PipelineGenerator

            gen = PipelineGenerator(
                meta_model=ModelConfig(model="openai/gpt-4o"),
                worker_models=[
                    ModelConfig(model="openai/gpt-4o"),
                    ModelConfig(model="openai/gpt-4o-mini"),
                ],
            )
            await gen.generate("test task", pool)
            assert captured["allowed_models"] == ["openai/gpt-4o", "openai/gpt-4o-mini"]


# ---------------------------------------------------------------------------
# validate_allowed_models
# ---------------------------------------------------------------------------


class TestValidateAllowedModelsAccepts:
    """Rule 6: model in allowed list does not raise."""

    def test_model_in_list(self, allowed_models):
        raw = {"agents": [{"name": "a", "model": allowed_models[0]}]}
        validate_allowed_models(raw, allowed_models)  # should not raise


class TestValidateAllowedModelsRejects:
    """Rule 7: model not in allowed list raises ValueError."""

    def test_model_not_in_list(self, allowed_models):
        raw = {"agents": [{"name": "bad_agent", "model": "unknown/model"}]}
        with pytest.raises(ValueError, match="bad_agent.*unknown/model"):
            validate_allowed_models(raw, allowed_models)


class TestValidateAllowedModelsNullModel:
    """Rule 8: agent with null model does not raise."""

    def test_null_model_ok(self, allowed_models):
        raw = {"agents": [{"name": "a", "model": None}]}
        validate_allowed_models(raw, allowed_models)  # should not raise

    def test_missing_model_key_ok(self, allowed_models):
        raw = {"agents": [{"name": "a"}]}
        validate_allowed_models(raw, allowed_models)  # should not raise


class TestValidateAllowedModelsMASFormat:
    """Rule 9: MAS format (coordinator + workers) is validated."""

    def test_mas_format_valid(self, allowed_models):
        raw = {
            "coordinator": {"name": "coord", "model": allowed_models[0]},
            "workers": [{"name": "w1", "model": allowed_models[-1]}],
        }
        validate_allowed_models(raw, allowed_models)  # should not raise

    def test_mas_format_invalid_worker(self, allowed_models):
        raw = {
            "coordinator": {"name": "coord", "model": allowed_models[0]},
            "workers": [{"name": "w1", "model": "bad/model"}],
        }
        with pytest.raises(ValueError, match="w1.*bad/model"):
            validate_allowed_models(raw, allowed_models)
