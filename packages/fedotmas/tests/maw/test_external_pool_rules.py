"""Caller-supplied agent pool rules — reuse modes and what generation may rewrite."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas import MAW
from fedotmas._settings import ModelConfig
from fedotmas.maw.models import AgentPoolConfig
from fedotmas.meta._adk_runner import LLMCallResult


def _result(raw_output: dict) -> LLMCallResult:
    return LLMCallResult(
        raw_output=raw_output, prompt_tokens=10, completion_tokens=20, elapsed=1.0
    )


@pytest.fixture()
def existing() -> AgentPoolConfig:
    return AgentPoolConfig(
        agents=[
            {
                "name": "researcher",
                "instruction": "Their own prompt, written by them",
                "model": "openai/gpt-4o",
                "tools": ["urban.getproject"],
            }
        ]
    )


@pytest.fixture()
def one_agent_data() -> dict:
    return {
        "agents": [
            {
                "name": "researcher",
                "instruction": "Rewritten by the meta-agent",
                "model": "openai/gpt-4o-mini",
                "output_key": "research_result",
                "tools": [],
            }
        ],
        "pipeline": {"type": "agent", "agent_name": "researcher"},
    }


@pytest.fixture()
def maw() -> MAW:
    return MAW(
        meta_model=ModelConfig(model="openai/gpt-4o"),
        worker_models=[ModelConfig(model="openai/gpt-4o")],
        tool_catalog={"urban.getproject": "Get a project"},
    )


class TestReuseOnlySkipsPoolStage:
    """Rule 1: with reuse='only' no agents are invented."""

    async def test_pool_stage_not_called(self, maw, existing, one_agent_data):
        async def _pipeline(**kwargs):
            return _result(one_agent_data)

        with (
            patch(
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
                side_effect=_pipeline,
            ),
            patch("fedotmas.meta.maw_pool_stage.run_meta_agent_call") as pool_call,
        ):
            await maw.generate_config("task", existing_agents=existing, reuse="only")

        pool_call.assert_not_called()


class TestPreferOffersExistingAgents:
    """Rule 2: with reuse='prefer' the pool stage is told what already exists."""

    async def test_existing_agents_reach_the_pool_prompt(
        self, maw, existing, two_agent_data
    ):
        captured = {}

        async def _pool(**kwargs):
            captured.update(kwargs)
            return _result(
                {
                    "agents": [
                        {"name": "researcher", "instruction": "x"},
                        {"name": "writer", "instruction": "y"},
                    ]
                }
            )

        async def _pipeline(**kwargs):
            return _result(two_agent_data)

        with (
            patch(
                "fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_pool
            ),
            patch(
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
                side_effect=_pipeline,
            ),
        ):
            await maw.generate_config("task", existing_agents=existing)

        assert "EXISTING AGENTS" in captured["user_message"]
        assert "Their own prompt, written by them" in captured["user_message"]

    async def test_prompt_unchanged_without_existing_agents(self, maw, two_agent_data):
        captured = {}

        async def _pool(**kwargs):
            captured.update(kwargs)
            return _result(
                {
                    "agents": [
                        {"name": "researcher", "instruction": "x"},
                        {"name": "writer", "instruction": "y"},
                    ]
                }
            )

        async def _pipeline(**kwargs):
            return _result(two_agent_data)

        with (
            patch(
                "fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_pool
            ),
            patch(
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
                side_effect=_pipeline,
            ),
        ):
            await maw.generate_config("task")

        assert captured["user_message"] == "TASK: task"


class TestExternalAgentsSurviveGeneration:
    """Rule 3: an agent handed in is wired, never rewritten."""

    async def test_instruction_model_and_tools_restored(
        self, maw, existing, one_agent_data
    ):
        async def _pipeline(**kwargs):
            return _result(one_agent_data)

        with patch(
            "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
            side_effect=_pipeline,
        ):
            config = await maw.generate_config(
                "task", existing_agents=existing, reuse="only"
            )

        researcher = next(a for a in config.agents if a.name == "researcher")
        assert researcher.instruction == "Their own prompt, written by them"
        assert researcher.model == "openai/gpt-4o"
        assert researcher.tools == ["urban.getproject"]
        # Wiring is generation's to decide, and the pool entry carries none.
        assert researcher.output_key == "research_result"

    async def test_synthesized_agents_kept_as_generated(
        self, maw, existing, two_agent_data
    ):
        async def _pool(**kwargs):
            return _result(
                {
                    "agents": [
                        {"name": "researcher", "instruction": "x"},
                        {"name": "writer", "instruction": "y"},
                    ]
                }
            )

        async def _pipeline(**kwargs):
            return _result(two_agent_data)

        with (
            patch(
                "fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_pool
            ),
            patch(
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
                side_effect=_pipeline,
            ),
        ):
            config = await maw.generate_config("task", existing_agents=existing)

        assert next(a for a in config.agents if a.name == "researcher").instruction == (
            "Their own prompt, written by them"
        )
        assert next(a for a in config.agents if a.name == "writer").instruction == (
            "Write a report"
        )


class TestForeignModelsKeptOutOfGeneration:
    """Rule 4: a caller's model names never reach a prompt that forbids them."""

    async def test_pool_text_carries_no_models(self, maw, existing):
        captured = {}

        async def _pipeline(**kwargs):
            captured.update(kwargs)
            return _result(
                {
                    "agents": [
                        {
                            "name": "researcher",
                            "instruction": "x",
                            "model": "openai/gpt-4o",
                            "output_key": "research_result",
                        }
                    ],
                    "pipeline": {"type": "agent", "agent_name": "researcher"},
                }
            )

        with patch(
            "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
            side_effect=_pipeline,
        ):
            await maw.generate_config("task", existing_agents=existing, reuse="only")

        assert "researcher" in captured["user_message"]
        assert "model:" not in captured["user_message"]


class TestWiringSurvivesRestoration:
    """Rule 5: a reused agent keeps the inputs stage 2 wired into it."""

    async def test_added_state_refs_appended(self, maw, two_agent_data):
        pool = AgentPoolConfig(
            agents=[{"name": "writer", "instruction": "Write a report."}]
        )
        two_agent_data["agents"][1]["instruction"] = (
            "Write a report from {research_result}"
        )

        async def _pool(**kwargs):
            return _result(
                {
                    "agents": [
                        {"name": "researcher", "instruction": "x"},
                        {"name": "writer", "instruction": "y"},
                    ]
                }
            )

        async def _pipeline(**kwargs):
            return _result(two_agent_data)

        with (
            patch(
                "fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_pool
            ),
            patch(
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
                side_effect=_pipeline,
            ),
        ):
            config = await maw.generate_config("task", existing_agents=pool)

        writer = next(a for a in config.agents if a.name == "writer")
        assert writer.instruction.startswith("Write a report.")
        assert "{research_result?}" in writer.instruction


class TestModelFallback:
    """Rule 6: an entry without a model keeps the one generation assigned."""

    async def test_assigned_model_kept(self, maw, one_agent_data):
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "Their prompt"}]
        )

        async def _pipeline(**kwargs):
            return _result(one_agent_data)

        with patch(
            "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
            side_effect=_pipeline,
        ):
            config = await maw.generate_config(
                "task", existing_agents=pool, reuse="only"
            )

        assert config.agents[0].model == "openai/gpt-4o-mini"
        assert config.agents[0].instruction == "Their prompt"


class TestPoolGuards:
    """Rule 7: unusable pools are caught before an LLM call is paid for."""

    async def test_empty_pool_is_an_ordinary_generation(self, two_agent_data):
        single = MAW(
            meta_model=ModelConfig(model="openai/gpt-4o"),
            worker_models=[ModelConfig(model="openai/gpt-4o")],
            two_stage=False,
        )

        async def _single(**kwargs):
            return _result(two_agent_data)

        with (
            patch(
                "fedotmas.meta.maw_single_stage.run_meta_agent_call",
                side_effect=_single,
            ),
            patch("fedotmas.meta.maw_pool_stage.run_meta_agent_call") as pool_call,
        ):
            await single.generate_config(
                "task", existing_agents=AgentPoolConfig(agents=[])
            )

        pool_call.assert_not_called()

    async def test_model_without_provider_prefix_rejected_upfront(self, maw):
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "x", "model": "gpt-4o"}]
        )

        with patch("fedotmas.meta.maw_pool_stage.run_meta_agent_call") as pool_call:
            with pytest.raises(ValueError, match="provider prefix"):
                await maw.generate_config("task", existing_agents=pool)

        pool_call.assert_not_called()


class TestPreferPathGuards:
    """Rule 10: the default reuse mode is under the same guards as the strict one."""

    async def test_foreign_models_kept_out_of_the_pool_prompt(self, maw, two_agent_data):
        pool = AgentPoolConfig(
            agents=[
                {
                    "name": "researcher",
                    "instruction": "Theirs",
                    "model": "openrouter/deepseek-v4-pro",
                }
            ]
        )
        captured = {}

        async def _pool(**kwargs):
            captured.update(kwargs)
            return _result(
                {
                    "agents": [
                        {"name": "researcher", "instruction": "x"},
                        {"name": "writer", "instruction": "y"},
                    ]
                }
            )

        async def _pipeline(**kwargs):
            return _result(two_agent_data)

        with (
            patch("fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_pool),
            patch(
                "fedotmas.meta.maw_pipeline_stage.run_meta_agent_call",
                side_effect=_pipeline,
            ),
        ):
            config = await maw.generate_config("task", existing_agents=pool)

        assert "openrouter/deepseek-v4-pro" not in captured["user_message"]
        assert captured["allowed_models"] == ["openai/gpt-4o"]
        # Stripped for the prompt only; the caller's model is what ships.
        researcher = next(a for a in config.agents if a.name == "researcher")
        assert researcher.model == "openrouter/deepseek-v4-pro"


class TestStateRefMatching:
    """Rule 11: a wired key is recognised whole, not by prefix."""

    def test_prefix_of_an_existing_ref_is_still_added(self):
        from fedotmas.maw.maw import _keep_added_state_refs

        out = _keep_added_state_refs(
            "Use {data_summary} well.", "Use {data_summary?} and {data?}"
        )
        assert "{data?}" in out
        assert out.startswith("Use {data_summary} well.")

    def test_nothing_appended_when_already_wired(self):
        from fedotmas.maw.maw import _keep_added_state_refs

        assert (
            _keep_added_state_refs("Use {data}.", "Use {data?}.") == "Use {data}."
        )


class TestEmptyPoolUnderStrictReuse:
    """Rule 12: 'invent nothing' over an empty pool is a mistake, not a mode switch."""

    async def test_raises_before_any_call(self, maw):
        with patch("fedotmas.meta.maw_pool_stage.run_meta_agent_call") as pool_call:
            with pytest.raises(ValueError, match="pool is empty"):
                await maw.generate_config(
                    "task", existing_agents=AgentPoolConfig(agents=[]), reuse="only"
                )

        pool_call.assert_not_called()
