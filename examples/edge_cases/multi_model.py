import asyncio

from fedotmas import MAS, AgentConfig, PipelineConfig, StepConfig
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.multi_model")

MODELS = [
    "openai/gpt-4o-mini",
    "openrouter/qwen/qwen3-235b-a22b-2507",
    "openai/gpt-oss-120b",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
]


async def each_model_solo():
    _log.info("------ Scenario: each_model_solo ------")
    for model in MODELS:
        _log.info("Testing model: {}", model)
        try:
            config = PipelineConfig(
                agents=[
                    AgentConfig(
                        name="agent",
                        instruction="Answer in one sentence: {user_query}",
                        model=model,
                        output_key="answer",
                    ),
                ],
                pipeline=StepConfig(type="agent", agent_name="agent"),
            )
            mas = MAS()
            state = await mas.build_and_run(config, "What is gravity?")
            _log.info("Result state keys: {}", list(state.keys()))
            _log.info("Output [{}]: {}", model, state.get("answer", "")[:200])
            _log.info(
                "Tokens: prompt={} completion={} elapsed={:.1f}s",
                mas.total_prompt_tokens,
                mas.total_completion_tokens,
                mas.elapsed,
            )
        except Exception as e:
            _log.error("Model {} failed: {}", model, e)


async def cross_model_pipeline():
    _log.info("------ Scenario: cross_model_pipeline ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="researcher",
                    instruction="Research the topic: {user_query}. Provide key facts.",
                    model="qwen/qwen3-235b-a22b-2507",
                    output_key="research",
                ),
                AgentConfig(
                    name="writer",
                    instruction="Write a concise summary based on:\n{research}",
                    model="meta-llama/llama-3.3-70b-instruct",
                    output_key="draft",
                ),
                AgentConfig(
                    name="editor",
                    instruction="Polish and improve this text:\n{draft}",
                    model="openai/gpt-4o-mini",
                    output_key="final",
                ),
            ],
            pipeline=StepConfig(
                type="sequential",
                children=[
                    StepConfig(type="agent", agent_name="researcher"),
                    StepConfig(type="agent", agent_name="writer"),
                    StepConfig(type="agent", agent_name="editor"),
                ],
            ),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "Blockchain technology")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("final", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            mas.total_prompt_tokens,
            mas.total_completion_tokens,
            mas.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def default_model_fallback():
    _log.info("------ Scenario: default_model_fallback ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="default_agent",
                    instruction="Answer briefly: {user_query}",
                    output_key="answer",
                ),
            ],
            pipeline=StepConfig(type="agent", agent_name="default_agent"),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "What is the speed of light?")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("answer", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            mas.total_prompt_tokens,
            mas.total_completion_tokens,
            mas.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def main():
    _log.info("Starting multi_model edge cases")
    await each_model_solo()
    await cross_model_pipeline()
    await default_model_fallback()
    _log.info("All multi_model scenarios completed")


if __name__ == "__main__":
    asyncio.run(main())
