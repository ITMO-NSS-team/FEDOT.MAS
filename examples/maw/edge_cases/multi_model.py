import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig
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
            config = MAWConfig(
                agents=[
                    MAWAgentConfig(
                        name="agent",
                        instruction="Answer in one sentence: {user_query}",
                        model=model,
                        output_key="answer",
                    ),
                ],
                pipeline=MAWStepConfig(type="agent", agent_name="agent"),
            )
            maw = MAW()
            state = await maw.build_and_run(config, "What is gravity?")
            _log.info("Result state keys: {}", list(state.keys()))
            _log.info("Output [{}]: {}", model, state.get("answer", "")[:200])
            _log.info(
                "Tokens: prompt={} completion={} elapsed={:.1f}s",
                maw.total_prompt_tokens,
                maw.total_completion_tokens,
                maw.elapsed,
            )
        except Exception as e:
            _log.error("Model {} failed: {}", model, e)


async def cross_model_pipeline():
    _log.info("------ Scenario: cross_model_pipeline ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="researcher",
                    instruction="Research the topic: {user_query}. Provide key facts.",
                    model="qwen/qwen3-235b-a22b-2507",
                    output_key="research",
                ),
                MAWAgentConfig(
                    name="writer",
                    instruction="Write a concise summary based on:\n{research}",
                    model="meta-llama/llama-3.3-70b-instruct",
                    output_key="draft",
                ),
                MAWAgentConfig(
                    name="editor",
                    instruction="Polish and improve this text:\n{draft}",
                    model="openai/gpt-4o-mini",
                    output_key="final",
                ),
            ],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="researcher"),
                    MAWStepConfig(type="agent", agent_name="writer"),
                    MAWStepConfig(type="agent", agent_name="editor"),
                ],
            ),
        )
        maw = MAW()
        state = await maw.build_and_run(config, "Blockchain technology")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("final", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def default_model_fallback():
    _log.info("------ Scenario: default_model_fallback ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="default_agent",
                    instruction="Answer briefly: {user_query}",
                    output_key="answer",
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="default_agent"),
        )
        maw = MAW()
        state = await maw.build_and_run(config, "What is the speed of light?")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("answer", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
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
