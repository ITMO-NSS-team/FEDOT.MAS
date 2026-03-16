import asyncio

from fedotmas import MAW
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.maw.edge_cases.meta_generation")

META_MODEL = "openai/gpt-oss-120b"


async def two_stage_simple():
    _log.info("------ Scenario: two_stage_simple ------")
    try:
        maw = MAW(two_stage=True, meta_model=META_MODEL)
        config = await maw.generate_config("Explain what a REST API is")
        _log.info(
            "Generated config: {} agents, pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        state = await maw.build_and_run(config, "Explain what a REST API is")
        _log.info("Result state keys: {}", list(state.keys()))
        for key in state:
            if key != "user_query":
                _log.info("Output [{}]: {}", key, str(state[key])[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def single_stage_simple():
    _log.info("------ Scenario: single_stage_simple ------")
    try:
        maw = MAW(two_stage=False, meta_model=META_MODEL)
        config = await maw.generate_config("List 3 benefits of exercise")
        _log.info(
            "Generated config: {} agents, pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        state = await maw.build_and_run(config, "List 3 benefits of exercise")
        _log.info("Result state keys: {}", list(state.keys()))
        for key in state:
            if key != "user_query":
                _log.info("Output [{}]: {}", key, str(state[key])[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def complex_task():
    _log.info("------ Scenario: complex_task ------")
    try:
        task = (
            "Design a comprehensive comparison of microservices vs monolithic architecture "
            "for a mid-size e-commerce platform. Consider scalability, development speed, "
            "deployment complexity, team structure requirements, and cost implications. "
            "Provide concrete recommendations with trade-offs for each approach."
        )
        maw = MAW(two_stage=True, meta_model=META_MODEL)
        config = await maw.generate_config(task)
        _log.info(
            "Generated config: {} agents, pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        state = await maw.build_and_run(config, task)
        _log.info("Result state keys: {}", list(state.keys()))
        for key in state:
            if key != "user_query":
                _log.info("Output [{}]: {}", key, str(state[key])[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def minimal_task():
    _log.info("------ Scenario: minimal_task ------")
    try:
        maw = MAW(two_stage=True, meta_model=META_MODEL)
        config = await maw.generate_config("Summarize AI")
        _log.info(
            "Generated config: {} agents, pipeline_type={}",
            len(config.agents),
            config.pipeline.type,
        )
        state = await maw.build_and_run(config, "Summarize AI")
        _log.info("Result state keys: {}", list(state.keys()))
        for key in state:
            if key != "user_query":
                _log.info("Output [{}]: {}", key, str(state[key])[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def main():
    _log.info("Starting meta_generation edge cases")
    await two_stage_simple()
    await single_stage_simple()
    await complex_task()
    await minimal_task()
    _log.info("All meta_generation scenarios completed")


if __name__ == "__main__":
    asyncio.run(main())
