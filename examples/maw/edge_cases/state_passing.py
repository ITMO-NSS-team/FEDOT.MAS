import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.state_passing")

MODEL = "openai/gpt-4o-mini"


async def angle_bracket_vars():
    _log.info("------ Scenario: angle_bracket_vars ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="greeter",
                    instruction="Greet the user about <user_query> in a friendly way.",
                    model=MODEL,
                    output_key="greeting",
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="greeter"),
        )
        maw = MAW()
        state = await maw.build_and_run(config, "machine learning")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("greeting", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def curly_bracket_vars():
    _log.info("------ Scenario: curly_bracket_vars ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="explainer",
                    instruction="Explain {user_query} in one sentence.",
                    model=MODEL,
                    output_key="explanation",
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="explainer"),
        )
        maw = MAW()
        state = await maw.build_and_run(config, "quantum computing")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("explanation", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def initial_state_injection():
    _log.info("------ Scenario: initial_state_injection ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="contextual",
                    instruction="Given this context: {context}\n\nAnswer: {user_query}",
                    model=MODEL,
                    output_key="answer",
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="contextual"),
        )
        maw = MAW()
        state = await maw.build_and_run(
            config,
            "What should I focus on?",
            initial_state={
                "context": "The user is a junior developer learning Python and interested in web development."
            },
        )
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


async def large_state_chain():
    _log.info("------ Scenario: large_state_chain ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="generator_a",
                    instruction="Write a detailed 500-word essay about: {user_query}",
                    model=MODEL,
                    output_key="essay_a",
                ),
                MAWAgentConfig(
                    name="generator_b",
                    instruction="Expand on this essay with 500 more words of analysis:\n{essay_a}",
                    model=MODEL,
                    output_key="essay_b",
                ),
                MAWAgentConfig(
                    name="generator_c",
                    instruction="Write a 500-word conclusion synthesizing:\n{essay_a}\n\n{essay_b}",
                    model=MODEL,
                    output_key="essay_c",
                ),
            ],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="generator_a"),
                    MAWStepConfig(type="agent", agent_name="generator_b"),
                    MAWStepConfig(type="agent", agent_name="generator_c"),
                ],
            ),
        )
        maw = MAW()
        state = await maw.build_and_run(config, "The future of artificial intelligence")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("essay_a length: {} chars", len(state.get("essay_a", "")))
        _log.info("essay_b length: {} chars", len(state.get("essay_b", "")))
        _log.info("essay_c length: {} chars", len(state.get("essay_c", "")))
        _log.info("Output: {}", state.get("essay_c", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            maw.total_prompt_tokens,
            maw.total_completion_tokens,
            maw.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def missing_var_graceful():
    _log.info("------ Scenario: missing_var_graceful ------")
    try:
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="tolerant",
                    instruction="Answer: {user_query}. Extra context: {nonexistent}",
                    model=MODEL,
                    output_key="answer",
                ),
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="tolerant"),
        )
        maw = MAW()
        state = await maw.build_and_run(config, "What is 2+2?")
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
    _log.info("Starting state_passing edge cases")
    await angle_bracket_vars()
    await curly_bracket_vars()
    await initial_state_injection()
    await large_state_chain()
    await missing_var_graceful()
    _log.info("All state_passing scenarios completed")


if __name__ == "__main__":
    asyncio.run(main())
