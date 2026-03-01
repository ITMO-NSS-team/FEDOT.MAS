import asyncio

from fedotmas import MAS, AgentConfig, PipelineConfig, StepConfig
from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.examples.pipeline_structures")

MODEL = "openai/gpt-4o-mini"


async def single_agent():
    _log.info("------ Scenario: single_agent ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="answerer",
                    instruction="Answer briefly: {user_query}",
                    model=MODEL,
                    output_key="answer",
                ),
            ],
            pipeline=StepConfig(type="agent", agent_name="answerer"),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "What is the capital of France?")
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


async def sequential_three():
    _log.info("------ Scenario: sequential_three ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="researcher",
                    instruction="Research the topic: {user_query}. Provide 3 key facts.",
                    model=MODEL,
                    output_key="research",
                ),
                AgentConfig(
                    name="analyst",
                    instruction="Analyze these facts and find patterns:\n{research}",
                    model=MODEL,
                    output_key="analysis",
                ),
                AgentConfig(
                    name="writer",
                    instruction="Write a concise summary combining:\n{research}\n\nAnalysis:\n{analysis}",
                    model=MODEL,
                    output_key="summary",
                ),
            ],
            pipeline=StepConfig(
                type="sequential",
                children=[
                    StepConfig(type="agent", agent_name="researcher"),
                    StepConfig(type="agent", agent_name="analyst"),
                    StepConfig(type="agent", agent_name="writer"),
                ],
            ),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "Benefits of renewable energy")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("summary", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            mas.total_prompt_tokens,
            mas.total_completion_tokens,
            mas.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def parallel_two():
    _log.info("------ Scenario: parallel_two ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="pros_analyst",
                    instruction="List 3 advantages of: {user_query}",
                    model=MODEL,
                    output_key="pros",
                ),
                AgentConfig(
                    name="cons_analyst",
                    instruction="List 3 disadvantages of: {user_query}",
                    model=MODEL,
                    output_key="cons",
                ),
                AgentConfig(
                    name="synthesizer",
                    instruction="Given pros:\n{pros}\n\nAnd cons:\n{cons}\n\nWrite a balanced verdict.",
                    model=MODEL,
                    output_key="verdict",
                ),
            ],
            pipeline=StepConfig(
                type="sequential",
                children=[
                    StepConfig(
                        type="parallel",
                        children=[
                            StepConfig(type="agent", agent_name="pros_analyst"),
                            StepConfig(type="agent", agent_name="cons_analyst"),
                        ],
                    ),
                    StepConfig(type="agent", agent_name="synthesizer"),
                ],
            ),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "Remote work")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("verdict", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            mas.total_prompt_tokens,
            mas.total_completion_tokens,
            mas.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def loop_critic():
    _log.info("------ Scenario: loop_critic ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="writer",
                    instruction=(
                        "Write a haiku about: {user_query}.\n"
                        "If there is feedback, incorporate it: {feedback}"
                    ),
                    model=MODEL,
                    output_key="draft",
                ),
                AgentConfig(
                    name="critic",
                    instruction=(
                        "Review this haiku:\n{draft}\n\n"
                        "If it follows 5-7-5 syllable structure, call the exit_loop tool.\n"
                        "Otherwise provide feedback for improvement."
                    ),
                    model=MODEL,
                    output_key="feedback",
                ),
            ],
            pipeline=StepConfig(
                type="loop",
                max_iterations=3,
                children=[
                    StepConfig(type="agent", agent_name="writer"),
                    StepConfig(type="agent", agent_name="critic"),
                ],
            ),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "autumn leaves falling")
        _log.info("Result state keys: {}", list(state.keys()))
        _log.info("Output: {}", state.get("draft", "")[:200])
        _log.info(
            "Tokens: prompt={} completion={} elapsed={:.1f}s",
            mas.total_prompt_tokens,
            mas.total_completion_tokens,
            mas.elapsed,
        )
    except Exception as e:
        _log.error("Scenario failed: {}", e)


async def nested_seq_par_seq():
    _log.info("------ Scenario: nested_seq_par_seq ------")
    try:
        config = PipelineConfig(
            agents=[
                AgentConfig(
                    name="fact_finder",
                    instruction="Find 3 interesting facts about: {user_query}",
                    model=MODEL,
                    output_key="facts",
                ),
                AgentConfig(
                    name="opinion_maker",
                    instruction="Give a strong opinion about: {user_query}",
                    model=MODEL,
                    output_key="opinion",
                ),
                AgentConfig(
                    name="editor",
                    instruction="Combine these into a polished paragraph:\nFacts: {facts}\nOpinion: {opinion}",
                    model=MODEL,
                    output_key="final",
                ),
            ],
            pipeline=StepConfig(
                type="sequential",
                children=[
                    StepConfig(
                        type="parallel",
                        children=[
                            StepConfig(type="agent", agent_name="fact_finder"),
                            StepConfig(type="agent", agent_name="opinion_maker"),
                        ],
                    ),
                    StepConfig(type="agent", agent_name="editor"),
                ],
            ),
        )
        mas = MAS()
        state = await mas.build_and_run(config, "Electric vehicles")
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


async def main():
    _log.info("Starting pipeline_structures edge cases")
    await single_agent()
    await sequential_three()
    await parallel_two()
    await loop_critic()
    await nested_seq_par_seq()
    _log.info("All pipeline_structures scenarios completed")


if __name__ == "__main__":
    asyncio.run(main())
