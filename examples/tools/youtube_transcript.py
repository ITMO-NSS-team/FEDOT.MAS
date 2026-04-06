import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

_log = get_logger("fedotmas.examples.tools.youtube_transcript")

MODEL = "openai/gpt-4o-mini"

VIDEO_URL = "https://www.youtube.com/watch?v=u5GVOvC5rKY"


async def main():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="summarizer",
                instruction=(
                    "You are a video content analyst. For this video:\n"
                    "{video_url}\n\n"
                    "1. Use get_video_info to fetch metadata.\n"
                    "2. Use get_available_languages to check which transcript "
                    "languages are available.\n"
                    "3. Use get_transcript with an available language code "
                    "(prefer 'en', fall back to any available language).\n\n"
                    "Provide a concise summary that includes:\n"
                    "- Video title and channel\n"
                    "- Main topics discussed\n"
                    "- Key takeaways (3-5 bullet points)"
                ),
                model=MODEL,
                output_key="result",
                tools=["youtube-transcript"],
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="summarizer"),
    )

    maw = MAW(mcp_servers=["youtube-transcript"])
    state = await maw.build_and_run(
        config,
        VIDEO_URL,
    )

    _log.info("Result: {}", str(state.get("result", "")))
    _log.info(
        "Tokens: prompt={} completion={} elapsed={:.1f}s",
        maw.total_prompt_tokens,
        maw.total_completion_tokens,
        maw.elapsed,
    )


if __name__ == "__main__":
    asyncio.run(main())
