from __future__ import annotations

import os
from functools import cache
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import AudioUrl, BinaryContent, ImageUrl, VideoUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

_DEFAULT_MODEL = os.getenv("MEDIA_MODEL", "google/gemini-2.5-flash")
_AUDIO_MODEL = os.getenv("MEDIA_AUDIO_MODEL", _DEFAULT_MODEL)
_IMAGE_MODEL = os.getenv("MEDIA_IMAGE_MODEL", _DEFAULT_MODEL)
_VIDEO_MODEL = os.getenv("MEDIA_VIDEO_MODEL", _DEFAULT_MODEL)

mcp = FastMCP("media")


@cache
def _make_model(model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )


_PROMPTS = {
    "audio": (
        "You are an audio transcription specialist. "
        "Transcribe the audio content accurately, word-for-word, "
        "using proper punctuation and formatting. "
        "Return ONLY the transcribed text."
    ),
    "image": (
        "You are an image analysis specialist. "
        "Analyze images and provide accurate, detailed responses. "
        "Be objective and factual. "
        "Return ONLY your analysis or answer to the user's question."
    ),
    "video": (
        "You are a video analysis specialist. "
        "Analyze video content and provide accurate, detailed responses. "
        "Describe visual elements, actions, speech, and any relevant context. "
        "Return ONLY your analysis or answer to the user's question."
    ),
}


@cache
def _get_agent(kind: str) -> Agent[None, str]:
    models = {"audio": _AUDIO_MODEL, "image": _IMAGE_MODEL, "video": _VIDEO_MODEL}
    return Agent(
        model=_make_model(models[kind]),
        output_type=str,
        system_prompt=_PROMPTS[kind],
    )


def _content(path: str, url_cls: type[AudioUrl] | type[ImageUrl] | type[VideoUrl]):
    """Return appropriate pydantic-ai content part for a URL or local file."""
    if path.startswith(("http://", "https://", "gs://")):
        return url_cls(url=path)
    return BinaryContent.from_path(path)


class MediaResult(BaseModel):
    result: str = Field(default="", description="Analysis or transcription result")
    error: str | None = Field(default=None, description="Error message if failed")


@mcp.tool
async def transcribe_audio(
    file_path: Annotated[str, Field(description="Path to audio file or URL")],
    ctx: Context,
    prompt: Annotated[
        str, Field(description="Instruction for transcription")
    ] = "Transcribe this audio file accurately, word-for-word.",
) -> MediaResult:
    """Transcribe audio from a local file or URL.

    Supported formats: mp3, wav, flac, oga, ogg, aiff, aac, m4a, wma, opus.
    """
    try:
        result = await _get_agent("audio").run([prompt, _content(file_path, AudioUrl)])
        return MediaResult(result=result.output)
    except Exception as e:
        await ctx.error(f"Transcription failed: {e}")
        return MediaResult(error=str(e))


@mcp.tool
async def analyze_image(
    file_path: Annotated[str, Field(description="Path to image file or URL")],
    ctx: Context,
    prompt: Annotated[
        str, Field(description="Question or prompt about the image")
    ] = "What's in this image? Describe it in detail.",
) -> MediaResult:
    """Analyze and describe an image from a local file or URL.

    Supported formats: jpg, jpeg, png, gif, webp, bmp, tiff, svg, ico.
    """
    try:
        result = await _get_agent("image").run([prompt, _content(file_path, ImageUrl)])
        return MediaResult(result=result.output)
    except Exception as e:
        await ctx.error(f"Image analysis failed: {e}")
        return MediaResult(error=str(e))


@mcp.tool
async def analyze_video(
    file_path: Annotated[str, Field(description="Path to video file or URL")],
    ctx: Context,
    prompt: Annotated[
        str, Field(description="Instruction for video analysis")
    ] = "Describe what is happening in this video, including visual elements, actions, and any audio content.",
) -> MediaResult:
    """Analyze video content from a local file or URL.

    Supported formats: mp4, mov, webm, mpeg.
    """
    try:
        result = await _get_agent("video").run([prompt, _content(file_path, VideoUrl)])
        return MediaResult(result=result.output)
    except Exception as e:
        await ctx.error(f"Video analysis failed: {e}")
        return MediaResult(error=str(e))


def main():
    mcp.run(show_banner=False)
