# media MCP server

Analyze and transcribe media files (audio, images, video) using multimodal LLMs.

## Tools

| Tool | Description |
|------|-------------|
| `transcribe_audio` | Transcribe audio from a local file or URL (mp3, wav, flac, ogg, etc.) |
| `analyze_image` | Analyze and describe an image from a local file or URL (jpg, png, gif, webp, etc.) |
| `analyze_video` | Analyze video content from a local file or URL (mp4, mov, webm, mpeg) |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | required | API key for the LLM provider |
| `OPENAI_BASE_URL` | optional | Custom API endpoint |
| `MEDIA_MODEL` | optional | Default model for all media tasks (default: `google/gemini-2.5-flash`) |
| `MEDIA_AUDIO_MODEL` | optional | Override model for audio transcription |
| `MEDIA_IMAGE_MODEL` | optional | Override model for image analysis |
| `MEDIA_VIDEO_MODEL` | optional | Override model for video analysis |

## Usage

```python
AgentConfig(tools=["media"])
```
