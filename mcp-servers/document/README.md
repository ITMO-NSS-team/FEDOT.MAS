# document MCP server

Extract text from documents and archives using [MarkItDown](https://github.com/microsoft/markitdown).

## Tools

| Tool | Description |
|------|-------------|
| `read_document` | Read a document and extract text as markdown (PDF, DOCX, PPTX, XLSX, CSV, HTML, etc.) |
| `list_zip_contents` | List contents of a ZIP archive without extracting |
| `extract_zip` | Extract a ZIP archive to the local filesystem |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | optional | Required when `describe_images=True` for embedded image descriptions |
| `OPENAI_BASE_URL` | optional | Custom API endpoint for vision model |
| `DOCUMENT_VISION_MODEL` | optional | Vision model for image descriptions (default: `google/gemini-2.5-flash`) |

## Usage

```python
AgentConfig(tools=["document"])
```
