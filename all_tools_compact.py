# compressed descriptions

COMPRESSED_EXTERNAL_SERVER_DESCRIPTIONS = {
    "filesystem": """
Secure file operations in allowed directories only.

Tools:
- read_file, read_multiple_files, get_file_info
- write_file, create_directory, move_file
- list_directory, search_files, directory_tree

Important:
- Works only inside authorized paths
- Good for file reads, edits, search, and tree listing
""",
    "youtube-transcript": """
Get YouTube transcript and video metadata.

Tools:
- get_transcript: transcript without timestamps
- get_timed_transcript: transcript with timestamps
- get_video_info: title, author, duration, and similar metadata

Important:
- Default language is `en`
- Can use manual or auto captions
- Long transcripts may be split into pages
""",
    "sequential-thinking": """
Use for step-by-step reasoning on hard or unclear tasks.

Tool:
- sequential_thinking: lets the model think in steps, revise, and branch

Important:
- Best for decomposition, planning, and course correction
- Logging can be disabled with `DISABLE_THOUGHT_LOGGING=true`
""",
}

COMPRESSED_CURRENT_SERVER_DESCRIPTIONS = {
    "sequential-thinking": """
Step-by-step reasoning for hard or unclear tasks.

Tool:
- sequential_thinking: think in steps, revise, branch, keep context

Important:
- Use for decomposition, planning, and course correction
- Logging can be disabled with `DISABLE_THOUGHT_LOGGING=true`
""",
    "browser-usage": """
Browser automation for multi-step web tasks.

Can:
- Open pages, click, type, navigate, and extract data
- Work with dynamic or JavaScript-heavy sites
- Follow natural-language instructions across many steps

Important:
- Use for web workflows, forms, research, and link verification
- Use this for dynamic multi-step browser workflows; use `web-search` for normal search and page extraction
""",
    "web-search": """
Web research toolkit.

Tools:
- search: search the web and academic sources
- extract: get the actual content of a URL as clean Markdown
- map: list pages in a domain
- screenshot_and_save
- screenshot_and_analyze

Important:
- Use `extract` when you need page content
- Use `map` first when you need site structure
""",
    "download-url-content": """
Download files from URLs to local storage.

Can:
- Download one file or many files
- Save with a custom filename

Important:
- Use this when you need a real local file from a URL
- Typical max file size is 500MB
- Filenames are sanitized and collisions are handled
""",
    "file-analysis": """
Read and extract content from local files.

Tools:
- read_pdf, read_docx, read_pptx, read_xlsx_xls: documents and spreadsheets
- list_zip_contents, extract_and_list_zip: ZIP archives
- read_image: analyze a local image, supports custom prompt
- extract_text: plain text, code, markup, and data files

Important:
- Use for local docs, archives, images, code, and text/data files
- Prefer this over media-analysis for normal local documents and text files
""",
    "media-analysis": """
Analyze audio, video, and images.

Tools:
- transcribe_audio: audio transcription
- analyze_video: video description, transcription, or analysis
- analyze_image: image analysis

Inputs:
- Local files
- HTTP/HTTPS URLs
- `gs://` for supported audio and image inputs

Important:
- Supports custom prompts
- Use this for media files; use file-analysis for normal docs and text files
""",
    "youtube-transcript": """
Get YouTube transcript and video metadata.

Tools:
- get_transcript: transcript without timestamps
- get_timed_transcript: transcript with timestamps
- get_video_info: title, author, duration, and similar metadata

Important:
- Default language is `en`
- Can use manual or auto captions
- Long transcripts may be split into pages
""",
    "e2b-sandbox": """
Remote sandbox for safe Python and shell execution.

Tools:
- e2b_create_sandbox_and_return_id
- e2b_upload_file, e2b_download_file
- e2b_run_code, e2b_run_command

Important:
- Use this when built-in tools are not enough and you need custom code or shell commands
- Typical flow: create sandbox -> upload files if needed -> run code/commands -> download results if needed
- Call `plt.show()` for matplotlib output
- Use `print()` so results appear in tool output
- Output can include text, HTML, Markdown, images, SVG, PDF, JSON, and more
""",
}
