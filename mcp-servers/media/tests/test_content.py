from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import AudioUrl, BinaryContent, ImageUrl, VideoUrl

from mcp_media.server import _content


class TestContentRouting:
    @pytest.mark.parametrize("url_cls", [AudioUrl, ImageUrl, VideoUrl])
    def test_http_url_returns_url_type(self, url_cls):
        result = _content("https://example.com/file.mp3", url_cls)
        assert isinstance(result, url_cls)
        assert result.url == "https://example.com/file.mp3"

    @pytest.mark.parametrize("url_cls", [AudioUrl, ImageUrl, VideoUrl])
    def test_gs_url_returns_url_type(self, url_cls):
        result = _content("gs://bucket/file.wav", url_cls)
        assert isinstance(result, url_cls)

    def test_local_file_returns_binary_content(self, tmp_path: Path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"\x00" * 64)
        result = _content(str(wav), AudioUrl)
        assert isinstance(result, BinaryContent)
        assert result.data == b"\x00" * 64

    def test_local_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            _content("/nonexistent/file.mp3", AudioUrl)
