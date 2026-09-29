"""Exercise the installed ProtoLLM connector with a local mock HTTP transport."""

import json
import os
from unittest.mock import patch

import httpx
import protollm.connectors
from review_report import review_report


def main() -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen/qwen3-32b"
        assert payload["messages"][1]["content"] == "Test report"
        calls.append(payload)
        return httpx.Response(200, json={
            "id": "offline-test",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Validation required",
            }}],
        })

    factory = protollm.connectors.create_llm_connector
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        def local_connector(url, **kwargs):
            return factory(url, http_client=client, **kwargs)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-key"}), patch.object(
            protollm.connectors, "create_llm_connector", local_connector,
        ):
            assert review_report("Test report", "qwen/qwen3-32b") == "Validation required"
    assert len(calls) == 1
    print("ProtoLLM connector smoke test passed (mock HTTP, no paid API call)")


if __name__ == "__main__":
    main()
