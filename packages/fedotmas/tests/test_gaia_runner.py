from benchmarks.gaia.run_gaia import (
    DEFAULT_GAIA_MCP_SERVERS,
    ProviderErrorCooldown,
    _assert_model_list_contains,
    _gaia_provider_extra_body,
    _media_document_model_configs,
    _model_ids_from_response,
    _is_provider_error,
    extract_answer_from_state,
    root_cause_summary,
)
from fedotmas.plugins import WebSearchLimitExceeded
from tenacity import Future, RetryError


def test_extract_answer_ignores_user_query_solution_example():
    state = {
        "user_query": "Example: <solution>42</solution>",
        "result": "",
    }

    assert extract_answer_from_state(state) == ""


def test_extract_answer_uses_last_non_query_solution():
    state = {
        "user_query": "Example: <solution>42</solution>",
        "draft": "<solution>wrong</solution>",
        "final": "<solution>right</solution>",
    }

    assert extract_answer_from_state(state) == "right"


def test_gaia_default_servers_include_file_tools():
    assert "document" in DEFAULT_GAIA_MCP_SERVERS
    assert "media" in DEFAULT_GAIA_MCP_SERVERS


def test_gaia_provider_extra_body_from_env(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_GAIA_PROVIDER_IGNORE", "Azure,OpenAI")
    monkeypatch.setenv("FEDOTMAS_GAIA_PROVIDER_ONLY", "Chutes")
    monkeypatch.setenv("FEDOTMAS_GAIA_PROVIDER_ALLOW_FALLBACKS", "0")
    monkeypatch.setenv("FEDOTMAS_GAIA_PROVIDER_REQUIRE_PARAMETERS", "1")
    monkeypatch.setenv("FEDOTMAS_GAIA_PROVIDER_SORT_BY", "throughput")
    monkeypatch.setenv("FEDOTMAS_GAIA_PROVIDER_SORT_PARTITION", "none")

    assert _gaia_provider_extra_body() == {
        "provider": {
            "ignore": ["Azure", "OpenAI"],
            "only": ["Chutes"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "sort": {
                "by": "throughput",
                "partition": "none",
            },
        }
    }


def test_detects_provider_errors():
    error = RuntimeError(
        "Error code: 403 - {'error': {'message': 'Provider returned error', "
        "'metadata': {'provider_name': 'Azure'}}}"
    )

    assert _is_provider_error(error)


def test_provider_cooldown_error_is_provider_error():
    assert _is_provider_error(ProviderErrorCooldown("temporarily blocked"))


def test_root_cause_unwraps_retry_error_exception_group():
    future = Future(1)
    future.set_exception(
        ExceptionGroup(
            "wrapped",
            [
                RuntimeError("noise"),
                WebSearchLimitExceeded("web search limit exceeded"),
            ],
        )
    )

    summary = root_cause_summary(RetryError(future))

    assert summary["root_cause"] == "resource_limit.web_search"
    assert summary["last_exception"] == "WebSearchLimitExceeded"
    assert summary["wrapper_exception"] == "RetryError"


def test_model_ids_from_openrouter_response():
    assert _model_ids_from_response(
        {"data": [{"id": "openai/gpt-5-mini"}, {"id": "google/gemini"}]}
    ) == {"openai/gpt-5-mini", "google/gemini"}


def test_model_healthcheck_rejects_missing_model():
    payload = '{"data": [{"id": "openai/gpt-5-mini"}]}'

    try:
        _assert_model_list_contains(
            payload,
            "google/gemini-2.5-flash",
            base_url="https://openrouter.ai/api/v1",
        )
    except RuntimeError as exc:
        assert "google/gemini-2.5-flash" in str(exc)
    else:
        raise AssertionError("missing model should fail healthcheck")


def test_media_document_model_configs_follow_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MEDIA_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("MEDIA_AUDIO_MODEL", "google/gemini-audio")
    monkeypatch.setenv("DOCUMENT_VISION_MODEL", "openai/gpt-4o")

    configs = dict(_media_document_model_configs())

    assert configs["MEDIA_MODEL"].model == "openai/gpt-4o-mini"
    assert configs["MEDIA_AUDIO_MODEL"].model == "google/gemini-audio"
    assert configs["MEDIA_IMAGE_MODEL"].model == "openai/gpt-4o-mini"
    assert configs["DOCUMENT_VISION_MODEL"].model == "openai/gpt-4o"
    assert configs["MEDIA_MODEL"].api_base == "https://openrouter.ai/api/v1"
