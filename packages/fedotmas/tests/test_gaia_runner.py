from examples.gaia.run_gaia import (
    ProviderErrorCooldown,
    _gaia_provider_extra_body,
    _is_provider_error,
    extract_answer_from_state,
)


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
