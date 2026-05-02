from examples.gaia.run_gaia import extract_answer_from_state


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
