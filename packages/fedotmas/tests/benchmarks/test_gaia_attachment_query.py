from types import SimpleNamespace

from benchmarks.gaia.run_gaia import _task_query


def task():
    return SimpleNamespace(
        task_id="gaia-task",
        question="Count towers",
        file_path="/cache/blobs/abc",
        file_name="map.txt",
    )


def test_light_sandbox_query_requires_real_reader_without_invented_helper():
    query = _task_query(task(), {"sandbox-light": object(), "document": object()})
    assert "solve_with_code" not in query
    assert "Assign document" in query and "cannot open host filesystem" in query
    assert "never append" in query
    assert "Host file path: /cache/blobs/abc" in query
    assert "/cache/blobs/abc/map.txt" not in query


def test_code_agent_query_preserves_file_staging_contract():
    query = _task_query(task(), {"code-agent": object(), "document": object()})
    assert "solve_with_code(files=[path])" in query
    assert "do not copy" in query and "fresh sandbox" in query


def test_no_reader_does_not_promise_file_access():
    query = _task_query(task(), {"sandbox-light": object()})
    assert "report the missing capability" in query
    assert "solve_with_code" not in query


def test_text_task_does_not_receive_file_rules():
    item = task()
    item.file_path = item.file_name = ""
    assert _task_query(item, {}) == "Question: Count towers"
