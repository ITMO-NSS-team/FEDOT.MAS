from __future__ import annotations

import asyncio
import json
import sys
import types
import zipfile
from types import SimpleNamespace

import httpx
import pytest
from mcp_code_agent import server


class FakeSandbox:
    def __init__(self, execution_results):
        self.execution_results = list(execution_results)
        self.uploads = {}
        self.codes = []
        self.killed = False
        self.files = SimpleNamespace(write=self.write)

    async def write(self, path, content):
        self.uploads[path] = content

    async def run_code(self, code, **kwargs):
        self.codes.append((code, kwargs))
        return self.execution_results.pop(0)

    async def kill(self):
        self.killed = True


def execution(*, stdout=(), error=None):
    return SimpleNamespace(
        logs=SimpleNamespace(stdout=list(stdout), stderr=[]),
        error=error,
        results=[],
    )


def configure(monkeypatch, sandbox=None):
    monkeypatch.setenv("CODE_AGENT_API_KEY", "code-agent-test-secret")
    monkeypatch.setenv("CODE_AGENT_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("CODE_AGENT_MODEL", "test-model")
    monkeypatch.setenv("E2B_API_KEY", "sandbox-test-secret")
    if sandbox is not None:
        fake_module = types.ModuleType("e2b_code_interpreter")

        class AsyncSandbox:
            @staticmethod
            async def create(**kwargs):
                configure.sandbox_options = kwargs
                return sandbox

        fake_module.AsyncSandbox = AsyncSandbox
        monkeypatch.setitem(sys.modules, "e2b_code_interpreter", fake_module)


def script_model(monkeypatch, actions, *, captured=None, tokens=(10, 4)):
    queue = list(actions)

    async def request(_client, _settings, messages, usage):
        usage.llm_invocations += 1
        usage.available = True
        usage.prompt_tokens += tokens[0]
        usage.completion_tokens += tokens[1]
        usage.total_tokens += sum(tokens)
        if captured is not None:
            captured.append(json.loads(json.dumps(messages)))
        return queue.pop(0)

    monkeypatch.setattr(server, "_request_model", request)


def test_mcp_exposes_only_solve_with_code():
    async def list_tools():
        return await server.mcp.list_tools()

    tools = asyncio.run(list_tools())
    assert [tool.name for tool in tools] == ["solve_with_code"]


@pytest.mark.asyncio
async def test_simple_arithmetic_and_token_usage(monkeypatch):
    sandbox = FakeSandbox([execution(stdout=["42"])])
    configure(monkeypatch, sandbox)
    script_model(
        monkeypatch,
        [
            {"action": "execute", "code": "print(6 * 7)"},
            {
                "action": "finish",
                "status": "completed",
                "answer": "42",
                "evidence": ["6 × 7 = 42"],
            },
        ],
    )

    result = await server._solve("Calculate 6 times 7", [], "", 3, 5, 1_000)

    assert result.status == "completed"
    assert result.answer == "42"
    assert result.steps_taken == 1
    assert result.usage.llm_invocations == 2
    assert result.usage.total_tokens == 28
    assert sandbox.codes[0][0] == "print(6 * 7)"
    assert sandbox.killed
    assert configure.sandbox_options["allow_internet_access"] is False


@pytest.mark.asyncio
async def test_csv_filter_stages_only_explicit_file(monkeypatch, tmp_path):
    source = tmp_path / "orders.csv"
    source.write_text("region,amount\neast,40\nwest,80\neast,15\n")
    sandbox = FakeSandbox([execution(stdout=["row 3: west, 80"])])
    configure(monkeypatch, sandbox)
    script_model(
        monkeypatch,
        [
            {
                "action": "execute",
                "code": "import pandas as pd; print(pd.read_csv('/tmp/code_agent/input_0.csv').query(\"region == 'west'\"))",
            },
            {
                "action": "finish",
                "answer": "80",
                "evidence": ["orders.csv row 3 has region west and amount 80"],
            },
        ],
    )

    result = await server._solve(
        "Find the west-region order amount",
        [str(source)],
        "Only filter the west region",
        3,
        5,
        1_000,
    )

    assert result.answer == "80"
    assert result.files_used == ["orders.csv"]
    assert sandbox.uploads == {"/tmp/code_agent/input_0.csv": source.read_bytes()}


@pytest.mark.asyncio
async def test_xlsx_is_staged_and_available_for_inspection(monkeypatch, tmp_path):
    source = tmp_path / "inventory.xlsx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "xl/workbook.xml", "<workbook><sheet name='Stock'/></workbook>"
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml", "<sheetData><row r='8'/></sheetData>"
        )
    sandbox = FakeSandbox([execution(stdout=["Stock row 8: SKU C-17, qty 4"])])
    configure(monkeypatch, sandbox)
    script_model(
        monkeypatch,
        [
            {"action": "execute", "code": "print('inspect workbook')"},
            {
                "action": "finish",
                "answer": "4",
                "evidence": ["Sheet 'Stock', row 8: SKU C-17, quantity 4"],
            },
        ],
    )

    result = await server._solve(
        "Find inventory for SKU C-17", [str(source)], "", 3, 5, 1_000
    )

    assert result.status == "completed"
    assert sandbox.uploads["/tmp/code_agent/input_0.xlsx"].startswith(b"PK")
    assert "Sheet 'Stock', row 8" in result.evidence[0]


@pytest.mark.asyncio
async def test_code_repairs_after_python_error(monkeypatch):
    sandbox = FakeSandbox(
        [
            execution(
                error=SimpleNamespace(name="SyntaxError", value="invalid syntax")
            ),
            execution(stdout=["answer=12"]),
        ]
    )
    configure(monkeypatch, sandbox)
    script_model(
        monkeypatch,
        [
            {"action": "execute", "code": "print(6 *"},
            {"action": "execute", "code": "print(6 * 2)"},
            {
                "action": "finish",
                "answer": "12",
                "evidence": ["6 × 2 = 12"],
            },
        ],
    )

    result = await server._solve("Multiply", [], "", 3, 5, 1_000)

    assert result.status == "completed"
    assert result.steps_taken == 2
    assert result.telemetry["execution_failures"] == 1
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_missing_file_returns_machine_readable_error(monkeypatch, tmp_path):
    configure(monkeypatch)
    missing = tmp_path / "missing.csv"

    result = await server._solve("Read file", [str(missing)], "", 3, 5, 1_000)

    assert result.status == "failed"
    assert result.error_code == "CODE_AGENT_FILE_NOT_FOUND"
    assert "missing.csv" in result.errors[0]
    assert result.usage.llm_invocations == 0


@pytest.mark.asyncio
async def test_execution_without_e2b_key_returns_blocked_result(monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv("E2B_API_KEY")
    script_model(monkeypatch, [{"action": "execute", "code": "print(1 + 1)"}])

    result = await server._solve("Calculate", [], "", 2, 5, 1_000)

    assert result.status == "blocked"
    assert result.error_code == "CODE_AGENT_RUNTIME_UNAVAILABLE"
    assert result.steps_taken == 0


@pytest.mark.asyncio
async def test_step_limit_stops_additional_execution(monkeypatch):
    sandbox = FakeSandbox([execution(stdout=["intermediate"])])
    configure(monkeypatch, sandbox)
    script_model(
        monkeypatch,
        [
            {"action": "execute", "code": "print(1)"},
            {"action": "execute", "code": "print(2)"},
        ],
    )

    result = await server._solve("Keep computing", [], "", 1, 5, 1_000)

    assert result.status == "incomplete"
    assert result.error_code == "CODE_AGENT_STEP_LIMIT"
    assert result.steps_taken == 1
    assert len(sandbox.codes) == 1


@pytest.mark.asyncio
async def test_execution_timeout_is_structured(monkeypatch):
    class SlowSandbox(FakeSandbox):
        async def run_code(self, code, **kwargs):
            self.codes.append((code, kwargs))
            await asyncio.sleep(1)

    sandbox = SlowSandbox([])
    configure(monkeypatch, sandbox)
    script_model(monkeypatch, [{"action": "execute", "code": "while True: pass"}])

    result = await server._solve("wait", [], "", 2, 0.1, 1_000)

    assert result.status == "incomplete"
    assert result.error_code == "CODE_AGENT_TIMEOUT"
    assert result.telemetry["timeouts"] == 1
    assert sandbox.killed


@pytest.mark.asyncio
async def test_execution_output_and_response_are_bounded(monkeypatch):
    captured = []
    sandbox = FakeSandbox([execution(stdout=["Z" * 2_000])])
    configure(monkeypatch, sandbox)
    script_model(
        monkeypatch,
        [
            {"action": "execute", "code": "print('large')"},
            {"action": "finish", "answer": "A" * 10_000},
        ],
        captured=captured,
    )

    result = await server._solve("Summarize", [], "", 2, 5, 256)
    payload = json.dumps(result.model_dump())

    assert len(result.answer) <= server.MAX_ANSWER_CHARS
    assert "Z" * 300 not in json.dumps(captured[1])
    assert "Z" * 300 not in payload
    assert "stdout" not in result.model_dump()


@pytest.mark.asyncio
async def test_api_secrets_are_redacted_from_prompt_errors_and_result(
    monkeypatch, caplog
):
    secret = "code-agent-test-secret"
    sandbox = FakeSandbox(
        [
            execution(
                stdout=[f"found {secret}"],
                error=SimpleNamespace(name="ValueError", value=f"bad {secret}"),
            )
        ]
    )
    configure(monkeypatch, sandbox)
    captured = []
    script_model(
        monkeypatch,
        [
            {"action": "execute", "code": "print(secret)"},
            {
                "action": "finish",
                "answer": f"The value is {secret}",
                "evidence": [f"token {secret}"],
            },
        ],
        captured=captured,
    )

    result = await server._solve(f"Analyze {secret}", [], secret, 1, 5, 1_000)
    all_logs = "\n".join(record.getMessage() for record in caplog.records)

    assert secret not in json.dumps(captured)
    assert secret not in json.dumps(result.model_dump())
    assert secret not in all_logs
    assert "[redacted]" in result.answer
    assert "[redacted]" in result.errors[0]


@pytest.mark.asyncio
async def test_document_retrieval_is_routed_to_document_tool(monkeypatch):
    configure(monkeypatch)
    script_model(monkeypatch, [{"action": "document"}])

    result = await server._solve("Read and summarize this PDF", [], "", 2, 5, 1_000)

    assert result.status == "blocked"
    assert result.error_code == "CODE_AGENT_DOCUMENT_READING_RECOMMENDED"


def test_invalid_cross_provider_override_does_not_borrow_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODE_AGENT_BASE_URL", "https://custom.example/v1")
    monkeypatch.delenv("CODE_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("FEDOTMAS_GAIA_WORKER_MODEL", raising=False)
    monkeypatch.delenv("FEDOTMAS_GAIA_WORKER_API_KEY", raising=False)
    monkeypatch.delenv("FEDOTMAS_GAIA_WORKER_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="CODE_AGENT_API_KEY"):
        server._llm_settings()


@pytest.mark.asyncio
async def test_openrouter_completion_usage_is_accounted_with_mock_http(monkeypatch):
    seen = {}

    async def respond(request):
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 5,
                    "total_tokens": 22,
                    "cost": 0.003,
                },
            },
        )

    transport = httpx.MockTransport(respond)
    usage = server.NestedUsage()
    async with httpx.AsyncClient(transport=transport) as client:
        action = await server._request_model(
            client,
            ("test-model", "local-mock-key", "https://openrouter.ai/api/v1"),
            [{"role": "user", "content": "Calculate"}],
            usage,
        )

    assert action == {"action": "finish"}
    assert seen["authorization"] == "Bearer local-mock-key"
    assert seen["body"]["usage"] == {"include": True}
    assert usage.llm_invocations == 1
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 22
    assert usage.cost_usd == 0.003
    assert usage.available
