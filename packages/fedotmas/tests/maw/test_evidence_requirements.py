import json
from unittest.mock import MagicMock

import pytest
from google.adk.models.llm_request import LlmRequest

from fedotmas.maw import builder
from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from fedotmas.maw._file_evidence import record_file_evidence, missing_source_evidence
from fedotmas.maw._evidence_requirements import (
    preserve_task_evidence,
    evidence_flow_errors,
)
from fedotmas.maw.handoffs import is_unresolved_answer
from fedotmas.meta._helpers import validate_pipeline_output
from fedotmas.control.fixes._fix_instruction import fix_instruction
from fedotmas.control.fixes._fix_agent_configs import fix_agent_configs


URL = "https://example.org/paper"
TASK = f"Use the equation from {URL} and retain its units."


def payload(source=URL, kind="url", task=TASK):
    return {
        "name": "reader",
        "instruction": "Read the required source.",
        "output_key": "data",
        "tools": ["document"],
        "output_contract": {"required_fields": ["formula", "units", "source_url"]},
        "evidence_requirements": [
            {
                "task_quote": task,
                "source": source,
                "source_kind": kind,
                "source_field": "source_url",
                "required_fields": ["formula", "units"],
            }
        ],
    }


def config(**kwargs):
    return MAWConfig.model_validate(
        {"agents": [payload(**kwargs)], "pipeline": {"agent_name": "reader"}}
    )


def context(state):
    ctx = MagicMock()
    ctx.state = state
    return ctx


@pytest.mark.parametrize("tools", [[], ["websearch-tavily"], ["download"]])
def test_source_reader_cannot_be_tool_free_discovery_or_download_only(tools):
    raw = payload()
    raw["tools"] = tools
    with pytest.raises(ValueError, match="no inspection route"):
        MAWAgentConfig.model_validate(raw)


def test_source_reader_preserves_units_and_source_fields_in_contract():
    raw = payload()
    raw["output_contract"]["required_fields"] = ["formula"]
    with pytest.raises(ValueError, match="evidence fields"):
        MAWAgentConfig.model_validate(raw)


def test_generation_checks_quotes_and_explicit_attachment_coverage():
    cfg = config()
    validate_pipeline_output(cfg.model_dump(), available_tools={"document"}, task=TASK)
    with pytest.raises(ValueError, match="verbatim excerpt"):
        validate_pipeline_output(
            cfg.model_dump(), available_tools={"document"}, task="An unrelated task"
        )
    with pytest.raises(ValueError, match="supplied host files"):
        validate_pipeline_output(
            cfg.model_dump(),
            available_tools={"document"},
            task=TASK + "\nHost file path: /tmp/data.zip",
        )
    attached = config(
        source="/tmp/data.zip", kind="file", task="Host file path: /tmp/data.zip"
    )
    validate_pipeline_output(
        attached.model_dump(),
        available_tools={"document"},
        task="Host file path: /tmp/data.zip",
    )


@pytest.mark.parametrize(
    "parallel,fields",
    [
        (False, ["formula", "units", "source_url"]),
        (True, ["formula", "units", "source_url"]),
        (False, ["formula"]),
    ],
)
def test_source_obligations_are_consumed_after_reading_with_units(parallel, fields):
    raw = config().model_dump()
    raw["agents"].append(
        {
            "name": "calculator",
            "instruction": "Calculate from {data}",
            "output_key": "answer",
            "input_requirements": [{"source_key": "data", "required_fields": fields}],
        }
    )
    raw["pipeline"] = {
        "type": "parallel" if parallel else "sequential",
        "children": [{"agent_name": "reader"}, {"agent_name": "calculator"}],
    }
    errors = evidence_flow_errors(MAWConfig.model_validate(raw))
    assert bool(errors) == (parallel or "units" not in fields)


@pytest.mark.asyncio
async def test_instruction_repair_cannot_remove_source_obligation_or_bypass_no_read():
    cfg = config()
    state = {"config": cfg.model_dump_json()}
    ctx = context(state)
    await fix_instruction(
        ctx, "reader", "Assume a formula and compatible units.", "Shortcut"
    )
    changed = MAWConfig.model_validate_json(state["config"])
    assert (
        changed.agents[0].evidence_requirements == cfg.agents[0].evidence_requirements
    )
    agent = builder._build_llm_agent(changed.agents[0], None, None)
    ctx.state = {"user_query": TASK}
    await agent.before_agent_callback(ctx)
    request = LlmRequest()
    await agent.before_model_callback(ctx, request)
    assert "REQUIRED SOURCE EVIDENCE" in str(request.config.system_instruction)
    ctx.state["data"] = json.dumps(
        {"formula": "invented", "units": "assumed", "source_url": URL}
    )
    await agent.after_agent_callback(ctx)
    assert is_unresolved_answer(ctx.state["data"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [
        {"tools": []},
        {"research_mode": "discovery_only"},
        {"output_contract": {"required_fields": ["formula"]}},
    ],
)
async def test_atomic_recovery_cannot_weaken_declared_evidence(update):
    before = config().model_dump_json()
    state = {"config": before, "_fedotmas_recovery_available_tools": ["document"]}
    result = await fix_agent_configs(
        context(state), json.dumps([{"agent_name": "reader", **update}]), "Shortcut"
    )
    assert "Error" in result
    assert state["config"] == before


def test_full_config_replacement_must_preserve_obligations():
    old = config()
    raw = old.model_dump()
    raw["agents"][0]["evidence_requirements"] = []
    with pytest.raises(ValueError, match="preserve source obligations"):
        preserve_task_evidence(old, MAWConfig.model_validate(raw))


@pytest.mark.parametrize("read_url", [URL, "https://example.org/unrelated"])
def test_citation_must_match_an_executed_read_of_required_url(read_url):
    cfg = config().agents[0]
    state = {}
    record_file_evidence(
        state,
        "reader",
        "read_document",
        {"file_path": read_url},
        {"content": "Equation and units"},
    )
    value = {"formula": "source equation", "units": "source units", "source_url": URL}
    assert bool(missing_source_evidence(state, cfg, value)) == (read_url != URL)


def test_current_page_read_has_source_but_navigation_alone_is_not_evidence():
    cfg = config().agents[0]
    state = {}
    value = {"formula": "source equation", "units": "source units", "source_url": URL}
    record_file_evidence(
        state, "reader", "goto", {"url": URL}, {"content": "Navigated"}
    )
    assert missing_source_evidence(state, cfg, value)
    record_file_evidence(
        state, "reader", "markdown", {}, {"content": "Equation and units"}
    )
    assert not missing_source_evidence(state, cfg, value)


@pytest.mark.parametrize(
    "response",
    [
        {"status": "failed", "content": "denied"},
        {"structuredContent": {"error": "denied", "content": "error page"}},
        {"content": ""},
    ],
)
def test_failed_or_empty_reads_do_not_fulfill_source_obligation(response):
    state = {}
    cfg = config().agents[0]
    record_file_evidence(state, "reader", "read_document", {"file_path": URL}, response)
    assert missing_source_evidence(
        state, cfg, {"formula": "invented", "units": "assumed", "source_url": URL}
    )


def test_failed_navigation_does_not_attach_wrong_source_to_old_dom():
    state = {}
    record_file_evidence(
        state,
        "reader",
        "goto",
        {"url": "https://example.org/old"},
        {"content": "Navigated"},
    )
    record_file_evidence(state, "reader", "goto", {"url": URL}, {"isError": True})
    record_file_evidence(state, "reader", "markdown", {}, {"content": "Old DOM"})
    assert missing_source_evidence(
        state,
        config().agents[0],
        {"formula": "invented", "units": "assumed", "source_url": URL},
    )


def test_download_unpack_read_preserves_origin_across_agents():
    state = {}
    cfg = config().agents[0]
    record_file_evidence(
        state,
        "downloader",
        "download",
        {"urls": [URL]},
        {"results": [{"success": True, "file_path": "/tmp/data.zip"}]},
    )
    record_file_evidence(
        state,
        "unpacker",
        "extract_zip",
        {"file_path": "/tmp/data.zip"},
        {"output_dir": "/tmp/data", "files": ["equation.txt"]},
    )
    value = {
        "formula": "equation",
        "units": "units",
        "source_url": "/tmp/data/equation.txt",
    }
    assert missing_source_evidence(state, cfg, value)
    record_file_evidence(
        state,
        "reader",
        "read_document",
        {"file_path": "/tmp/data/equation.txt"},
        {"content": "Equation and units"},
    )
    assert not missing_source_evidence(state, cfg, value)


def test_named_reference_requires_actual_cited_source_and_missing_units_stay_missing():
    cfg = config(
        source="named translation", kind="reference", task="Use the named translation"
    ).agents[0]
    state = {}
    value = {"formula": "equation", "units": "units", "source_url": URL}
    assert missing_source_evidence(state, cfg, value)
    record_file_evidence(
        state,
        "reader",
        "read_document",
        {"file_path": URL},
        {"content": "Equation and units"},
    )
    assert not missing_source_evidence(state, cfg, value)
    value["units"] = None
    assert missing_source_evidence(state, cfg, value)


@pytest.mark.parametrize(
    "name,args,response",
    [
        (
            "complete_browser_task",
            {"task": "Read the paper"},
            {
                "status": "completed",
                "findings": "Equation and units",
                "steps_taken": 3,
                "relevant_urls": [URL],
            },
        ),
        (
            "analyze_image",
            {"file_path": URL},
            {"result": "Equation and units", "error": None},
        ),
    ],
)
def test_browser_and_media_results_link_actual_sources(name, args, response):
    state = {}
    cfg = config().agents[0]
    record_file_evidence(state, "reader", name, args, response)
    assert not missing_source_evidence(
        state, cfg, {"formula": "equation", "units": "units", "source_url": URL}
    )
