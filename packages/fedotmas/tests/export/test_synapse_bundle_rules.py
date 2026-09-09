"""Synapse bundle rules — checked against their own save-time validation.

``_assert_accepted`` ports the validation Synapse runs when a workflow is saved
(``src/api/routes/workflow_definitions.py::validate_dag`` at 27424db), so a
bundle can be checked without their tenant. Tool and model identifiers still
need one.

``urban_bundle_structure.json`` is the structure of a bundle they built by hand,
kept here so the port is checked against their shapes and not only against ours.
It gates loops behind an `approval_gate` and holds no `validator` node, so the
one construct this exporter invents is the one their own work cannot vouch for.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path

import pytest

from fedotmas.export import to_synapse_bundle, to_wire_name
from fedotmas.maw.models import AgentPoolConfig, MAWAgentConfig, MAWConfig, MAWStepConfig

_WIRE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _assert_accepted(bundle: dict) -> None:
    workflow = bundle["items"]["workflows"][0]
    nodes, edges = workflow["nodes"], workflow["edges"]
    ids = [n["id"] for n in nodes]
    types = {n["id"]: n.get("type") for n in nodes}

    assert len(ids) == len(set(ids)), "duplicate node ids"
    assert [t for t in types.values()].count("start") == 1
    assert [t for t in types.values()].count("end") == 1

    for edge in edges:
        assert edge["from"] in types, f"dangling edge source {edge['from']}"
        assert edge["to"] in types, f"dangling edge target {edge['to']}"

    declared = {a["_id"] for a in bundle["items"]["agents"]}
    for agent in bundle["items"]["agents"]:
        assert _WIRE_NAME_RE.match(agent["_id"]), agent["_id"]

    for node in nodes:
        if types[node["id"]] == "phase":
            assert node.get("agent_selection") == "direct"
            assert node.get("agent_type"), f"phase {node['id']} names no agent"
            assert node["agent_type"] in declared
        if types[node["id"]] == "validator":
            assert node.get("checks"), "validator without checks"
            for check in node["checks"]:
                assert str(check.get("key") or "").strip()
                assert check.get("kind") in ("structural", "json_schema")
            assert any(
                e["from"] == node["id"] and e.get("condition") == "rejected"
                for e in edges
            ), "validator without a rejected edge terminates the workflow"

    # Cycle detection skips rejected back-edges, exactly as their engine does.
    in_degree = dict.fromkeys(ids, 0)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if types[edge["from"]] in ("approval_gate", "validator"):
            if edge.get("condition") == "rejected":
                continue
        adjacency[edge["from"]].append(edge["to"])
        in_degree[edge["to"]] += 1

    queue = deque(n for n, deg in in_degree.items() if deg == 0)
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for nxt in adjacency[node]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    assert seen == len(ids), "workflow has a cycle their validator would reject"


def _agent(name: str, key: str, **kw) -> MAWAgentConfig:
    return MAWAgentConfig(
        name=name, instruction=kw.pop("instruction", f"Do {name}"), output_key=key, **kw
    )


@pytest.fixture()
def linear() -> MAWConfig:
    return MAWConfig(
        agents=[_agent("researcher", "research"), _agent("writer", "summary")],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(type="agent", agent_name="researcher"),
                MAWStepConfig(type="agent", agent_name="writer"),
            ],
        ),
    )


class TestSequentialPipeline:
    """Rule 1: a chain converts one node per agent, plus start and end."""

    def test_shape_is_accepted(self, linear):
        export = to_synapse_bundle(linear, workflow_id="generated_flow")
        _assert_accepted(export.bundle)

        workflow = export.bundle["items"]["workflows"][0]
        assert [n["id"] for n in workflow["nodes"]] == [
            "start",
            "researcher",
            "writer",
            "end",
        ]
        assert workflow["edges"] == [
            {"from": "start", "to": "researcher"},
            {"from": "researcher", "to": "writer"},
            {"from": "writer", "to": "end"},
        ]
        assert export.linearized_branches == 0

    def test_agent_fields_map_over(self, linear):
        export = to_synapse_bundle(linear, workflow_id="generated_flow")
        doc = export.bundle["items"]["agents"][0]

        assert doc["agent_class"] == "GenericAgent"
        assert doc["system_prompt"] == "Do researcher"
        assert doc["output_save_key"] == "research"
        assert doc["enabled"] is True
        assert export.bundle["version"] == 1
        # Identity is the wire name; `type` is only an auction kind, and their own
        # bundle has three agents sharing one.
        assert doc["name"] == doc["_id"] == "researcher"


class TestParallelIsLinearized:
    """Rule 2: their engine has one successor per node, so branches chain up."""

    def test_branches_counted_and_chained(self):
        config = MAWConfig(
            agents=[
                _agent("a", "ka"),
                _agent("b", "kb"),
                _agent("c", "kc"),
            ],
            pipeline=MAWStepConfig(
                type="parallel",
                children=[
                    MAWStepConfig(type="agent", agent_name="a"),
                    MAWStepConfig(type="agent", agent_name="b"),
                    MAWStepConfig(type="agent", agent_name="c"),
                ],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        _assert_accepted(export.bundle)
        assert export.linearized_branches == 2


class TestLoopBecomesValidator:
    """Rule 3: a loop is a validator with a rejected back-edge."""

    def test_back_edge_and_cap(self):
        config = MAWConfig(
            agents=[_agent("drafter", "draft"), _agent("critic", "verdict")],
            pipeline=MAWStepConfig(
                type="loop",
                max_iterations=4,
                children=[
                    MAWStepConfig(type="agent", agent_name="drafter"),
                    MAWStepConfig(type="agent", agent_name="critic"),
                ],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        _assert_accepted(export.bundle)

        workflow = export.bundle["items"]["workflows"][0]
        validator = next(n for n in workflow["nodes"] if n["type"] == "validator")
        assert validator["checks"][0]["key"] == "verdict"
        # Ours counts passes, theirs counts rejections after the first.
        assert validator["max_reject_retries"] == 3
        assert {"from": validator["id"], "to": "drafter", "condition": "rejected"} in (
            workflow["edges"]
        )
        assert {
            "from": validator["id"],
            "to": "end",
            "condition": "approved",
        } in workflow["edges"]


class TestExternalAgentsKeepTheirIdentity:
    """Rule 4: a reused agent is referenced by the caller's own id."""

    def test_id_survives_export(self, linear):
        pool = AgentPoolConfig(
            agents=[
                {
                    "name": "researcher",
                    "instruction": "Theirs",
                    "id": "urban_requirements_gatherer",
                }
            ]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        _assert_accepted(export.bundle)

        ids = [a["_id"] for a in export.bundle["items"]["agents"]]
        assert ids == ["urban_requirements_gatherer", "writer"]
        assert export.reused_agents == ("urban_requirements_gatherer",)
        node = export.bundle["items"]["workflows"][0]["nodes"][1]
        assert node["agent_type"] == "urban_requirements_gatherer"


class TestToolsOutsideTheCatalogue:
    """Rule 5: an id their tenant does not have is dropped, not shipped broken."""

    def test_dropped_and_reported(self):
        config = MAWConfig(
            agents=[_agent("fetcher", "data", tools=["urban.getproject", "download"])],
            pipeline=MAWStepConfig(type="agent", agent_name="fetcher"),
        )
        export = to_synapse_bundle(
            config,
            workflow_id="generated_flow",
            tool_catalog={"urban.getproject": "Get a project"},
        )
        assert export.bundle["items"]["agents"][0]["allowed_tools"] == [
            "urban.getproject"
        ]
        assert export.unresolved_tools == ("download",)

    def test_no_catalogue_keeps_every_tool(self):
        config = MAWConfig(
            agents=[_agent("fetcher", "data", tools=["download"])],
            pipeline=MAWStepConfig(type="agent", agent_name="fetcher"),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        assert export.bundle["items"]["agents"][0]["allowed_tools"] == ["download"]
        assert export.unresolved_tools == ()


class TestWireNames:
    """Rule 6: identifiers match the pattern their import enforces."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Urban Planner", "urban_planner"),
            ("report-writer!", "report_writer"),
            ("3rd_agent", "a_3rd_agent"),
            ("x", "x_1"),
        ],
    )
    def test_slugged(self, raw, expected):
        assert to_wire_name(raw) == expected
        assert _WIRE_NAME_RE.match(to_wire_name(raw))

    def test_collisions_get_a_suffix(self):
        taken: set[str] = set()
        assert to_wire_name("Writer", taken) == "writer"
        assert to_wire_name("writer", taken) == "writer_2"


class TestSentinelNamesDoNotCollide:
    """Rule 7: an agent named like a sentinel gets its own node id."""

    def test_agent_named_start(self):
        config = MAWConfig(
            agents=[_agent("start", "k"), _agent("writer", "summary")],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="start"),
                    MAWStepConfig(type="agent", agent_name="writer"),
                ],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        _assert_accepted(export.bundle)

        ids = [n["id"] for n in export.bundle["items"]["workflows"][0]["nodes"]]
        assert ids == ["start", "start_2", "writer", "end"]


class TestNestedLoops:
    """Rule 8: a validator feeding a validator still needs its approved edge."""

    def test_inner_forward_edge_is_conditional(self):
        config = MAWConfig(
            agents=[_agent("drafter", "draft")],
            pipeline=MAWStepConfig(
                type="loop",
                max_iterations=2,
                children=[
                    MAWStepConfig(
                        type="loop",
                        max_iterations=2,
                        children=[MAWStepConfig(type="agent", agent_name="drafter")],
                    )
                ],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        _assert_accepted(export.bundle)

        edges = export.bundle["items"]["workflows"][0]["edges"]
        validators = [
            n["id"]
            for n in export.bundle["items"]["workflows"][0]["nodes"]
            if n["type"] == "validator"
        ]
        assert len(validators) == 2
        for node_id in validators:
            forward = [
                e for e in edges if e["from"] == node_id and e.get("condition") != "rejected"
            ]
            assert forward and all(e["condition"] == "approved" for e in forward)


class TestIdentifiersTheirFormatRejects:
    """Rule 9: an id that cannot be kept is reported, never silently swapped."""

    def test_unusable_caller_id_reported(self, linear):
        pool = AgentPoolConfig(
            agents=[
                {
                    "name": "researcher",
                    "instruction": "Theirs",
                    "id": "Urban-Requirements-Gatherer",
                }
            ]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        _assert_accepted(export.bundle)

        assert export.reused_agents == ()
        assert export.renamed_ids == (
            ("Urban-Requirements-Gatherer", "urban_requirements_gatherer"),
        )

    def test_entry_without_an_id_is_not_reused(self, linear):
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "Theirs"}]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        assert export.reused_agents == ()
        assert export.renamed_ids == ()

    def test_workflow_id_is_a_wire_name(self, linear):
        export = to_synapse_bundle(linear, workflow_id="Generated Flow")
        workflow = export.bundle["items"]["workflows"][0]
        assert workflow["_id"] == "generated_flow"
        assert _WIRE_NAME_RE.match(workflow["_id"])


class TestLoopFidelityIsReported:
    """Rule 10: their structural gate cannot reject, so a loop runs once."""

    def test_degraded_loops_counted(self):
        config = MAWConfig(
            agents=[_agent("drafter", "draft")],
            pipeline=MAWStepConfig(
                type="loop",
                max_iterations=3,
                children=[MAWStepConfig(type="agent", agent_name="drafter")],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        assert export.degraded_loops == 1

    def test_loop_without_a_limit_still_carries_one(self):
        config = MAWConfig(
            agents=[_agent("drafter", "draft")],
            pipeline=MAWStepConfig(
                type="loop",
                children=[MAWStepConfig(type="agent", agent_name="drafter")],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        validator = next(
            n
            for n in export.bundle["items"]["workflows"][0]["nodes"]
            if n["type"] == "validator"
        )
        assert 0 <= validator["max_reject_retries"] <= 10

    def test_iterations_beyond_their_cap_are_clamped(self):
        config = MAWConfig(
            agents=[_agent("drafter", "draft")],
            pipeline=MAWStepConfig(
                type="loop",
                max_iterations=50,
                children=[MAWStepConfig(type="agent", agent_name="drafter")],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        validator = next(
            n
            for n in export.bundle["items"]["workflows"][0]["nodes"]
            if n["type"] == "validator"
        )
        assert validator["max_reject_retries"] == 10


class TestNonLatinNames:
    """Rule 11: a Russian agent name keeps its identity through the slug."""

    def test_transliterated(self):
        assert to_wire_name("Исследователь") == "issledovatel"
        assert to_wire_name("Мастер-планировщик") == "master_planirovshchik"

    def test_names_stay_distinct_in_the_bundle(self):
        config = MAWConfig(
            agents=[_agent("Исследователь", "a"), _agent("Писатель", "b")],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="Исследователь"),
                    MAWStepConfig(type="agent", agent_name="Писатель"),
                ],
            ),
        )
        export = to_synapse_bundle(config, workflow_id="generated_flow")
        _assert_accepted(export.bundle)

        agents = export.bundle["items"]["agents"]
        assert [a["_id"] for a in agents] == ["issledovatel", "pisatel"]
        assert agents[0]["display_name"] == "Исследователь"


class TestSuppliedAgentsMissingFromTheConfig:
    """Rule 12: an id that never reached the config is reported, not lost."""

    def test_unmatched_entry_reported(self, linear):
        pool = AgentPoolConfig(
            agents=[
                {
                    "name": "requirements_gatherer",
                    "instruction": "Theirs",
                    "id": "urban_requirements_gatherer",
                }
            ]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        _assert_accepted(export.bundle)

        assert export.unmatched_agents == ("requirements_gatherer",)
        assert export.reused_agents == ()
        assert "urban_requirements_gatherer" not in [
            a["_id"] for a in export.bundle["items"]["agents"]
        ]

    def test_nothing_reported_when_every_entry_is_used(self, linear):
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "Theirs", "id": "researcher"}]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        assert export.unmatched_agents == ()
        assert export.reused_agents == ("researcher",)


class TestReusedAgentsAreNotRewritten:
    """Rule 13: their import is an upsert, so a reused agent is emitted thin."""

    def test_only_owned_fields_are_emitted(self, linear):
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "Theirs", "id": "researcher"}]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        _assert_accepted(export.bundle)

        reused, generated = export.bundle["items"]["agents"]
        assert set(reused) == {
            "_id",
            "name",
            "type",
            "model",
            "allowed_tools",
            "output_save_key",
            "system_prompt",
        }
        # Everything left out stays as the platform has it; what cannot be left
        # out is named in the report.
        for field in ("temperature", "allowed_phases", "allowed_delegation_targets"):
            assert field not in reused
            assert field in generated
        assert set(export.overwritten_fields) >= {"system_prompt", "allowed_tools"}

    def test_tools_are_emitted_even_when_empty(self, linear):
        # Their import derives both tool lists from what arrives, so an absent
        # `allowed_tools` clears the ones the agent already has.
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "Theirs", "id": "researcher"}]
        )
        export = to_synapse_bundle(
            linear, workflow_id="generated_flow", existing_agents=pool
        )
        assert export.bundle["items"]["agents"][0]["allowed_tools"] == []

    def test_their_tools_survive_a_narrower_catalogue(self):
        config = MAWConfig(
            agents=[
                _agent("researcher", "research", tools=["urbanindicators.getprovision"]),
                _agent("writer", "summary", tools=["urbanindicators.getprovision"]),
            ],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="researcher"),
                    MAWStepConfig(type="agent", agent_name="writer"),
                ],
            ),
        )
        pool = AgentPoolConfig(
            agents=[{"name": "researcher", "instruction": "Theirs", "id": "researcher"}]
        )
        export = to_synapse_bundle(
            config,
            workflow_id="generated_flow",
            existing_agents=pool,
            tool_catalog={"urbanprojects.getprojectbyid": "Get a project"},
        )
        _assert_accepted(export.bundle)

        reused, generated = export.bundle["items"]["agents"]
        assert reused["allowed_tools"] == ["urbanindicators.getprovision"]
        # A generated agent has no tools of its own to lose, so the filter stands.
        assert generated["allowed_tools"] == []
        assert export.unresolved_tools == ("urbanindicators.getprovision",)

    def test_nothing_reported_without_reuse(self, linear):
        export = to_synapse_bundle(linear, workflow_id="generated_flow")
        assert export.overwritten_fields == ()
        assert "temperature" in export.bundle["items"]["agents"][0]


class TestTheirOwnBundleIsAccepted:
    """Rule 14: the ported validator accepts a bundle Synapse itself wrote."""

    def test_hand_built_bundle_passes(self):
        fixture = Path(__file__).parent / "urban_bundle_structure.json"
        _assert_accepted(json.loads(fixture.read_text()))
