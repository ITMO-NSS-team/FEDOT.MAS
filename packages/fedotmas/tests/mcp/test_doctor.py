from __future__ import annotations

import asyncio

import pytest

from fedotmas.mcp import doctor
from fedotmas.mcp.doctor import DEGRADED, FAIL, OK, Prerequisite, ServerReport


def _prerequisite(problem: str | None, *, fatal: bool = False) -> Prerequisite:
    return Prerequisite(check=lambda: problem, fix="just fix-it", fatal=fatal)


@pytest.fixture
def no_prerequisites(monkeypatch):
    monkeypatch.setattr(doctor, "PREREQUISITES", {})


class TestFatalPrerequisite:
    async def test_missing_binary_fails_without_probing(self, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "PREREQUISITES",
            {"srv": (_prerequisite("lightpanda is not on PATH", fatal=True),)},
        )

        async def unreachable(name):
            raise AssertionError("probe should be skipped")

        monkeypatch.setattr(doctor, "_probe", unreachable)

        report = await doctor.check_server("srv")

        assert report.status == FAIL
        assert report.detail == "lightpanda is not on PATH"
        assert report.fix == "just fix-it"

    async def test_satisfied_prerequisite_lets_the_probe_run(self, monkeypatch):
        monkeypatch.setattr(
            doctor, "PREREQUISITES", {"srv": (_prerequisite(None, fatal=True),)}
        )
        monkeypatch.setattr(doctor, "_probe", _probe_ok)

        report = await doctor.check_server("srv")

        assert report.status == OK


class TestNonFatalPrerequisite:
    async def test_the_message_comes_from_the_first_check(self, monkeypatch):
        """A second check would re-probe the service and could disagree."""
        answers = iter(["KEY is not set", None])
        monkeypatch.setattr(
            doctor,
            "PREREQUISITES",
            {"srv": (Prerequisite(check=lambda: next(answers), fix="just fix-it"),)},
        )
        monkeypatch.setattr(doctor, "_probe", _probe_ok)

        report = await doctor.check_server("srv")

        assert report.detail == "3 tools, but KEY is not set"

    async def test_listing_tools_is_not_enough(self, monkeypatch):
        monkeypatch.setattr(
            doctor, "PREREQUISITES", {"srv": (_prerequisite("KEY is not set"),)}
        )
        monkeypatch.setattr(doctor, "_probe", _probe_ok)

        report = await doctor.check_server("srv")

        assert report.status == DEGRADED
        assert report.detail == "3 tools, but KEY is not set"
        assert report.fix == "just fix-it"

    async def test_a_failing_probe_stays_failed(self, monkeypatch):
        monkeypatch.setattr(
            doctor, "PREREQUISITES", {"srv": (_prerequisite("KEY is not set"),)}
        )

        async def probe(name):
            return FAIL, "ConnectionError: nope"

        monkeypatch.setattr(doctor, "_probe", probe)

        report = await doctor.check_server("srv")

        assert report.status == FAIL
        assert report.detail == "ConnectionError: nope"


class TestProbe:
    async def test_a_broken_server_is_reported_not_raised(
        self, monkeypatch, no_prerequisites
    ):
        def explode(name):
            raise ConnectionError("Client failed to connect")

        monkeypatch.setattr("fedotmas.mcp.registry.create_toolset", explode)

        report = await doctor.check_server("srv")

        assert report.status == FAIL
        assert report.detail == "ConnectionError: Client failed to connect"

    async def test_a_hanging_server_is_cut_short(self, monkeypatch, no_prerequisites):
        class Hanging:
            async def get_tools(self):
                await asyncio.sleep(3600)

            async def close(self):
                pass

        monkeypatch.setattr(doctor, "_PROBE_TIMEOUT_S", 0.05)
        monkeypatch.setattr(
            "fedotmas.mcp.registry.create_toolset", lambda name: Hanging()
        )

        report = await doctor.check_server("srv")

        assert report.status == FAIL
        assert "timed out" in report.detail

    async def test_a_long_error_is_truncated(self, monkeypatch, no_prerequisites):
        def explode(name):
            raise ConnectionError("x" * 500)

        monkeypatch.setattr("fedotmas.mcp.registry.create_toolset", explode)

        report = await doctor.check_server("srv")

        assert len(report.detail) < 200


class TestRender:
    def test_problems_are_listed_with_their_fixes(self):
        rendered = doctor._render(
            [
                ServerReport("alpha", OK, 1.0, "2 tools"),
                ServerReport("beta", FAIL, 0.0, "missing", "just install-beta"),
            ]
        )

        assert "1 ok, 0 degraded, 1 failed" in rendered
        assert "just install-beta" in rendered
        # A healthy server has nothing to fix and must not appear in the list.
        assert rendered.count("alpha") == 1

    def test_a_clean_report_has_no_fix_section(self):
        rendered = doctor._render([ServerReport("alpha", OK, 1.0, "2 tools")])

        assert "to fix:" not in rendered

    def test_a_problem_with_no_known_fix_leaves_no_empty_header(self):
        rendered = doctor._render([ServerReport("alpha", FAIL, 1.0, "boom")])

        assert "to fix:" not in rendered


async def _probe_ok(name: str) -> tuple[str, str]:
    return OK, "3 tools"
