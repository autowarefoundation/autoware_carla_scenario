"""A sweep run from the viewer runs what it resolved and what it previews (#681).

``/api/run`` with a sweeper expands each scenario with the Run panel's extra
overrides and runs every job with them too; the expansion is the CLI's
(``sweeper.expand.expand_config``), and an expansion that fails is reported
rather than run as "no lanelets matched".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner, sweep_resolver

EXTRA = ["ego.initial_speed_kmh=42.0"]


@pytest.fixture
def started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What ``runner.start_run`` was asked to run (nothing is run)."""
    seen: dict[str, Any] = {}

    def fake_start_run(
        overrides_list: list[list[str]],
        *,
        base_path: Path,
        extra_overrides: list[str],
        group_as_multirun: bool,
    ) -> bool:
        seen["jobs"] = overrides_list
        seen["extra"] = extra_overrides
        return True

    monkeypatch.setattr(runner, "start_run", fake_start_run)
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    return seen


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


def _command(job: list[str], extra: list[str]) -> list[str]:
    """The command the runner makes of a job (``runner._run_worker``)."""
    return ["uv", "run", "scenario", *job, *extra]


def test_sweep_jobs_run_with_the_extra_overrides(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_with: list[list[str]] = []
    real_resolve = sweep_resolver.resolve_sweep

    def spy(name: str, extra: list[str] | None = None, **kwargs: Any) -> Any:
        resolved_with.append(list(extra or []))
        return real_resolve(name, extra, **kwargs)

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", spy)
    body = {"scenario": "cut_in/left", "extra_overrides": EXTRA, "sweeper": "lanelet"}

    preview = client.post("/api/run/preview", json=body).json()["command"]
    assert EXTRA[0] in preview

    response = client.post("/api/run", json=body)
    assert response.json() == {"status": "started"}, response.json()
    assert resolved_with == [EXTRA]
    assert started["jobs"], "the sweep produced no jobs"
    assert started["extra"] == EXTRA
    for job in started["jobs"]:
        assert job[0] == "scenario=cut_in/left"
        assert any(o.startswith("ego.spawn_lanelet_id=") for o in job)
        # Run as the preview shows it: the scenario, then (after the case's
        # own overrides) the extra ones.
        command = _command(job, started["extra"])
        assert command[:4] == preview.split("  #")[0].split()[:4]
        assert command[-len(EXTRA) :] == EXTRA


def test_an_odd_sampled_sweep_draws_its_cases(
    started: dict[str, Any], client: TestClient
) -> None:
    # sweep.odd_sample, which the viewer used to ignore: expanded as the CLI
    # expands it (a scenario without constraints needs no map).
    extra = ["+sweep.odd_sample.count=3"]
    response = client.post(
        "/api/run",
        json={
            "scenario": "intersection_passing/straight_3npc",
            "extra_overrides": extra,
            "sweeper": "lanelet",
        },
    )
    assert response.json() == {"status": "started"}, response.json()
    assert len(started["jobs"]) == 3
    for job in started["jobs"]:
        assert job[0] == "scenario=intersection_passing/straight_3npc"
        assert any(o.startswith("++odd_case=") for o in job)
    assert started["extra"] == extra


def test_resolve_sweep_is_the_cli_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoware_carla_scenario.sweeper import expand

    calls: list[Any] = []

    def fake_expand_config(cfg: Any, arguments: Any = (), *, drawn: Any = None) -> Any:
        calls.append(cfg.scenario.name)
        return [["scenario.route.index=0", "ego.spawn_lanelet_id=7"]]

    monkeypatch.setattr(expand, "expand_config", fake_expand_config)
    jobs = sweep_resolver.resolve_sweep("cut_in/left", EXTRA)
    assert calls == ["cut_in"]
    assert jobs == [
        ["scenario=cut_in/left", "scenario.route.index=0", "ego.spawn_lanelet_id=7"]
    ]


def test_a_sweep_that_cannot_be_resolved_is_reported(
    started: dict[str, Any], client: TestClient
) -> None:
    response = client.post(
        "/api/run",
        json={
            "scenario": "cut_in/left",
            "extra_overrides": ["no_such_group=nothing"],
            "sweeper": "lanelet",
        },
    )
    data = response.json()
    assert data["status"] == "error"
    assert data["message"].startswith("Sweep resolution failed")
    assert "cut_in/left" in data["message"]
    assert "no lanelets matched" not in data["message"]
    assert "jobs" not in started  # nothing was run


def test_one_failing_scenario_stops_the_run(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(name: str, extra: Any = None, **_: Any) -> list[list[str]]:
        if name == "b/bad":
            raise sweep_resolver.SweepResolutionError("b/bad: the map is missing")
        return [[f"scenario={name}"]]

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", resolve)
    monkeypatch.setattr(
        app_module.scanner, "list_scenario_configs", lambda: ["a/good", "b/bad"]
    )
    response = client.post(
        "/api/run", json={"scenario": "*/*", "extra_overrides": [], "sweeper": "x"}
    )
    assert response.json() == {
        "status": "error",
        "message": "Sweep resolution failed: b/bad: the map is missing",
    }
    assert "jobs" not in started
