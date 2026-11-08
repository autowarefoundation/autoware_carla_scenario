"""A sweep run from the viewer runs what it resolved and what it previews (#681).

``/api/run`` with a sweeper expands each scenario with the Run panel's extra
overrides and runs every job with them too; the expansion is the CLI's
(``sweeper.expand.expand_config``), and an expansion that fails is reported
rather than run as "no lanelets matched".
"""

from __future__ import annotations

import asyncio
import json
import shlex
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner, sweep_resolver

EXTRA = ["ego.initial_speed_kmh=42.0"]

#: How long a test waits on another thread before it gives up.
_HANG_S = 120.0


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
    return TestClient(app_module.app, base_url="http://127.0.0.1")


def _command(job: list[str], extra: list[str]) -> list[str]:
    """The command the runner makes of a job (``runner._run_worker``)."""
    return ["uv", "run", "scenario", *job, *extra]


@pytest.mark.usefixtures("at_repo_root")
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

    def fake_expand_config(
        cfg: Any, arguments: Any = (), *, drawn: Any = None, composed: Any = ()
    ) -> Any:
        calls.append(cfg.scenario.name)
        # The Run panel's overrides, which every job runs with after its
        # case's: a case may not set what they set (#1995).
        assert list(composed) == list(EXTRA)
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
        app_module, "_resolve_scenario_names", lambda pattern: ["a/good", "b/bad"]
    )
    response = client.post(
        "/api/run", json={"scenario": "*/*", "extra_overrides": [], "sweeper": "x"}
    )
    assert response.json() == {
        "status": "error",
        "message": "Sweep resolution failed: b/bad: the map is missing",
    }
    assert "jobs" not in started


# A scenario of a glob whose sweep expands to no case is refused, as the
# lanelet sweeper and scenario-batch refuse it (#1407): the run used to start
# with the other scenarios' jobs alone.
def test_a_scenario_whose_sweep_expands_to_no_case_stops_the_run(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(name: str, extra: Any = None, **_: Any) -> list[list[str]]:
        return [] if name == "b/empty" else [[f"scenario={name}"]]

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", resolve)
    monkeypatch.setattr(
        app_module, "_resolve_scenario_names", lambda pattern: ["a/good", "b/empty"]
    )
    response = client.post(
        "/api/run", json={"scenario": "*/*", "extra_overrides": [], "sweeper": "x"}
    )
    assert response.json() == {
        "status": "error",
        "message": "Sweep resolution failed: b/empty: the sweep expands to no "
        "case (its constraints, route or ODD sample match nothing on this map)",
    }
    assert "jobs" not in started


@pytest.mark.usefixtures("at_repo_root")
def test_an_odd_sample_count_of_no_case_is_refused_on_the_real_config(
    started: dict[str, Any], client: TestClient
) -> None:
    response = client.post(
        "/api/run",
        json={
            "scenario": "intersection_passing/straight_3npc",
            "extra_overrides": ["+sweep.odd_sample.count=0"],
            "sweeper": "x",
        },
    )
    data = response.json()
    assert data["status"] == "error"
    # Refused for its count, named, before anything is drawn (#1437).
    assert "intersection_passing/straight_3npc: " in data["message"]
    assert (
        "sweep.odd_sample.count must be a whole number of at least 1, got 0"
        in data["message"]
    )
    assert "jobs" not in started


def test_a_scenario_without_a_sweep_runs_once(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As in scenario-batch: a concrete scenario is its own one case, rather
    # than dropped from a sweep run of a glob.
    from autoware_carla_scenario.sweeper import expand

    monkeypatch.setattr(expand, "expand_config", lambda cfg, *a, **k: [[]])
    assert sweep_resolver.resolve_sweep("cut_in/left") == [["scenario=cut_in/left"]]
    response = client.post(
        "/api/run",
        json={"scenario": "cut_in/left", "extra_overrides": EXTRA, "sweeper": "x"},
    )
    assert response.json() == {"status": "started"}
    assert started["jobs"] == [["scenario=cut_in/left"]]
    assert started["extra"] == EXTRA


def test_the_scenarios_of_a_run_share_their_odd_draws(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.odd.sampler import OddDraws

    seen: list[Any] = []

    def resolve(name: str, extra: Any = None, *, drawn: Any = None) -> Any:
        seen.append(drawn)
        return [[f"scenario={name}"]]

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", resolve)
    monkeypatch.setattr(
        app_module, "_resolve_scenario_names", lambda pattern: ["a/one", "b/two"]
    )
    response = client.post(
        "/api/run", json={"scenario": "*/*", "extra_overrides": [], "sweeper": "x"}
    )
    assert response.json() == {"status": "started"}
    assert started["jobs"] == [["scenario=a/one"], ["scenario=b/two"]]
    assert len(seen) == 2
    assert isinstance(seen[0], OddDraws)
    assert seen[0] is seen[1]


def test_a_busy_run_is_refused_before_its_sweep_is_resolved(
    started: dict[str, Any], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution can load or clone a map: a refused run never pays for it (#1345)."""
    resolved: list[str] = []

    def resolve(name: str, extra: Any = None, *, drawn: Any = None) -> Any:
        resolved.append(name)
        return [[f"scenario={name}"]]

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", resolve)
    monkeypatch.setattr(runner, "is_running", lambda: True)
    response = client.post(
        "/api/run", json={"scenario": "a/one", "extra_overrides": [], "sweeper": "x"}
    )
    assert response.status_code == 409
    assert response.json() == app_module._BUSY
    assert resolved == []
    assert "jobs" not in started


def test_a_slow_sweep_resolution_leaves_other_requests_served(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """While one run's sweep resolves, another request is answered (#1345).

    The client is entered as a context manager so every request shares one
    event loop, as they do under uvicorn: a resolution run inline would hold
    that loop, and the second request would wait for the release that only
    comes after it.
    """
    began = threading.Event()
    release = threading.Event()
    seen: dict[str, Any] = {}

    def slow(name: str, extra: Any = None, *, drawn: Any = None) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            seen["on_loop"] = False
        else:
            seen["on_loop"] = True
        began.set()
        seen["released"] = release.wait(_HANG_S)
        return [[f"scenario={name}"]]

    monkeypatch.setattr(sweep_resolver, "resolve_sweep", slow)
    with TestClient(app_module.app, base_url="http://127.0.0.1") as client:
        results: list[Any] = []
        poster = threading.Thread(
            target=lambda: results.append(
                client.post(
                    "/api/run",
                    json={"scenario": "a/one", "extra_overrides": [], "sweeper": "x"},
                )
            )
        )
        poster.start()
        try:
            assert began.wait(_HANG_S)
            during = client.get("/api/run/status")
            answered_while_resolving = not release.is_set()
        finally:
            release.set()
            poster.join(_HANG_S)
    assert not poster.is_alive(), "the run request never finished"
    assert during.status_code == 200
    assert answered_while_resolving
    assert seen == {"on_loop": False, "released": True}
    assert results and results[0].json() == {"status": "started"}
    assert started["jobs"] == [["scenario=a/one"]]


def test_hydra_is_composed_under_the_shared_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydra's global state is cleared and composed holding HYDRA_LOCK (#1345)."""
    from autoware_carla_scenario.sweeper import expand

    held: list[bool] = []
    real_compose = sweep_resolver.compose

    def spy(*args: Any, **kwargs: Any) -> Any:
        held.append(sweep_resolver.HYDRA_LOCK.locked())
        return real_compose(*args, **kwargs)

    monkeypatch.setattr(sweep_resolver, "compose", spy)
    monkeypatch.setattr(expand, "expand_config", lambda cfg, *a, **k: [[]])
    assert sweep_resolver.resolve_sweep("cut_in/left") == [["scenario=cut_in/left"]]
    assert held == [True]
    assert not sweep_resolver.HYDRA_LOCK.locked()


# ── The run body's types (#1500) ───────────────────────────────────────────

#: Bodies /api/run and /api/run/preview refuse, and what the refusal names.
_BAD_BODIES = [
    pytest.param([1], "JSON object", id="not-an-object"),
    pytest.param("x", "JSON object", id="a-string-body"),
    pytest.param(None, "JSON object", id="null-body"),
    pytest.param(
        {"scenario": "a", "extra_overrides": "carla.port=2001"},
        "extra_overrides",
        id="overrides-a-string",
    ),
    pytest.param(
        {"scenario": "a", "extra_overrides": ["x=1", 2]},
        "extra_overrides",
        id="overrides-a-non-string-item",
    ),
    pytest.param(
        {"scenario": "a", "extra_overrides": {"x": "1"}},
        "extra_overrides",
        id="overrides-an-object",
    ),
    pytest.param({"scenario": ["a"]}, "scenario", id="scenario-a-list"),
    pytest.param({"scenario": None}, "scenario", id="scenario-null"),
    pytest.param({"scenario": "a", "sweeper": 1}, "sweeper", id="sweeper-a-number"),
    pytest.param({"scenario": "a", "launcher": "ssh"}, "launcher", id="launcher"),
    pytest.param({"scenario": "a", "launcher": 1}, "launcher", id="launcher-number"),
    pytest.param(
        {"scenario": "a", "launcher": "slurm", "slurm": ["main"]},
        "slurm",
        id="slurm-a-list",
    ),
    pytest.param(
        {"scenario": "a", "launcher": "slurm", "slurm": {"partition": ["main"]}},
        "slurm.partition",
        id="slurm-text-a-list",
    ),
    pytest.param(
        {"scenario": "a", "launcher": "slurm", "slurm": {"cases_per_task": True}},
        "slurm.cases_per_task",
        id="slurm-number-a-bool",
    ),
    pytest.param(
        {"scenario": "a", "launcher": "slurm", "slurm": {"setup": [1]}},
        "slurm.setup",
        id="slurm-setup-a-number",
    ),
    pytest.param(
        {"scenario": "a", "launcher": "slurm", "slurm": {"env": ["A=1"]}},
        "slurm.env",
        id="slurm-env-a-list",
    ),
]


#: Sent as written (``json=None`` would send no body at all).
_JSON = {"Content-Type": "application/json"}


@pytest.mark.parametrize(("body", "names"), _BAD_BODIES)
def test_a_run_body_of_the_wrong_type_is_refused_before_it_runs(
    started: dict[str, Any], client: TestClient, body: Any, names: str
) -> None:
    """A wrong body is a 400 with the run's error shape, not a 500 nor a run
    of a string's characters as overrides."""
    response = client.post("/api/run", content=json.dumps(body), headers=_JSON)
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert names in response.json()["message"]
    assert started == {}


@pytest.mark.parametrize(("body", "names"), _BAD_BODIES)
def test_a_preview_body_of_the_wrong_type_is_refused(
    client: TestClient, body: Any, names: str
) -> None:
    response = client.post("/api/run/preview", content=json.dumps(body), headers=_JSON)
    assert response.status_code == 400
    command = response.json()["command"]
    assert command.startswith("# ") and names in command


def test_the_run_panel_s_body_is_accepted(
    started: dict[str, Any], client: TestClient
) -> None:
    """What base.html's collectRunOptions() and scenario.html send."""
    panel = {
        "scenario": "a/one",
        "extra_overrides": ["carla.port=2001"],
        "sweeper": "",
        "launcher": "local",
    }
    assert client.post("/api/run", json=panel).json() == {"status": "started"}
    assert started["jobs"] == [["scenario=a/one"]]
    assert started["extra"] == ["carla.port=2001"]
    preview = client.post("/api/run/preview", json=panel)
    assert preview.status_code == 200
    assert preview.json()["command"] == "uv run scenario scenario=a/one carla.port=2001"
    rerun = {"scenario": "a/one", "extra_overrides": [], "sweeper": ""}
    assert client.post("/api/run", json=rerun).json() == {"status": "started"}


def test_the_local_preview_is_shell_quoted(client: TestClient) -> None:
    """The preview is meant to be copied into a shell (#1512): a glob or a
    space in an argument is quoted, as the SLURM preview quotes it, and the
    sweep note stays an unquoted comment."""
    body = {
        "scenario": "intersection_passing/*",
        "extra_overrides": ["+note=two words"],
        "sweeper": "lanelet",
        "launcher": "local",
    }
    preview = client.post("/api/run/preview", json=body).json()["command"]
    assert preview == (
        "uv run scenario 'scenario=intersection_passing/*' '+note=two words'"
        "  # sweep: lanelet (resolved to N individual jobs)"
    )
    command, _, _ = preview.partition("  # ")
    expected, _ = runner.build_command(
        "intersection_passing/*", ["+note=two words"], "lanelet"
    )
    assert shlex.split(command) == expected


def test_the_page_s_default_preview_is_shell_quoted(client: TestClient) -> None:
    """base.html's preview before the first /api/run/preview answer quotes
    its glob as the endpoint does (#1512)."""
    page = client.get("/").text
    assert ">uv run scenario 'scenario=*/*'</code>" in page
    preview = client.post("/api/run/preview", json={"scenario": "*/*"}).json()
    assert preview["command"] == "uv run scenario 'scenario=*/*'"


def test_null_or_absent_overrides_and_sweeper_are_none(
    started: dict[str, Any], client: TestClient
) -> None:
    body = {"scenario": "a/one", "extra_overrides": None, "sweeper": None}
    assert client.post("/api/run", json=body).json() == {"status": "started"}
    assert started["extra"] == []
    assert client.post("/api/run", json={"scenario": "a/two"}).json() == {
        "status": "started"
    }
    assert started["extra"] == []
