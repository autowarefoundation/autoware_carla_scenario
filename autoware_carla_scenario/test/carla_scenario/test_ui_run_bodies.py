"""Run, preview and SLURM resubmit bodies that are not JSON (#2239).

A body sent as JSON that does not parse -- cut off, garbled, not UTF-8 -- is
answered with 400 rather than read as no body, and a run body must name its
scenarios: a truncated run request never runs ``*/*``, and a garbled
resubmission never drops its flags.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner, scanner, slurm_launch

DATE = "2026-11-20"
_JSON = {"Content-Type": "application/json"}


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    scanner.clear_cache()
    yield tmp_path
    scanner.clear_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app, base_url="http://127.0.0.1")


# ── Bodies that are not JSON ────────────────────────────────────────────────


@pytest.fixture
def started(base: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[list[str]]]:
    runs: list[list[list[str]]] = []

    def fake_start_run(overrides_list: list[list[str]], **_: Any) -> bool:
        runs.append(overrides_list)
        return True

    monkeypatch.setattr(runner, "start_run", fake_start_run)
    monkeypatch.setattr(
        app_module, "_resolve_scenario_names", lambda pattern: [pattern]
    )
    return runs


_GARBLED = [
    b'{"scenario": "intersection_passing/left_turn", "extra_overrides": [',
    b'{"scenario": "a/b"',
    b"not json",
    b'{"scenario": "\xff"}',
    b"[" * 100_000,
]


@pytest.mark.parametrize("body", _GARBLED)
def test_a_run_body_that_is_not_json_is_a_400(
    started: list[Any], client: TestClient, body: bytes
) -> None:
    response = client.post("/api/run", content=body, headers=_JSON)
    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "message": "the request body is not valid JSON",
    }
    assert started == []


@pytest.mark.parametrize(
    ("body", "headers"),
    [(b"", _JSON), (b"", {}), (b"  \n", _JSON), (b"{}", _JSON)],
    ids=["empty-json", "empty-untyped", "blank", "no-scenario"],
)
def test_a_run_body_that_names_no_scenario_is_a_400(
    started: list[Any], client: TestClient, body: bytes, headers: dict[str, str]
) -> None:
    response = client.post("/api/run", content=body, headers=headers)
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert "scenario is required" in response.json()["message"]
    assert started == []


def test_every_scenario_runs_when_asked_for_by_name(
    started: list[Any], client: TestClient
) -> None:
    # What the Run panel sends: the scenario is always named.
    response = client.post(
        "/api/run",
        json={
            "scenario": "*/*",
            "extra_overrides": [],
            "sweeper": "",
            "launcher": "local",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert started == [[["scenario=*/*"]]]


@pytest.mark.parametrize("body", _GARBLED)
def test_a_preview_body_that_is_not_json_is_a_400(
    base: Path, client: TestClient, body: bytes
) -> None:
    response = client.post("/api/run/preview", content=body, headers=_JSON)
    assert response.status_code == 400
    assert response.json() == {"command": "# the request body is not valid JSON"}


def test_an_empty_preview_body_keeps_the_defaults(
    base: Path, client: TestClient
) -> None:
    response = client.post("/api/run/preview", content=b"", headers=_JSON)
    assert response.status_code == 200
    assert "'scenario=*/*'" in response.json()["command"]


def _slurm_batch(base: Path) -> Path:
    batch = base / "outputs" / DATE / "12-00-00"
    (batch / "slurm" / "results").mkdir(parents=True)
    (batch / "slurm" / "submissions").mkdir()
    (batch / "slurm" / "manifest.json").write_text(
        json.dumps({"cases": [{"index": 0, "label": "a"}]})
    )
    return batch


@pytest.fixture
def resubmitted(base: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    _slurm_batch(base)
    ran: list[list[str]] = []
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))

    def run_cli(args: list[str]) -> tuple[int, str]:
        ran.append(args)
        return 0, ""

    monkeypatch.setattr(slurm_launch, "run_cli", run_cli)
    return ran


def test_a_garbled_resubmit_body_is_a_400(
    resubmitted: list[list[str]], client: TestClient
) -> None:
    response = client.post(
        f"/api/slurm/{DATE}/12-00-00/resubmit",
        content=b'{"failed": true, "array_parallelism": 2',
        headers=_JSON,
    )
    assert response.status_code == 400
    assert response.json() == {
        "ok": False,
        "output": "the request body is not valid JSON",
    }
    # Not resubmitted without its flags.
    assert resubmitted == []


def test_an_empty_resubmit_body_resubmits_with_no_flags(
    resubmitted: list[list[str]], client: TestClient
) -> None:
    response = client.post(
        f"/api/slurm/{DATE}/12-00-00/resubmit", content=b"", headers=_JSON
    )
    assert response.status_code == 200
    assert len(resubmitted) == 1
    assert "--failed" not in resubmitted[0]


# ── Extra overrides that are flags (#2440) ──────────────────────────────────

#: Lines that would reach ``scenario`` as flags: ``--cfg=job`` prints the
#: config and exits 0 (a job "passed" though nothing ran), ``--multirun``
#: sends the results elsewhere.
_FLAGS = ["--cfg=job", "--multirun", "-m", "-c job", "--help"]


@pytest.mark.parametrize("flag", _FLAGS)
@pytest.mark.parametrize("launcher", ["local", "slurm"])
def test_a_run_with_a_flag_for_an_override_is_a_400(
    started: list[Any],
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    launcher: str,
) -> None:
    submitted: list[list[str]] = []
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))
    monkeypatch.setattr(
        slurm_launch, "start_submit", lambda command, base: submitted.append(command)
    )
    response = client.post(
        "/api/run",
        json={
            "scenario": "a/b",
            "extra_overrides": ["server.host=h", flag],
            "launcher": launcher,
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert f"'{flag}'" in body["message"]
    assert "an override (key=value), not a flag" in body["message"]
    assert started == [] and submitted == []


@pytest.mark.parametrize("launcher", ["local", "slurm"])
def test_a_preview_with_a_flag_for_an_override_is_a_400(
    base: Path, client: TestClient, launcher: str
) -> None:
    response = client.post(
        "/api/run/preview",
        json={
            "scenario": "a/b",
            "extra_overrides": ["--cfg=job"],
            "launcher": launcher,
        },
    )
    assert response.status_code == 400
    command = response.json()["command"]
    assert command.startswith("# extra override '--cfg=job' starts with '-'")


def test_overrides_that_only_hold_a_dash_are_still_overrides() -> None:
    # Hydra's overrides start with a key, '+' or '~'; a value may be negative.
    extra = ["a=-1", "+b=--x", "++c=1", "~d", "~e=-2", "f=-"]
    body = app_module.parse_run_body({"scenario": "a/b", "extra_overrides": extra})
    assert body.extra_overrides == extra


def test_the_slurm_command_refuses_a_flag_through_the_same_check() -> None:
    from autoware_carla_scenario.ui import request_values, slurm_options

    assert app_module.check_extra_overrides is request_values.check_extra_overrides
    with pytest.raises(ValueError, match="not a flag"):
        slurm_options.build_command(
            ["scenario"], "a/b", ["--multirun"], "", slurm_options.SlurmOptions()
        )
    with pytest.raises(ValueError, match="not a flag"):
        slurm_launch.build_command(
            "a/b", ["--cfg=job"], "", slurm_launch.SlurmOptions()
        )
    assert "a=-1" in slurm_options.build_command(
        ["scenario"], "a/b", ["a=-1"], "", slurm_options.SlurmOptions()
    )
