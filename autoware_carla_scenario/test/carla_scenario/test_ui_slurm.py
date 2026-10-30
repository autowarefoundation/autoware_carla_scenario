"""Running a batch from the viewer through SLURM, and following it.

The end-to-end test submits through the viewer's API to the fake SLURM of
``data/fake_slurm`` (sbatch, squeue, scancel as local subprocesses) with the
fake world in every process, as ``test_slurm_launcher.py`` does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import scanner, slurm_launch
from autoware_carla_scenario.ui.slurm_launch import SlurmOptions

from ._slurm_ports import RANGE_SLOTS, fake_carlas, port_range

_DATA = Path(__file__).parent / "data" / "fake_slurm"


# ── The command ────────────────────────────────────────────────────────────


def test_the_options_become_launcher_overrides() -> None:
    options = SlurmOptions.from_request(
        {
            "partition": "main",
            "array_parallelism": "2",
            "cases_per_task": "",
            "time": "01:30:00",
            "mem": "12G",
            "gres": "shard:1",
            "start_stagger_seconds": "30",
            "env": "ONEPLANNER_MAP_DIR=/maps/town 10\nFOO=a'b",
        }
    )
    assert options.overrides() == [
        "launcher=slurm",
        "launcher.partition=main",
        "launcher.array_parallelism=2",
        "launcher.time=01:30:00",
        "launcher.mem=12G",
        "launcher.gres=shard:1",
        "launcher.start_stagger_seconds=30.0",
        "++launcher.env.FOO='a\\'b'",
        "++launcher.env.ONEPLANNER_MAP_DIR='/maps/town 10'",
    ]


def test_a_submit_host_with_other_paths_sets_the_node_s() -> None:
    from hydra.core.override_parser.overrides_parser import OverridesParser

    options = SlurmOptions.from_request(
        {
            "remote_cwd": "workspace/OnePlanner",
            "python": ".venv/bin/python",
            "setup": 'source ~/workspace/opt/env.sh\n\necho "it\'s $HOME"\n',
        }
    )
    overrides = options.overrides()
    assert overrides == [
        "launcher=slurm",
        "launcher.remote_cwd=workspace/OnePlanner",
        "launcher.python=.venv/bin/python",
        "launcher.setup=['source ~/workspace/opt/env.sh','echo \"it\\'s $HOME\"']",
    ]
    parser = OverridesParser.create()
    assert parser.parse_override(overrides[-1]).value() == [
        "source ~/workspace/opt/env.sh",
        'echo "it\'s $HOME"',
    ]
    assert SlurmOptions.from_request({"remote_cwd": "~/work"}).remote_cwd == "~/work"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("array_parallelism", "0", "at least 1"),
        ("cases_per_task", "two", "must be a number"),
        ("time", "tomorrow", "--time"),
        ("partition", "main; rm -rf", "characters"),
        ("env", "1BAD=x", "not an environment variable name"),
        ("env", "NOVALUE", "NAME=value"),
        ("remote_cwd", "work; rm -rf", "not a path"),
        ("remote_cwd", "~bob/work", "not a path"),
        ("python", "$(evil)", "not a path"),
        ("setup", "echo ${HOME}", "rather than"),
        ("env", "X=${HOME}", "no \\$\\{"),
        ("env", "X=a${oc.env:HOME}", "exported as written"),
    ],
)
def test_options_sbatch_would_not_take_are_refused(
    field: str, value: str, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        SlurmOptions.from_request({field: value})


# Values Hydra must read back exactly: plain, its keywords and numbers,
# backslashes anywhere (a run before a quote, at the end), quotes of both
# kinds, and the characters its grammar gives a meaning.
_ROUND_TRIP_VALUES = [
    "",
    "plain",
    "/maps/town 10",
    "a\\b",
    "C:\\dir\\",
    "C:\\dir\\\\",
    "\\",
    "\\\\\\",
    "\\start",
    "a'b",
    "'",
    "''",
    "'quoted'",
    'a"b',
    '"',
    "a\\'b",
    "a\\\\'b",
    "\\'",
    "it's \\\\'quoted\\' \\",
    'a\\"b\\',
    "\"'\\\"\\'",
    "ünïcödé 東京 🚗",
    "a b  c ",
    " leading",
    "k=v",
    "a=b=c",
    "a,b",
    "[1,2]",
    "{a: 1}",
    "(x)",
    "x:y",
    "$HOME/x",
    "$",
    "%d*?|@",
    "a\tb\nc",
    "line\n",
    "#comment",
    "null",
    "NULL",
    "true",
    "False",
    "inf",
    "-inf",
    "nan",
    "1",
    "-1",
    "+1",
    "01",
    "1_000",
    "1.5",
    ".5",
    "1.",
    "1e3",
    "1.5e-3",
    "1.5.2",
    "01:30:00",
    "12G",
    "shard:1",
    "~/work",
    "=",
    "-",
]


@pytest.mark.parametrize("value", _ROUND_TRIP_VALUES)
def test_an_env_value_reaches_the_config_as_written(value: str) -> None:
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    (override,) = SlurmOptions(env={"NAME": value}).overrides()[1:]
    (parsed,) = OverridesParser.create().parse_overrides([override])
    assert parsed.key_or_group == "launcher.env.NAME"
    assert parsed.value() == value
    # What the launcher reads: the composed config, resolved.
    resolved = OmegaConf.to_container(OmegaConf.create({"v": value}), resolve=True)
    assert resolved == {"v": value}


@pytest.mark.parametrize("value", _ROUND_TRIP_VALUES)
def test_a_launcher_text_key_and_a_setup_line_reach_the_config_as_written(
    value: str,
) -> None:
    from hydra.core.override_parser.overrides_parser import OverridesParser

    overrides = SlurmOptions(partition=value, setup=[value, "x"]).overrides()
    parsed = OverridesParser.create().parse_overrides(overrides[1:])
    assert [p.value() for p in parsed] == [value, [value, "x"]]


@pytest.mark.parametrize("name", ["true", "null", "NAN", "_X1", "inf", "a"])
def test_an_env_name_is_the_key_hydra_reads(name: str) -> None:
    from hydra.core.override_parser.overrides_parser import OverridesParser

    (override,) = SlurmOptions.from_request({"env": {name: "v"}}).overrides()[1:]
    (parsed,) = OverridesParser.create().parse_overrides([override])
    assert parsed.key_or_group == f"launcher.env.{name}"
    assert parsed.value() == "v"


def test_a_backslash_is_not_doubled() -> None:
    # #740: `a\b` (one backslash) was written `'a\\b'`, read as two.
    assert SlurmOptions(env={"X": "a\\b"}).overrides()[-1] == "++launcher.env.X='a\\b'"
    assert (
        SlurmOptions(env={"X": "C:\\dir\\"}).overrides()[-1]
        == "++launcher.env.X='C:\\dir\\\\'"
    )
    # A value Hydra would read as a number, a bool or null stays a string.
    assert SlurmOptions(env={"X": "null"}).overrides()[-1] == "++launcher.env.X='null'"
    # A setup line may hold a backslash now that it is escaped as Hydra reads it.
    assert SlurmOptions.from_request({"setup": "echo a\\b"}).setup == ["echo a\\b"]


def test_the_command_submits_and_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIEWER_SCENARIO_COMMAND", raising=False)
    options = SlurmOptions(partition="main")
    assert slurm_launch.build_command(
        "cut_in/*", ["map=town10hd_opt"], "lanelet_constraint", options
    ) == [
        "uv",
        "run",
        "scenario",
        "--multirun",
        "scenario=cut_in/*",
        "map=town10hd_opt",
        "launcher=slurm",
        "launcher.partition=main",
        "--no-wait",
    ]
    monkeypatch.setenv("VIEWER_SCENARIO_COMMAND", ".venv/bin/scenario")
    assert slurm_launch.build_command("x", [], "", SlurmOptions())[:2] == [
        ".venv/bin/scenario",
        "scenario=x",
    ]


# ── Without SLURM ──────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    scanner.clear_cache()
    return TestClient(app_module.app)


def test_without_sbatch_slurm_is_offered_disabled_with_why(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    launchers = client.get("/api/launchers").json()
    assert launchers["local"]["available"]
    assert not launchers["slurm"]["available"]
    assert "sbatch" in launchers["slurm"]["reason"]
    run = client.post(
        "/api/run", json={"scenario": "x", "launcher": "slurm", "slurm": {}}
    ).json()
    assert run["status"] == "error" and "sbatch" in run["message"]
    # The page offers it, disabled until /api/launchers says otherwise.
    page = client.get("/").text
    assert '<option value="slurm" disabled>' in page
    preview = client.post(
        "/api/run/preview",
        json={"scenario": "x", "launcher": "slurm", "slurm": {"partition": "main"}},
    ).json()["command"]
    assert "launcher=slurm launcher.partition=main --no-wait" in preview


def _manifest_batch(base: Path, time_: str = "12-00-00") -> Path:
    """A SLURM batch of four cases as its tasks left it, the job gone: #0
    passed, #1 failed, #2 errored, #3 never ran."""
    batch = base / "outputs" / "2026-10-19" / time_
    root = batch / "slurm"
    (root / "submissions").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "tasks").mkdir()
    stems = ["Case", "Case-1", "Case-2", "Case-3"]
    manifest = {
        "version": 1,
        "created": "2026-10-19T03:00:00+00:00",
        "batch_dir": str(batch),
        "scenario_names": ["s"],
        "expand": True,
        "launcher": {"type": "slurm", "array_parallelism": 2},
        "cases": [
            {
                "index": i,
                "label": f"s#{i + 1}",
                "scenario": "s",
                "overrides": [f"x={i}"],
                "case_overrides": [f"x={i}"],
                "stem": stem,
            }
            for i, stem in enumerate(stems)
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "submissions" / "0.json").write_text(
        json.dumps(
            {
                "index": 0,
                "job_id": "999999",
                "cases": [0, 1, 2, 3],
                "chunks": [[0, 1], [2, 3]],
            }
        )
    )
    (root / "tasks" / "0-1.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "CARLA did not start",
                "done": [2],
                "host": "node1",
            }
        )
    )
    for i, (passed, errored) in enumerate(
        [(True, False), (False, False), (False, True)]
    ):
        result = {
            "passed": passed,
            "message": f"case {i}",
            "elapsed_seconds": 1.0 + i,
            "condition_statuses": [],
        }
        (root / "results" / f"{i}.json").write_text(
            json.dumps(
                {
                    "index": i,
                    "label": f"s#{i + 1}",
                    "stem": stems[i],
                    "result": result,
                    "errored": errored,
                }
            )
        )
        (batch / f"{stems[i]}_result.json").write_text(json.dumps(result))
        (batch / f"{stems[i]}.trajectory.jsonl").write_text('{"type":"world"}\n')
    return batch


def _fake_squeue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, states: str) -> None:
    """An squeue that lists *states* (``"999999_1 RUNNING"`` lines)."""
    bin_dir = tmp_path / "squeue-bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in ("squeue", "sbatch"):
        script = bin_dir / tool
        script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{states}EOF\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_a_running_batch_s_cases_and_tasks(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _manifest_batch(tmp_path)
    # Task 1 (cases 2 and 3) still runs case 3.
    (batch / "slurm" / "tasks" / "0-1.json").write_text(
        json.dumps({"state": "running", "done": [2], "current": 3, "host": "node1"})
    )
    _fake_squeue(tmp_path, monkeypatch, "999999_1 RUNNING\n")
    status = slurm_launch.batch_status(batch)
    assert status is not None
    assert [c["state"] for c in status["cases"]] == [
        "passed",
        "failed",
        "errored",
        "running",
    ]
    assert status["active"] and not status["summary"]
    assert status["submissions"][0]["tasks"][1]["squeue"] == "RUNNING"
    # The viewer lists it as a batch before it is collected, and reads it
    # again each time.
    sessions = scanner.scan_sessions(tmp_path)
    assert [(s.session_type, s.passed_count, s.total_count) for s in sessions] == [
        ("batch", 1, 4)
    ]
    page = client.get("/session/batch/2026-10-19/12-00-00")
    assert page.status_code == 200
    assert 'id="slurm-panel"' in page.text
    assert client.get("/session/batch/2026-10-19/12-00-00/1").status_code == 200
    assert (
        "No result yet" not in client.get("/session/batch/2026-10-19/12-00-00/0").text
    )
    (batch / "slurm" / "results" / "1.json").unlink()
    items = scanner.load_session(tmp_path, "batch", "2026-10-19", "12-00-00")
    assert [i.passed for i in items] == [True, None, False, None]


def test_an_ended_batch_names_its_missing_cases(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest_batch(tmp_path)
    _fake_squeue(tmp_path, monkeypatch, "")
    status = client.get("/api/slurm/2026-10-19/12-00-00/status").json()
    assert [c["state"] for c in status["cases"]] == [
        "passed",
        "failed",
        "errored",
        "missing",
    ]
    assert "CARLA did not start" in status["cases"][3]["message"]
    assert status["counts"] == {"passed": 1, "failed": 1, "errored": 1, "missing": 1}
    assert not status["active"]
    assert client.get("/api/slurm/2026-10-19/nope/status").status_code == 404
    assert client.post("/api/slurm/2026-10-19/12-00-00/rm").status_code == 404


def test_collect_writes_the_summary(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _manifest_batch(tmp_path)
    _fake_squeue(tmp_path, monkeypatch, "")
    collected = client.post("/api/slurm/2026-10-19/12-00-00/collect").json()
    assert collected["ok"] and collected["code"] == 1  # not every case passed
    assert "1 case(s) have no result" in collected["output"]
    summary = json.loads((batch / "batch_results.json").read_text())
    assert [e["passed"] for e in summary] == [True, False, False, False]
    assert scanner.load_session(tmp_path, "batch", "2026-10-19", "12-00-00")[0].passed


# ── Through the fake SLURM ─────────────────────────────────────────────────

_SITECUSTOMIZE = f"""
import importlib.abc, importlib.util, sys
_NAME = "autoware_carla_scenario.examples.run"
class _FakeWorld(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _NAME:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        original = spec.loader.exec_module
        def exec_module(module):
            original(module)
            sys.path.insert(0, {str(_DATA)!r})
            import fake_world
            fake_world.install(module)
        spec.loader.exec_module = exec_module
        return spec
sys.meta_path.insert(0, _FakeWorld())
"""


@pytest.fixture
def port_base() -> Iterator[int]:
    """Ports no other test uses meanwhile (#699), held past fake_slurm's clean-up."""
    with port_range() as base:
        yield base


@pytest.fixture
def fake_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port_base: int
) -> Iterator[Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("sbatch", "squeue", "scancel", "scontrol"):
        script = bin_dir / command
        script.write_text(
            f'#!/bin/sh\nexec {sys.executable} {_DATA / "fake_slurm.py"} {command} "$@"\n'
        )
        script.chmod(0o755)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    state = tmp_path / "slurm_state"
    record = tmp_path / "record"
    record.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH")])),
    )
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))
    monkeypatch.setenv("FAKE_SLURM_KILL_WAIT", "5")
    monkeypatch.setenv("FAKE_RECORD_DIR", str(record))
    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    monkeypatch.setenv(
        "VIEWER_SCENARIO_COMMAND",
        f"{sys.executable} -m autoware_carla_scenario.examples.run",
    )
    yield bin_dir
    for job_dir in (state / "jobs").glob("*") if (state / "jobs").is_dir() else []:
        subprocess.run([str(bin_dir / "scancel"), job_dir.name], check=False)
    for carla in record.glob("carla-*.json"):
        for pid in fake_carlas(record, json.loads(carla.read_text())["port"]):
            try:
                os.kill(pid, 9)
            except OSError:
                pass


def _wait(
    client: TestClient, url: str, done: Any, timeout: float = 120.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        data = client.get(url).json()
        if done(data):
            return data
        assert time.monotonic() < deadline, f"timed out on {url}: {data}"
        time.sleep(0.5)


def test_a_batch_submitted_from_the_viewer_is_followed_and_collected(
    client: TestClient, fake_slurm: Path, port_base: int
) -> None:
    run = client.post(
        "/api/run",
        json={
            "scenario": "intersection_passing/straight_3npc",
            "extra_overrides": [
                "+sweep.odd_sample.count=4",
                "+sweep.odd_sample.seed=1",
                "launcher.poll_seconds=0.2",
                f"launcher.port_base={port_base}",
                f"launcher.port_slots={RANGE_SLOTS}",
            ],
            "sweeper": "lanelet_constraint",
            "launcher": "slurm",
            "slurm": {"array_parallelism": "2", "cases_per_task": "2"},
        },
    ).json()
    assert run == {"status": "submitting"}
    submitted = _wait(
        client, "/api/slurm/submission", lambda d: d["state"] != "submitting"
    )
    assert submitted["state"] == "submitted", submitted["output"]
    assert "--multirun" in submitted["command"] and "--no-wait" in submitted["command"]
    session = submitted["session_url"]
    assert session.startswith("/session/batch/")
    _, _, _, date, time_ = session.split("/")
    status_url = f"/api/slurm/{date}/{time_}/status"
    # Followed until the array and its collect job have ended.
    status = _wait(client, status_url, lambda d: not d["active"] and d["summary"])
    assert status["total"] == 4
    # Every case ran to an outcome (the fake world fails and errors some).
    assert sum(status["counts"].get(k, 0) for k in ("passed", "failed", "errored")) == 4
    assert [len(t["cases"]) for t in status["submissions"][0]["tasks"]] == [2, 2]
    page = client.get(session)
    assert 'id="slurm-panel"' in page.text
    # Resubmitting the failed ones too: a second submission of those.
    failed = [c["index"] for c in status["cases"] if c["state"] != "passed"]
    resubmitted = client.post(
        f"/api/slurm/{date}/{time_}/resubmit",
        json={"failed": True, "array_parallelism": 1},
    ).json()
    assert resubmitted["ok"], resubmitted["output"]
    if not failed:
        assert "Nothing to resubmit" in resubmitted["output"]
        return
    status = _wait(
        client, status_url, lambda d: len(d["submissions"]) == 2 and not d["active"]
    )
    assert status["submissions"][1]["cases"] == len(failed)
