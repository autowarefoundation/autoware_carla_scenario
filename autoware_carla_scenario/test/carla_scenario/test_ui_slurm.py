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
        ("setup", "module load a \\", "ends in a backslash"),
        ("setup", "echo \\\\\\", "ends in a backslash"),
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
    # One that ends in an escaped backslash ends there.
    assert SlurmOptions.from_request({"setup": "echo a\\\\"}).setup == ["echo a\\\\"]


def test_the_job_script_runs_the_fields_as_written(tmp_path: Path) -> None:
    """From the panel's fields to the job script: Hydra parses the overrides
    into the launcher config, OmegaConf resolves it, and the script exports
    each env value and runs each setup line exactly as typed."""
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    from autoware_carla_scenario.slurm import submit
    from autoware_carla_scenario.slurm.manifest import BatchLayout
    from autoware_carla_scenario.slurm.settings import SlurmSettings

    env = {
        "A": "a\\b",
        "B": "C:\\dir\\",
        "C": "it's \\'q\\' \"d\" $HOME `x` *",
        "D": "null",
        "E": "",
    }
    setup = ["X=a\\b", "Y='c\\'", 'Z="$X"\\\\']
    options = SlurmOptions.from_request({"env": env, "setup": "\n".join(setup)})
    cfg = OmegaConf.create({"launcher": {"type": "slurm", "setup": [], "env": {}}})
    ConfigLoaderImpl._apply_overrides_to_config(
        OverridesParser.create().parse_overrides(options.overrides()[1:]), cfg
    )
    launcher = OmegaConf.to_container(cfg.launcher, resolve=True)
    settings = SlurmSettings.from_mapping(launcher)  # type: ignore[arg-type]
    assert settings.setup == tuple(setup)
    assert dict(settings.env) == env
    script = submit.job_script(
        BatchLayout(tmp_path),
        {"python": "python3", "cwd": str(tmp_path), "env": {}},
        settings,
        0,
    ).splitlines()
    start = script.index("# launcher.setup") + 1
    assert script[start : start + len(setup)] == setup
    body = script[start : script.index("# launcher.env") + 1 + len(env)]
    names = [*env, "X", "Y", "Z"]
    dump = "".join(f'printf "%s\\0" "${{{n}-}}";' for n in names)
    done = subprocess.run(
        ["bash", "-c", "\n".join([*body, dump])],
        capture_output=True,
        check=True,
    )
    values = done.stdout.decode().split("\0")[:-1]
    # The setup lines as bash reads them: `a\b` is `ab`, `'c\'` is `c\`.
    assert values == [*env.values(), "ab", "c\\", "ab\\"]


def test_a_command_that_cannot_be_run_fails_the_submission(tmp_path: Path) -> None:
    record = slurm_launch.Submission(command=["scenario", "x\0y"])
    slurm_launch._submit(record.command, tmp_path, record)
    assert record.state == "failed"
    assert "Could not run" in record.output


def test_a_batch_directory_with_spaces_is_read_whole(tmp_path: Path) -> None:
    batch_dir = tmp_path / "My Runs" / "outputs" / "2026-01-01" / "00-00-00"
    line = f"SLURM job 1: 1 case(s) in 1 task(s) of up to 1 case(s); outputs in {batch_dir}"
    record = slurm_launch.Submission(
        command=[sys.executable, "-c", f"print({line!r}); print('more output')"]
    )
    slurm_launch._submit(record.command, tmp_path, record)
    assert record.batch_dir == str(batch_dir)
    assert record.state == "submitted"


def _alive(pid: int) -> bool:
    """Whether *pid* runs (a zombie, ended but not yet reaped, does not)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _group_running(pgid: int) -> list[int]:
    """The processes of group *pgid* that run (zombies left out)."""
    running = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] != "Z" and int(fields[2]) == pgid:
            running.append(int(stat_path.parent.name))
    return running


def _wait_for(predicate: Any, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def test_a_submit_that_hangs_is_stopped_and_a_new_one_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1786: a submit that does not end in time kills its whole process
    group and is recorded failed (124), so the viewer takes the next one."""
    monkeypatch.setattr(slurm_launch, "SUBMIT_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(slurm_launch, "_submission", None)
    pids = tmp_path / "pids"
    # The child starts a grandchild (as `uv run` starts `scenario`), writes
    # both pids, says it started and hangs; stdin must not be the viewer's.
    hang = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
        "print('checking the batch', flush=True)\n"
        "assert sys.stdin.read() == ''\n"
        "time.sleep(60)\n"
    )
    record = slurm_launch.start_submit([sys.executable, "-c", hang], tmp_path)
    started: list[int] = []
    try:
        assert _wait_for(lambda: record.state != "submitting", 30.0)
        started = [int(p) for p in pids.read_text().split()]
        assert record.state == "failed"
        assert record.returncode == slurm_launch.TIMED_OUT == 124
        assert "checking the batch" in record.output
        assert "did not end within 1 s" in record.output
        assert _wait_for(lambda: not any(_alive(p) for p in started), 10.0)
        # The group (the child's session) has no process left running.
        assert _wait_for(lambda: not _group_running(started[0]), 10.0)
    finally:
        for pid in started:  # only what this test started, should it run
            if _alive(pid):
                os.kill(pid, 9)
    batch_dir = tmp_path / "outputs" / "2026-01-01" / "00-00-00"
    line = f"SLURM job 1: 1 case(s); outputs in {batch_dir}"
    again = slurm_launch.start_submit(
        [sys.executable, "-c", f"print({line!r})"], tmp_path
    )
    assert again is not record
    assert _wait_for(lambda: again.state != "submitting", 30.0)
    assert again.state == "submitted"
    assert again.batch_dir == str(batch_dir)


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
    return TestClient(app_module.app, base_url="http://127.0.0.1")


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
    response = client.post(
        "/api/run", json={"scenario": "x", "launcher": "slurm", "slurm": {}}
    )
    # Refused as the batch actions refuse it (#1964).
    assert response.status_code == 503
    run = response.json()
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


def _function(page: str, signature: str) -> str:
    """The source of the JavaScript function *signature* in *page*, up to
    the next function at the same indentation."""
    start = page.index(signature)
    indent = page.rfind("\n", 0, start) + 1
    prefix = page[indent:start]
    end = page.find("\n" + prefix + "function ", start + len(signature))
    end = page.find("\n" + prefix + "panel.", start) if end < 0 else end
    return page[start:end] if end >= 0 else page[start:]


def test_the_batch_page_keeps_following_the_batch_after_a_failed_request(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SLURM panel re-arms its poll whatever the request did, shows the
    error, reports an action's error body or network error, and keeps the
    new-results hint until the page is reloaded (#2007)."""
    _manifest_batch(tmp_path)
    _fake_squeue(tmp_path, monkeypatch, "")
    page = client.get("/session/batch/2026-10-19/12-00-00").text
    script = page.split('id="slurm-panel"', 1)[1]
    refresh = _function(script, "function refresh()")
    assert "r.ok" in refresh
    assert ".catch(" in refresh and "slurm-state" in refresh
    assert ".finally(" in refresh
    rearm = refresh.split(".finally(", 1)[1]
    assert "setTimeout(refresh" in rearm
    actions = script.split("querySelectorAll('.slurm-btn').forEach", 1)[1]
    assert "res.output || res.detail" in actions
    assert ".catch(" in actions.split(".finally(", 1)[0]
    render = _function(script, "function render(st)")
    hint = "New results: reload the page"
    assert hint in render
    # The hint follows a flag set for good, not a change since the last poll.
    assert "lastDone" not in script
    assert "if (newResults)" in render.split(hint, 1)[0]
    # The run panel's SLURM submit reports a lost request and starts afresh.
    submit = _function(page, "function submitSlurm(body)")
    assert submit.count(".catch(e => showSubmitError(String(e)))") == 2
    reset = submit.split("fetch(", 1)[0]
    assert "removeAttribute('title')" in reset
    assert "fill.className = 'bg-blue-600" in reset


def _pending_batch(base: Path, time_: str = "13-00-00") -> Path:
    """A SLURM batch of three cases submitted, none of them collected yet."""
    batch = base / "outputs" / "2026-10-19" / time_
    root = batch / "slurm"
    (root / "results").mkdir(parents=True)
    manifest = {
        "version": 1,
        "batch_dir": str(batch),
        "scenario_names": ["s"],
        "launcher": {"type": "slurm"},
        "cases": [
            {"index": i, "label": f"s#{i + 1}", "scenario": "s", "stem": f"C-{i}"}
            for i in range(3)
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return batch


def test_a_batch_s_pending_cases_are_counted_apart(tmp_path: Path) -> None:
    """Cases without a result yet are pending (#1820): all of a batch none of
    whose cases was collected, and the one case of a batch that never ran."""
    _pending_batch(tmp_path)
    _manifest_batch(tmp_path)
    sessions = scanner.scan_sessions(tmp_path)
    assert [
        (s.time, s.passed_count, s.pending_count, s.total_count) for s in sessions
    ] == [("13-00-00", 0, 3, 3), ("12-00-00", 1, 1, 4)]


def test_an_all_pending_batch_is_not_listed_as_failed(
    client: TestClient, tmp_path: Path
) -> None:
    """``/`` and the ``/compare`` picker show an all-pending batch's cases as
    pending, in a neutral colour, not as failures (#1820)."""
    _pending_batch(tmp_path)
    index = client.get("/")
    assert index.status_code == 200
    assert "3 pending" in index.text
    assert "0/3 passed" in index.text
    assert "bg-red-100" not in index.text
    compare = client.get("/compare")
    assert compare.status_code == 200
    row = compare.text[compare.text.index("2026-10-19 13-00-00") :]
    row = row[: row.index("</tr>")]
    assert "0/3" in row and "3 pending" in row
    assert (
        "text-red-700" not in row
        and "text-right whitespace-nowrap text-gray-700" in row
    )


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


@pytest.mark.parametrize(
    "content", ["", "{not json", "[1, 2]", "null"], ids=["empty", "bad", "list", "null"]
)
def test_a_batch_with_an_unreadable_submission_record_is_shown_with_a_note(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """#1356: a corrupt record does not keep the batch's page (and its
    Resubmit) from the user: that submission is noted, the others shown."""
    batch = _manifest_batch(tmp_path)
    path = batch / "slurm" / "submissions" / "1.json"
    path.write_text(content)
    _fake_squeue(tmp_path, monkeypatch, "")
    response = client.get("/api/slurm/2026-10-19/12-00-00/status")
    assert response.status_code == 200
    status = response.json()
    assert [c["state"] for c in status["cases"]] == [
        "passed",
        "failed",
        "errored",
        "missing",
    ]
    assert "CARLA did not start" in status["cases"][3]["message"]
    first, second = status["submissions"]
    assert first["job_id"] == "999999" and not first["unreadable"]
    assert second["unreadable"] and str(path) in second["note"]
    assert not status["active"]
    assert client.get("/session/batch/2026-10-19/12-00-00").status_code == 200


@pytest.mark.parametrize("kind", ["in-flight", "timed-out", "stale"])
def test_a_submission_with_no_job_id_names_no_job_and_its_cases_are_not_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """#1791: a resubmit of case 3 whose sbatch is in flight, or timed out
    (the job may exist), shows no "job None" and leaves case 3 pending or
    unknown, not missing; a stale record's case is missing, naming no job."""
    from autoware_carla_scenario.slurm.manifest import SBATCH_IN_FLIGHT_SECONDS

    batch = _manifest_batch(tmp_path)
    record: dict[str, Any] = {
        "index": 1,
        "job_id": None,
        "submitted_at": time.time()
        - (SBATCH_IN_FLIGHT_SECONDS + 1 if kind == "stale" else 0),
        "cases": [3],
        "chunks": [[3]],
    }
    if kind == "timed-out":
        record["sbatch_timed_out"] = True
        record["sbatch_timed_out_at"] = "2026-11-06T12:00:00+09:00"
    (batch / "slurm" / "submissions" / "1.json").write_text(json.dumps(record))
    asked: list[str] = []
    real = slurm_launch.squeue_states

    def squeue_states(job: str) -> Any:
        asked.append(job)
        return real(job)

    monkeypatch.setattr(slurm_launch, "squeue_states", squeue_states)
    _fake_squeue(tmp_path, monkeypatch, "")
    status = slurm_launch.batch_status(batch)
    assert status is not None
    assert asked == ["999999"]  # never squeue -j None
    want = {"in-flight": "pending", "timed-out": "unknown", "stale": "missing"}[kind]
    assert [c["state"] for c in status["cases"]] == [
        "passed",
        "failed",
        "errored",
        want,
    ]
    message = status["cases"][3]["message"]
    assert "None" not in message and "job" in message
    if kind == "timed-out":
        assert "timed out at 2026-11-06T12:00:00+09:00" in message
        assert "squeue --me" in message
    second = status["submissions"][1]
    assert second["job_id"] is None and not second["unreadable"]
    assert second["in_flight"] is (kind == "in-flight")
    assert second["sbatch_timed_out"] is (kind == "timed-out")
    note = second["note"]
    assert "None" not in note and "ended" not in note
    if kind == "in-flight":
        assert note == "no job id yet (sbatch in flight)"
    elif kind == "timed-out":
        assert "sbatch timed out" in note and "squeue --me" in note
    else:
        assert "stale" in note
    # Being submitted is followed as a queued job is; the others are not.
    assert status["active"] is (kind == "in-flight")
    response = client.get("/api/slurm/2026-10-19/12-00-00/status")
    assert response.status_code == 200
    assert "job None" not in response.text
    assert client.get("/session/batch/2026-10-19/12-00-00").status_code == 200


@pytest.mark.parametrize(
    "content", ["[]", "null", '{"version": true}'], ids=["list", "null", "bool"]
)
def test_a_batch_whose_manifest_is_not_an_object_is_not_a_slurm_batch(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """#1630: a manifest that is not an object is a 404, not a 500."""
    batch = _manifest_batch(tmp_path)
    (batch / "slurm" / "manifest.json").write_text(content)
    _fake_squeue(tmp_path, monkeypatch, "")
    assert slurm_launch.batch_status(batch) is None
    response = client.get("/api/slurm/2026-10-19/12-00-00/status")
    assert response.status_code == 404


@pytest.mark.parametrize(
    "result",
    [{"passed": "true"}, {"passed": 1}, "passed", [True], None, "missing"],
    ids=["str-true", "int", "str-result", "list-result", "null-result", "no-result"],
)
def test_a_result_with_no_verdict_is_shown_failed_and_collected_as_an_error(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: Any,
) -> None:
    """#1718: case 0's record holds no boolean verdict: the batch status is
    no 500 and shows it failed, and collect reports it as an error naming its
    file."""
    batch = _manifest_batch(tmp_path)
    path = batch / "slurm" / "results" / "0.json"
    record = json.loads(path.read_text())
    if result == "missing":
        del record["result"]
    else:
        record["result"] = result
    path.write_text(json.dumps(record))
    _fake_squeue(tmp_path, monkeypatch, "")
    response = client.get("/api/slurm/2026-10-19/12-00-00/status")
    assert response.status_code == 200
    status = response.json()
    assert [c["state"] for c in status["cases"]] == [
        "failed",
        "failed",
        "errored",
        "missing",
    ]
    collected = client.post("/api/slurm/2026-10-19/12-00-00/collect").json()
    assert collected["ok"] and collected["code"] == 1
    assert str(path) in collected["output"]
    summary = json.loads((batch / "batch_results.json").read_text())
    assert [e["passed"] for e in summary] == [False, False, False, False]
    assert summary[0].get("error") is True


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
    # Collecting again rewrites collected.json: still collected.
    again = client.post("/api/slurm/2026-10-19/12-00-00/collect").json()
    assert again["ok"] and again["code"] == 1


def test_a_collect_that_could_not_collect_is_not_ok(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1746: ``scenario-slurm collect`` exits 1 on an ``Error:`` too (an
    unreadable manifest here): the viewer reports it failed, not "collected,
    a case failed" -- even with an earlier collect's record in place."""
    batch = _manifest_batch(tmp_path)
    _fake_squeue(tmp_path, monkeypatch, "")
    assert client.post("/api/slurm/2026-10-19/12-00-00/collect").json()["ok"]
    (batch / "slurm" / "manifest.json").write_text("{not json")
    # _slurm_batch() only asks that the manifest be there.
    collected = client.post("/api/slurm/2026-10-19/12-00-00/collect").json()
    assert collected["code"] == 1 and "Error:" in collected["output"]
    assert not collected["ok"]


def test_a_collect_that_exits_1_without_collecting_is_not_ok(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1746: exit 1 with no new ``collected.json`` (a traceback, say) is not
    collected."""
    _manifest_batch(tmp_path)
    monkeypatch.setattr(
        slurm_launch, "run_cli", lambda args, timeout=300.0: (1, "Traceback ...")
    )
    collected = client.post("/api/slurm/2026-10-19/12-00-00/collect").json()
    assert collected == {"ok": False, "code": 1, "output": "Traceback ..."}


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (subprocess.TimeoutExpired(["scenario-slurm"], 300.0), slurm_launch.TIMED_OUT),
        (FileNotFoundError("no python"), slurm_launch.NOT_STARTED),
    ],
    ids=["timeout", "oserror"],
)
@pytest.mark.parametrize("action", ["collect", "cancel", "resubmit"])
def test_a_scenario_slurm_that_could_not_run_is_not_ok(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    code: int,
    action: str,
) -> None:
    """#1746: run_cli turns a timeout or an OSError into a code of its own,
    neither 0 nor 1, which no action reports done."""
    _manifest_batch(tmp_path)
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))

    def run(*args: Any, **kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(slurm_launch.subprocess, "run", run)
    assert slurm_launch.run_cli(["status", "x"])[0] == code
    response = client.post(f"/api/slurm/2026-10-19/12-00-00/{action}", json={})
    assert response.status_code == 200
    out = response.json()
    assert not out["ok"] and out["code"] == code
    assert out["output"] == str(raised)


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


# ── The submission's body (#1500) ──────────────────────────────────────────


@pytest.fixture
def submitted(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The commands submitted through sbatch (nothing is submitted)."""
    commands: list[list[str]] = []
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))
    monkeypatch.setattr(
        slurm_launch,
        "start_submit",
        lambda command, base: commands.append(command),
    )
    monkeypatch.delenv("VIEWER_SCENARIO_COMMAND", raising=False)
    return commands


@pytest.mark.parametrize(
    ("body", "names"),
    [
        (
            {"scenario": "x", "extra_overrides": "carla.port=2001"},
            "extra_overrides",
        ),
        ({"scenario": "x", "extra_overrides": ["a=1", None]}, "extra_overrides"),
        ({"scenario": 1}, "scenario"),
        ({"scenario": "x", "sweeper": ["lanelet_constraint"]}, "sweeper"),
        ({"scenario": "x", "slurm": "partition=main"}, "slurm"),
        ({"scenario": "x", "slurm": {"time": {"h": 2}}}, "slurm.time"),
        ({"scenario": "x", "slurm": {"array_parallelism": [4]}}, "slurm."),
    ],
)
def test_a_slurm_body_of_the_wrong_type_is_not_submitted(
    client: TestClient, submitted: list[list[str]], body: dict[str, Any], names: str
) -> None:
    response = client.post("/api/run", json={**body, "launcher": "slurm"})
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert names in response.json()["message"]
    assert submitted == []


def test_the_slurm_panel_s_body_is_submitted_with_whole_overrides(
    client: TestClient, submitted: list[list[str]]
) -> None:
    """What base.html's collectRunOptions() sends for SLURM: every field text."""
    slurm = {
        "partition": "main",
        "array_parallelism": "4",
        "cases_per_task": "",
        "time": "",
        "mem": "",
        "gres": "",
        "start_stagger_seconds": "",
        "remote_cwd": "",
        "python": "",
        "setup": "",
        "env": "",
    }
    body = {
        "scenario": "x",
        "extra_overrides": ["carla.port=2001"],
        "sweeper": "",
        "launcher": "slurm",
        "slurm": slurm,
    }
    assert client.post("/api/run", json=body).json() == {"status": "submitting"}
    assert submitted == [
        [
            "uv",
            "run",
            "scenario",
            "scenario=x",
            "carla.port=2001",
            "launcher=slurm",
            "launcher.partition=main",
            "launcher.array_parallelism=4",
            "--no-wait",
        ]
    ]


@pytest.mark.parametrize(
    ("body", "names"),
    [
        ([1], "JSON object"),
        ({"failed": "false"}, "failed"),
        ({"array_parallelism": [2]}, "array_parallelism"),
        ({"cases_per_task": True}, "cases_per_task"),
    ],
)
def test_a_resubmit_body_of_the_wrong_type_is_refused(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: Any,
    names: str,
) -> None:
    _manifest_batch(tmp_path)
    ran: list[list[str]] = []
    monkeypatch.setattr(slurm_launch, "availability", lambda: (True, ""))

    def run_cli(args: list[str], timeout: float = 300.0) -> tuple[int, str]:
        ran.append(args)
        return 0, ""

    monkeypatch.setattr(slurm_launch, "run_cli", run_cli)
    response = client.post("/api/slurm/2026-10-19/12-00-00/resubmit", json=body)
    assert response.status_code == 400
    assert not response.json()["ok"] and names in response.json()["output"]
    assert ran == []
    # What session.html sends is accepted.
    accepted = client.post(
        "/api/slurm/2026-10-19/12-00-00/resubmit",
        json={"failed": False, "array_parallelism": "2", "cases_per_task": ""},
    )
    assert accepted.status_code == 200 and accepted.json()["ok"]
    assert ran[-1][-3:] == ["--array-parallelism", "2", "--no-wait"]


def test_new_manifests_are_ordered_by_session_number(tmp_path: Path) -> None:
    """#1845: a submit that finds no batch in its output takes the last new
    manifest, and ``-10`` comes after ``-9`` (a text sort reverses them)."""
    for time_dir in ("12-00-00-10", "12-00-00-9", "12-00-00"):
        manifest = tmp_path / "outputs" / "2026-11-01" / time_dir / "slurm"
        manifest.mkdir(parents=True)
        (manifest / "manifest.json").write_text("{}")
    found = slurm_launch._new_manifests(tmp_path, set())
    assert [p.parents[1].name for p in found] == [
        "12-00-00",
        "12-00-00-9",
        "12-00-00-10",
    ]
