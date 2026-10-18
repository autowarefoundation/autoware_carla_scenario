"""The SLURM launcher (``launcher=slurm``, docs/slurm.md), against a fake SLURM.

``data/fake_slurm/fake_slurm.py`` stands in for sbatch, squeue, scancel and
scontrol, running an array's tasks as local subprocesses, and
``data/fake_slurm/fake_world.py`` for CARLA and the scenarios -- in this process
for a local batch, and in every task through a ``sitecustomize`` on
``PYTHONPATH``.  A case's outcome depends only on its config, so a batch run
under SLURM must report exactly what the same batch run here reports.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.slurm.collect import missing_cases, unfinished_cases
from autoware_carla_scenario.slurm.manifest import BatchLayout, sequential_stems
from autoware_carla_scenario.slurm.ports import (
    PORT_SLOT_WIDTH,
    acquire_slot,
    carla_processes_on_port,
    first_slot,
)
from autoware_carla_scenario.slurm.settings import (
    SlurmSettings,
    launcher_requested_in_argv,
    launcher_type,
    snapshot_environment,
)

_DATA = Path(__file__).parent / "data" / "fake_slurm"
#: A scenario without a sweep of its own: drawing from the ODD needs no map.
_CONCRETE = "intersection_passing/straight_3npc"
_SWEEP = ["+sweep.odd_sample.count=6", "+sweep.odd_sample.seed=1"]
#: Away from the defaults and from a real batch's 20000.
_PORT_BASE = 30000

_SITECUSTOMIZE = f'''
"""Fakes the world of examples.run in every process that imports it."""
import importlib.abc
import importlib.util
import sys

_NAME = "autoware_carla_scenario.examples.run"


class _FakeWorld(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _NAME:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            sys.path.insert(0, {str(_DATA)!r})
            import fake_world

            fake_world.install(module)

        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _FakeWorld())
'''


@pytest.fixture
def fake_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    """A fake SLURM on PATH, and the fake world in this process and its tasks."""
    from autoware_carla_scenario.examples import run

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("sbatch", "squeue", "scancel", "scontrol"):
        script = bin_dir / command
        script.write_text(
            f'#!/bin/sh\nexec {sys.executable} {_DATA / "fake_slurm.py"} '
            f'{command} "$@"\n'
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
    monkeypatch.delenv("FAKE_CASE_SECONDS", raising=False)
    monkeypatch.delenv("FAKE_CRASH_DIGESTS", raising=False)

    monkeypatch.syspath_prepend(str(_DATA))
    import fake_world  # type: ignore[import-not-found]

    monkeypatch.setattr(run, "build_scenario", fake_world.fake_build)
    monkeypatch.setattr(run, "ScenarioQueue", fake_world.FakeQueue)
    monkeypatch.setattr(run, "resolve_map_paths", fake_world.fake_map_paths)
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)

    dirs: list[Path] = []

    def batch_dir(name: str) -> Path:
        path = tmp_path / "outputs" / name
        path.mkdir(parents=True)
        dirs.append(path)
        return path

    yield SimpleNamespace(
        run=run,
        tmp=tmp_path,
        state=state,
        record=record,
        batch_dir=batch_dir,
        monkeypatch=monkeypatch,
    )
    # Nothing a test left behind keeps running: its tasks, their fake CARLAs.
    for job_dir in (state / "jobs").glob("*") if (state / "jobs").is_dir() else []:
        subprocess.run([str(bin_dir / "scancel"), job_dir.name], check=False)
    for carla in record.glob("carla-*.json"):
        port = json.loads(carla.read_text())["port"]
        for pid in carla_processes_on_port(port):
            try:
                # Only the fake ones: a real CARLA may listen on 2000.
                if b"--fake-carla" in Path(f"/proc/{pid}/cmdline").read_bytes():
                    os.kill(pid, 9)
            except (OSError, ProcessLookupError):
                pass


def _run_batch(
    env: SimpleNamespace,
    name: str,
    overrides: list[str],
    *,
    wait: bool | None = None,
) -> tuple[int, Path]:
    """``run_batch`` into a fresh output directory; (exit status, directory)."""
    output = env.batch_dir(name)
    env.monkeypatch.setattr(env.run, "_make_batch_output_dir", lambda: output)
    kwargs: dict[str, Any] = {} if wait is None else {"wait": wait}
    with pytest.raises(SystemExit) as exited:
        env.run.run_batch([_CONCRETE], overrides, expand=True, **kwargs)
    return int(exited.value.code or 0), output


def _slurm(*extra: str) -> list[str]:
    return [
        *_SWEEP,
        "launcher=slurm",
        "launcher.array_parallelism=2",
        "launcher.poll_seconds=0.2",
        f"launcher.port_base={_PORT_BASE}",
        "launcher.port_slots=50",
        *extra,
    ]


def _outputs(batch: Path) -> dict[str, bytes]:
    """The batch's files, but the launcher's own."""
    return {
        p.name: p.read_bytes()
        for p in sorted(batch.iterdir())
        if p.is_file() and p.name != "slurm"
    }


def _summary(text: str, batch: Path) -> str:
    start = text.rindex("Batch Scenario Results")
    return text[start:].replace(str(batch), "<batch>")


def _wait_for(predicate: Any, timeout: float = 60.0, what: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what or predicate}")


def _jobs_done(env: SimpleNamespace) -> bool:
    jobs = env.state / "jobs"
    for job in jobs.iterdir() if jobs.is_dir() else []:
        for task in job.glob("[0-9]*.json"):
            if json.loads(task.read_text()).get("state") in ("PENDING", "RUNNING"):
                return False
    return True


# ---------------------------------------------------------------------------
# The batch, end to end
# ---------------------------------------------------------------------------


def test_a_slurm_batch_reports_what_the_same_local_batch_does(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    local_status, local = _run_batch(fake_slurm, "local", list(_SWEEP))
    local_out = capsys.readouterr().out
    slurm_status, batch = _run_batch(fake_slurm, "slurm", _slurm())
    slurm_out = capsys.readouterr().out

    assert slurm_status == local_status
    # The same files, under the same names, with the same contents -- the
    # summary JSON byte for byte, in the batch's order.
    assert _outputs(batch) == _outputs(local)
    assert (batch / "batch_results.json").read_bytes() == (
        local / "batch_results.json"
    ).read_bytes()
    assert _summary(slurm_out, batch) == _summary(local_out, local)

    # Two tasks of three cases, at most two at once, each on ports of its own.
    layout = BatchLayout(batch)
    submission = layout.read_submission(0)
    assert submission["chunks"] == [[0, 1, 2], [3, 4, 5]]
    sbatch = [json.loads(line) for line in (fake_slurm.state / "sbatch.log").open()]
    assert any(arg == "--array=0-1%2" for arg in sbatch[0]["argv"])
    carlas = [json.loads(p.read_text()) for p in fake_slurm.record.glob("carla-*.json")]
    slurm_carlas = [c for c in carlas if c["port"] != 2000]
    ports = sorted(c["port"] for c in slurm_carlas)
    assert len(ports) == 2 and len(set(ports)) == 2
    for carla in slurm_carlas:
        assert (carla["port"] - _PORT_BASE) % PORT_SLOT_WIDTH == 0
        assert carla["tm_port"] == carla["port"] + 3
        assert f"-carla-streaming-port={carla['port'] + 1}" in carla["extra"]
        assert f"-carla-secondary-port={carla['port'] + 2}" in carla["extra"]
    # Each case ran with its task's server port.
    ran = [json.loads(line) for line in (fake_slurm.record / "stems.jsonl").open()]
    by_pid = {
        int(p.stem.split("-")[1]): json.loads(p.read_text())["port"]
        for p in fake_slurm.record.glob("carla-*.json")
    }
    for entry in ran:
        assert entry["port"] == by_pid[entry["pid"]]
    # Every task finished, and stopped its CARLA.
    for task in range(2):
        status = layout.read_task(0, task)
        assert status is not None and status["state"] == "done"
        assert not layout.task_port(0, task).exists()
    for port in ports:
        assert carla_processes_on_port(port) == []


def test_the_manifest_rebuilds_each_case_exactly(
    fake_slurm: SimpleNamespace,
) -> None:
    from omegaconf import OmegaConf

    _status, batch = _run_batch(fake_slurm, "slurm", _slurm())
    manifest = BatchLayout(batch).read_manifest()
    cases = manifest["cases"]
    assert [c["index"] for c in cases] == list(range(6))
    assert [c["label"] for c in cases] == [f"{_CONCRETE}#{k}" for k in range(1, 7)]
    assert [c["stem"] for c in cases] == sequential_stems(
        [c["scenario_class"] for c in cases]
    )
    assert manifest["launcher"]["array_parallelism"] == 2
    for case in cases:
        # The scenario and overrides alone rebuild the config, sweep case and all.
        cfg = fake_slurm.run._compose_config(case["scenario"], case["overrides"])
        assert OmegaConf.to_container(cfg, resolve=True) == case["config"]
        assert (
            case["overrides"][: len(case["case_overrides"])] == case["case_overrides"]
        )


def test_a_crashed_task_leaves_its_cases_missing_and_resubmit_runs_only_them(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm import cli

    local_status, local = _run_batch(fake_slurm, "local", list(_SWEEP))
    capsys.readouterr()
    ran = [json.loads(line) for line in (fake_slurm.record / "stems.jsonl").open()]
    # The worker of the first task dies on the batch's second case (stem -1).
    second = next(
        (entry["digest"] for entry in ran if entry["stem"].endswith("Scenario-1")), None
    )
    if second is None:
        pytest.skip("the second case errored in the fake world")
    fake_slurm.monkeypatch.setenv("FAKE_CRASH_DIGESTS", second[:16])

    status, batch = _run_batch(fake_slurm, "slurm", _slurm())
    out = capsys.readouterr().out
    assert status == 1
    layout = BatchLayout(batch)
    assert layout.read_result(0) is not None
    assert layout.read_result(1) is None and layout.read_result(2) is None
    assert "No result: the case did not finish under SLURM" in out
    assert "scenario-slurm resubmit" in out
    results = json.loads((batch / "batch_results.json").read_text())
    assert results[1]["error"] is True and results[2]["error"] is True
    # The job script stopped the CARLA the dead worker left running.
    task0 = layout.read_task(0, 0)
    assert task0 is not None
    port = task0["ports"]["rpc"]
    _wait_for(lambda: carla_processes_on_port(port) == [], 20, "the orphaned CARLA")
    assert not layout.task_port(0, 0).exists()

    # Resubmitted: the two cases with no result, and those that errored (the
    # fake world's errored cases raise again, as a case without ground would).
    errored = [
        i
        for i in range(6)
        if (record := layout.read_result(i)) is not None and record["errored"]
    ]
    assert missing_cases(layout) == [1, 2]
    code = cli.main(["resubmit", str(batch), "--dry-run"])
    assert code == 0
    assert f"Resubmitting {2 + len(errored)} case(s)" in capsys.readouterr().out
    code = cli.main(["resubmit", str(batch), "--wait"])
    assert code == local_status
    assert layout.read_submission(1)["cases"] == sorted([1, 2, *errored])
    assert _outputs(batch) == _outputs(local)
    assert (batch / "batch_results.json").read_bytes() == (
        local / "batch_results.json"
    ).read_bytes()
    # Only the cases that error whenever they run are left.
    assert missing_cases(layout) == []
    errored_now = [i for i in range(6) if layout.read_result(i)["errored"]]  # type: ignore[index]
    assert unfinished_cases(layout) == errored_now


def test_scancel_stops_the_tasks_and_their_carla_and_resubmit_finishes_the_batch(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm import cli

    local_status, local = _run_batch(fake_slurm, "local", list(_SWEEP))
    for carla in fake_slurm.record.glob("carla-*.json"):
        carla.unlink()
    fake_slurm.monkeypatch.setenv("FAKE_CASE_SECONDS", "60")
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(), wait=False)
    out = capsys.readouterr().out
    # --no-wait: submitted, and a collect job after the array.
    assert status == 0
    assert "not waiting" in out and "scenario-slurm collect" in out
    layout = BatchLayout(batch)
    submission = layout.read_submission(0)
    assert submission["collect_job_id"]

    _wait_for(
        lambda: len(list(fake_slurm.record.glob("carla-*.json"))) == 2,
        60,
        "two tasks' CARLAs",
    )
    ports = [
        json.loads(p.read_text())["port"]
        for p in fake_slurm.record.glob("carla-*.json")
    ]
    assert all(carla_processes_on_port(port) for port in ports)
    # The array alone, as a user's own scancel would: its collect job runs.
    subprocess.run(["scancel", str(submission["job_id"])], check=True)
    _wait_for(lambda: _jobs_done(fake_slurm), 30, "the cancelled tasks")
    for port in ports:
        _wait_for(lambda p=port: carla_processes_on_port(p) == [], 20, "CARLA")
    for task in range(2):
        task_status = layout.read_task(0, task)
        assert task_status is not None and task_status["state"] == "cancelled"
        assert not layout.task_port(0, task).exists()

    # The collect job ran (once the array ended): every case without a result.
    _wait_for(lambda: (batch / "batch_results.json").exists(), 30, "the collect job")
    capsys.readouterr()
    fake_slurm.monkeypatch.delenv("FAKE_CASE_SECONDS")
    assert cli.main(["resubmit", str(batch), "--wait"]) == local_status
    assert _outputs(batch) == _outputs(local)
    # And collecting again reports the same.
    capsys.readouterr()
    assert cli.main(["collect", str(batch)]) == local_status
    assert (batch / "batch_results.json").read_bytes() == (
        local / "batch_results.json"
    ).read_bytes()


def test_cancel_cancels_every_job_of_the_batch(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm import cli

    fake_slurm.monkeypatch.setenv("FAKE_CASE_SECONDS", "60")
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(), wait=False)
    assert status == 0
    submission = BatchLayout(batch).read_submission(0)
    assert cli.main(["cancel", str(batch)]) == 0
    assert (
        f"Cancelling job(s) {submission['job_id']}, {submission['collect_job_id']}"
        in capsys.readouterr().out
    )
    _wait_for(lambda: _jobs_done(fake_slurm), 30, "the cancelled jobs")
    assert not (batch / "batch_results.json").exists()
    for carla in fake_slurm.record.glob("carla-*.json"):
        port = json.loads(carla.read_text())["port"]
        _wait_for(lambda p=port: carla_processes_on_port(p) == [], 20, "CARLA")


def test_no_wait_submits_and_the_collect_job_writes_the_results(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm import cli

    local_status, local = _run_batch(fake_slurm, "local", list(_SWEEP))
    status, batch = _run_batch(
        fake_slurm, "slurm", _slurm("launcher.cases_per_task=1"), wait=False
    )
    assert status == 0
    layout = BatchLayout(batch)
    assert len(layout.read_submission(0)["chunks"]) == 6
    _wait_for(lambda: _jobs_done(fake_slurm), 120, "the array and its collect job")
    assert (batch / "batch_results.json").read_bytes() == (
        local / "batch_results.json"
    ).read_bytes()
    collect_log = next(layout.logs.glob("collect-*.out")).read_text()
    assert "Batch Scenario Results" in collect_log
    capsys.readouterr()
    assert cli.main(["status", str(batch)]) == 0
    status_out = capsys.readouterr().out
    assert "6 with a result" in status_out
    assert cli.main(["collect", str(batch)]) == local_status


def test_slurm_without_sbatch_is_refused_before_the_batch_expands(
    fake_slurm: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty_bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    def expanded(*_: object, **__: object) -> None:
        raise AssertionError("expanded the batch")

    monkeypatch.setattr(fake_slurm.run, "_expand_batch", expanded)
    with pytest.raises(SystemExit) as exited:
        fake_slurm.run.run_batch([_CONCRETE], _slurm(), expand=True)
    assert exited.value.code == 2
    assert "sbatch" in capsys.readouterr().err


def test_launcher_local_runs_the_batch_here_unchanged(
    fake_slurm: SimpleNamespace,
) -> None:
    plain_status, plain = _run_batch(fake_slurm, "plain", list(_SWEEP))
    local_status, local = _run_batch(fake_slurm, "local", [*_SWEEP, "launcher=local"])
    assert local_status == plain_status
    assert _outputs(local) == _outputs(plain)
    assert not (local / "slurm").exists()
    assert not (fake_slurm.state / "sbatch.log").exists()


def test_the_default_config_has_no_launcher() -> None:
    from autoware_carla_scenario.examples import run

    cfg = run._compose_config(_CONCRETE, [])
    assert "launcher" not in cfg
    assert launcher_type(cfg) == "local"
    slurm = run._compose_config(_CONCRETE, ["launcher=slurm"])
    assert launcher_type(slurm) == "slurm"
    from omegaconf import OmegaConf

    settings = SlurmSettings.from_mapping(
        OmegaConf.to_container(slurm.launcher, resolve=True)  # type: ignore[arg-type]
    )
    assert settings == SlurmSettings()


def test_no_wait_without_slurm_is_refused(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _batch = _run_batch(fake_slurm, "local", list(_SWEEP), wait=False)
    assert status == 1
    assert "--no-wait is for launcher=slurm only" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------


def test_the_launcher_a_command_line_selects() -> None:
    assert launcher_requested_in_argv(["map=x"]) is None
    assert launcher_requested_in_argv(["launcher=slurm"]) == "slurm"
    assert launcher_requested_in_argv(["+launcher.type=slurm"]) == "slurm"
    assert launcher_requested_in_argv(["launcher=slurm", "launcher=local"]) == "local"
    assert launcher_requested_in_argv(["launcher.mem=12G"]) is None
    with pytest.raises(ValueError, match="Unknown launcher.type"):
        launcher_type({"launcher": {"type": "pbs"}})


def test_settings_chunk_the_cases() -> None:
    assert SlurmSettings().chunk_size(24) == 1
    assert SlurmSettings(array_parallelism=2).chunk_size(24) == 12
    assert SlurmSettings(array_parallelism=5).chunk_size(24) == 5
    assert SlurmSettings(array_parallelism=2, cases_per_task=3).chunk_size(24) == 3
    with pytest.raises(ValueError, match="Unknown launcher setting"):
        SlurmSettings.from_mapping({"type": "slurm", "memory": "1G"})
    with pytest.raises(ValueError, match="at least 1"):
        SlurmSettings.from_mapping({"array_parallelism": 0})


def test_start_stagger_spaces_every_task_start(tmp_path: Path) -> None:
    import threading

    from autoware_carla_scenario.slurm.worker import wait_for_start_turn

    lock = BatchLayout(tmp_path).start_lock
    spacing = 0.4
    started: list[float] = []

    def start() -> None:
        wait_for_start_turn(lock, spacing)
        started.append(time.time())

    # Three tasks that start together (a first wave)...
    threads = [threading.Thread(target=start) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # ...and one that starts later, once a slot frees up, right after them.
    start()
    started.sort()
    gaps = [b - a for a, b in zip(started, started[1:])]
    assert len(gaps) == 3
    assert all(gap >= spacing - 0.05 for gap in gaps), gaps
    # Long after the last start, a task starts at once.
    time.sleep(spacing)
    assert wait_for_start_turn(lock, spacing) == 0.0


def test_sbatch_command_and_job_script(tmp_path: Path) -> None:
    from autoware_carla_scenario.slurm.submit import job_script, sbatch_command

    layout = BatchLayout(tmp_path / "2026-10-18" / "12-00-00")
    settings = SlurmSettings.from_mapping(
        {
            "partition": "main",
            "time": "01:00:00",
            "mem": "12G",
            "cpus_per_task": 8,
            "gres": "shard:1",
            "array_parallelism": 2,
            "extra_sbatch_args": ["--nice=10"],
            "setup": ["source /venv/bin/activate"],
            "env": {"ONEPLANNER_X": "set by launcher.env"},
        }
    )
    command = sbatch_command(layout, settings, Path("/job.sh"), 4)
    assert command[:2] == ["sbatch", "--parsable"]
    for expected in (
        "--job-name=acs-2026-10-18-12-00-00",
        "--array=0-3%2",
        "--partition=main",
        "--time=01:00:00",
        "--mem=12G",
        "--cpus-per-task=8",
        "--gres=shard:1",
        "--export=ALL",
        "--signal=B:TERM@120",
        "--nice=10",
    ):
        assert expected in command
    assert command[-1] == "/job.sh"
    assert not any(arg.startswith("--account") for arg in command)

    manifest = {
        "python": "/venv/bin/python",
        "cwd": "/work dir",
        "env": {"ONEPLANNER_MAP_DIR": "/maps/town 10", "GLIBC_TUNABLES": "a=2"},
    }
    script = job_script(layout, manifest, settings, 3)
    lines = script.splitlines()
    snapshot = lines.index("export ONEPLANNER_MAP_DIR='/maps/town 10'")
    setup = lines.index("source /venv/bin/activate")
    explicit = lines.index("export ONEPLANNER_X='set by launcher.env'")
    # The submitting shell's values, then launcher.setup, then launcher.env.
    assert snapshot < setup < explicit
    assert "unset DISPLAY WAYLAND_DISPLAY" in lines
    assert "cd '/work dir' || exit 1" in lines
    assert any(
        "/venv/bin/python -u -m autoware_carla_scenario.slurm.worker" in line
        and line.endswith(' 3 "$TASK" &')
        for line in lines
    )
    completed = subprocess.run(["bash", "-n"], input=script, text=True, check=False)
    assert completed.returncode == 0


def test_squeue_states_reads_task_lines_and_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.slurm import submit

    script = tmp_path / "squeue"
    script.write_text(
        "#!/bin/sh\n"
        'case "$*" in *" 7 "*) echo "7_0 RUNNING"; echo "7_[1-3,5%2] PENDING";;\n'
        '  *" 8 "*) ;;\n'
        '  *) echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1;;\n'
        "esac\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert submit.squeue_states("7") == {
        0: "RUNNING",
        1: "PENDING",
        2: "PENDING",
        3: "PENDING",
        5: "PENDING",
    }
    assert submit.squeue_states("8") == {}
    assert submit.squeue_states("9") == {}


def test_port_slots_are_exclusive_and_skip_ports_in_use(tmp_path: Path) -> None:
    base, slots = 31000, 20
    first = acquire_slot(base, slots, start=3, lock_dir=tmp_path)
    try:
        assert first.index == 3 and first.rpc == base + 30
        second = acquire_slot(base, slots, start=3, lock_dir=tmp_path)
        try:
            assert second.index == 4
            # A port someone else listens on keeps its slot out of use.
            listener = socket.socket()
            listener.bind(("127.0.0.1", base + 5 * PORT_SLOT_WIDTH + 3))
            listener.listen()
            try:
                third = acquire_slot(base, slots, start=5, lock_dir=tmp_path)
                assert third.index == 6
                third.release()
            finally:
                listener.close()
        finally:
            second.release()
        again = acquire_slot(base, slots, start=4, lock_dir=tmp_path)
        assert again.index == 4
        again.release()
    finally:
        first.release()


def test_first_slot_differs_by_task() -> None:
    env = {"SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "3"}
    assert first_slot(env, 1000) == 123
    assert first_slot({**env, "SLURM_ARRAY_TASK_ID": "4"}, 1000) == 124
    assert first_slot({"SLURM_JOB_ID": "1999"}, 1000) == 999


def test_sequential_stems_name_cases_as_a_local_batch() -> None:
    assert sequential_stems(["A", "A", "B", "A", "B"]) == [
        "A",
        "A-1",
        "B",
        "A-2",
        "B-1",
    ]


def test_environment_snapshot_takes_the_patterns_only() -> None:
    env = {"ONEPLANNER_MAP_DIR": "/m", "HF_TOKEN": "secret", "PATH": "/bin"}
    snapshot = snapshot_environment(SlurmSettings().env_passthrough, env)
    assert snapshot == {"ONEPLANNER_MAP_DIR": "/m", "PATH": "/bin"}


def test_queue_names_a_case_by_the_stem_it_is_given(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from autoware_carla_scenario import ScenarioQueue
    from autoware_carla_scenario.conditions import ScenarioResult

    seen: list[object] = []

    class _Runner:
        def run_scenario(self, scenario: object, **kwargs: object) -> ScenarioResult:
            seen.append(kwargs.get("file_stem"))
            return ScenarioResult(passed=True, message="ok", elapsed_seconds=0.0)

    queue = ScenarioQueue(server=MagicMock())
    queue._runner = _Runner()  # type: ignore[assignment]
    queue.add(MagicMock(), file_stem="CutInScenario-7")
    queue.add(MagicMock())
    queue.run_all(release=True)
    # Only when one is given: a runner that predates file_stem is not handed it.
    assert seen == ["CutInScenario-7", None]


def test_worker_points_the_config_at_its_ports() -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm.ports import PortSlot
    from autoware_carla_scenario.slurm.worker import worker_overrides

    cfg = run._compose_config(_CONCRETE, ["autoware.address=127.0.0.1:50052"])
    slot = PortSlot(index=0, base=20000)
    changes = worker_overrides(cfg, slot)
    assert changes == {
        "server.port": 20000,
        "traffic_manager.port": 20003,
        "autoware.address": "127.0.0.1:20004",
    }
    # The TrafficManager's port follows traffic_manager.port, as it interpolates.
    assert cfg.traffic.options.port == 20003
    # A driver the run does not host is dialled where it is.
    assert cfg.driver.address == "localhost:50051"


def test_job_script_guard_pattern_matches_only_its_port() -> None:
    pattern = re.compile(r"-carla-rpc-port=20000( |$)")
    assert pattern.search("CarlaUnreal -carla-rpc-port=20000 -RenderOffScreen")
    assert pattern.search("CarlaUnreal.sh -carla-rpc-port=20000")
    assert not pattern.search("CarlaUnreal -carla-rpc-port=200001")
