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
import signal
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


def _carla_records(record: Path, count: int) -> bool:
    """``count`` fake CARLAs have written a record that parses."""
    files = list(record.glob("carla-*.json"))
    if len(files) != count:
        return False
    try:
        for path in files:
            json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return True


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


def test_each_slurm_task_makes_the_recordings_a_local_batch_makes(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """rerun.enabled (#295): each task converts its own cases, and the batch
    reports them as the same local batch does."""
    pytest.importorskip("autoware_carla_scenario_rerun")
    local_status, local = _run_batch(
        fake_slurm, "local", [*_SWEEP, "rerun.enabled=true"]
    )
    local_out = capsys.readouterr().out
    slurm_status, batch = _run_batch(fake_slurm, "slurm", _slurm("rerun.enabled=true"))
    slurm_out = capsys.readouterr().out

    assert slurm_status == local_status
    recordings = sorted(p.name for p in local.glob("*.rrd"))
    assert recordings
    assert sorted(p.name for p in batch.glob("*.rrd")) == recordings
    assert "Rerun recordings:" in local_out
    assert _summary(slurm_out, batch) == _summary(local_out, local)
    assert (batch / "batch_results.json").read_text().replace(str(batch), "<b>") == (
        local / "batch_results.json"
    ).read_text().replace(str(local), "<b>")


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
    _wait_for(lambda: not layout.task_port(0, 0).exists(), 20, "the port file")

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
        lambda: _carla_records(fake_slurm.record, 2),
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
        # A worker that stops CARLA itself does so at once; one too slow for
        # the scheduler's grace period (5 s here) is killed, and the watchdog
        # then needs TERM, up to 5 s of grace, and KILL.  20 s covers both
        # with room for a loaded machine.
        _wait_for(lambda p=port: carla_processes_on_port(p) == [], 20, "CARLA")
    for task in range(2):
        task_status = layout.read_task(0, task)
        assert task_status is not None and task_status["state"] == "cancelled"
        # Removed after CARLA is gone, by the worker or (when it was killed)
        # its watchdog, so it can lag the CARLA check above (#525).
        _wait_for(
            lambda t=task: not layout.task_port(0, t).exists(), 20, "the port file"
        )

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


def test_a_worker_killed_before_it_stopped_carla_leaves_none_running(
    fake_slurm: SimpleNamespace,
) -> None:
    """The SIGKILL after the grace period reaches the worker, not its CARLA (a
    session of its own); a worker too slow to stop CARLA by then (#375) must
    not leave it running."""
    from autoware_carla_scenario.slurm import cli

    fake_slurm.monkeypatch.setenv("FAKE_CASE_SECONDS", "60")
    fake_slurm.monkeypatch.setenv("FAKE_SLURM_KILL_WAIT", "1")
    _status, batch = _run_batch(fake_slurm, "slurm", _slurm(), wait=False)
    records = fake_slurm.record
    _wait_for(lambda: _carla_records(records, 2), 90, "two CARLAs")
    workers = [int(p.stem.rpartition("-")[2]) for p in records.glob("carla-*.json")]
    ports = [json.loads(p.read_text())["port"] for p in records.glob("carla-*.json")]
    # Stopped, the workers cannot handle the SIGTERM before the SIGKILL.
    for worker in workers:
        os.kill(worker, signal.SIGSTOP)
    assert cli.main(["cancel", str(batch)]) == 0
    _wait_for(lambda: _jobs_done(fake_slurm), 30, "the cancelled jobs")
    for worker in workers:
        _wait_for(
            lambda w=worker: not Path(f"/proc/{w}").exists(), 30, "the worker's end"
        )
    for port in ports:
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


def test_a_kept_lock_survives_the_release_while_the_watchdog_holds_it(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.slurm.ports import CarlaWatchdog

    base = 31700
    slot = acquire_slot(base, 1, start=0, lock_dir=tmp_path)
    watchdog = CarlaWatchdog(slot)
    slot.release(keep_lock=True)
    with pytest.raises(RuntimeError):
        acquire_slot(base, 1, start=0, lock_dir=tmp_path)
    watchdog.finish()  # no CARLA runs: it ends, and with it the lock
    again = acquire_slot(base, 1, start=0, lock_dir=tmp_path)
    again.release()


def test_the_watchdog_removes_the_port_file_of_a_worker_that_died(
    tmp_path: Path,
) -> None:
    """A SIGKILLed worker never removes its ``.port`` file (#525): the watchdog,
    which sees the pipe close without ``done``, does."""
    from autoware_carla_scenario.slurm.ports import CarlaWatchdog

    slot = acquire_slot(31800, 1, start=0, lock_dir=tmp_path)
    port_file = tmp_path / "0-0.port"
    port_file.write_text(f"{slot.rpc}\n", encoding="utf-8")
    watchdog = CarlaWatchdog(slot, port_file)
    slot.release(keep_lock=True)
    process = watchdog._process
    assert process is not None and process.stdin is not None
    process.stdin.close()  # the worker died: end of file, no "done"
    process.wait(timeout=30)
    assert not port_file.exists()

    # A worker that finished cleans up itself: the watchdog leaves the file.
    slot = acquire_slot(31800, 1, start=0, lock_dir=tmp_path)
    port_file.write_text("x\n", encoding="utf-8")
    watchdog = CarlaWatchdog(slot, port_file)
    slot.release(keep_lock=True)
    watchdog.finish()
    assert port_file.exists()


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


@pytest.mark.parametrize(
    ("transport", "address"),
    [("direct", "localhost:50051"), ("unix", "unix:///shared/policy.sock")],
)
def test_worker_leaves_the_driver_transport_alone(transport: str, address: str) -> None:
    """direct and unix use no port of the slot: a policy the task serves is reached
    without one (in-process, or on a socket in a directory of the task's own), and
    a Unix socket someone else serves is dialled where it is, as a TCP one."""
    from autoware_carla_scenario.driver.base import DriverClientConfig
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm.ports import PortSlot
    from autoware_carla_scenario.slurm.worker import worker_overrides

    cfg = run._compose_config(
        _CONCRETE,
        [
            f"driver.transport={transport}",
            f"driver.address={address}",
            "driver.policy=constant_speed"
            if transport == "direct"
            else "driver.policy=null",
        ],
    )
    changes = worker_overrides(cfg, PortSlot(index=0, base=20000))
    assert not any(key.startswith("driver.") for key in changes)
    assert (cfg.driver.transport, cfg.driver.address) == (transport, address)
    config = DriverClientConfig.from_mapping(
        {
            k: v
            for k, v in run._to_dict(cfg.driver).items()
            if k not in ("control", "bridge")
        }
    )
    assert config.transport == transport


def test_job_script_guard_pattern_matches_only_its_port() -> None:
    pattern = re.compile(r"-carla-rpc-port=20000( |$)")
    assert pattern.search("CarlaUnreal -carla-rpc-port=20000 -RenderOffScreen")
    assert pattern.search("CarlaUnreal.sh -carla-rpc-port=20000")
    assert not pattern.search("CarlaUnreal -carla-rpc-port=200001")


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def _batch_on_disk(tmp_path: Path, cases: int = 3) -> BatchLayout:
    """A batch with a manifest and one submission (job 77), never run."""
    from autoware_carla_scenario.slurm.manifest import MANIFEST_VERSION, write_json

    layout = BatchLayout(tmp_path / "batch")
    write_json(
        layout.manifest,
        {
            "version": MANIFEST_VERSION,
            "cases": [{"index": i, "label": f"s#{i + 1}"} for i in range(cases)],
            "launcher": {},
        },
    )
    write_json(
        layout.submission(0),
        {"index": 0, "job_id": "77", "chunks": [list(range(cases))]},
    )
    return layout


def test_resubmit_refuses_when_squeue_cannot_tell_unless_forced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.slurm import cli

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: None)
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "squeue could not tell" in err and "77" in err and "--force" in err
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run", "--force"]) == 0
    assert "Resubmitting 3 case(s)" in capsys.readouterr().out
    # A job squeue lists as running is refused as before; one gone is not.
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {0: "RUNNING"})
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 1
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 0


def test_resubmit_runs_a_case_named_twice_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.slurm import cli

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    submitted: list[list[int]] = []

    def submit_cases(
        _layout: object, _manifest: object, _settings: object, chosen: list[int]
    ) -> int:
        submitted.append(chosen)
        return 1

    monkeypatch.setattr(cli, "submit_cases", submit_cases)
    monkeypatch.setattr(cli, "after_submit", lambda *_: 0)
    argv = ["resubmit", str(layout.batch_dir), "--cases", "2,s#3, 0,s#1,2"]
    assert cli.main(argv) == 0
    assert "Resubmitting 2 case(s): s#1, s#3" in capsys.readouterr().out
    assert submitted == [[0, 2]]


def test_port_locks_go_to_a_node_wide_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.slurm import ports

    shm = tmp_path / "shm"
    shm.mkdir()
    lock_dir = ports.default_lock_dir(shm)
    assert lock_dir == shm / f"acs-slurm-ports-{os.getuid()}"
    assert lock_dir.is_dir() and (lock_dir.stat().st_mode & 0o777) == 0o700
    # Where /dev/shm cannot be used, /tmp.
    assert ports.default_lock_dir(tmp_path / "missing") == ports.LOCK_DIR
    # A slot taken without a lock directory locks there.
    monkeypatch.setattr(ports, "default_lock_dir", lambda: lock_dir)
    slot = acquire_slot(31500, 5, start=0)
    try:
        assert (lock_dir / f"autoware_carla_scenario-port-{slot.base}.lock").exists()
    finally:
        slot.release()
    # launcher.port_lock_dir, and its default.
    assert SlurmSettings().port_lock_dir is None
    configured = SlurmSettings.from_mapping({"port_lock_dir": "/scratch/locks"})
    assert configured.port_lock_dir == "/scratch/locks"
    assert configured.replace(wait=False).port_lock_dir == "/scratch/locks"


def test_the_worker_takes_its_slot_in_launcher_port_lock_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.slurm import worker
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path, cases=1)
    manifest = layout.read_manifest()
    manifest["launcher"] = {"port_lock_dir": str(tmp_path / "locks")}
    manifest["cases"][0].update(scenario="x", overrides=[], stem="S")
    write_json(layout.manifest, manifest)
    seen: dict[str, Any] = {}

    def acquire(*_args: object, lock_dir: Path | None = None, **_kw: object) -> Any:
        seen["lock_dir"] = lock_dir
        raise RuntimeError("stop here")

    monkeypatch.setattr(worker, "acquire_slot", acquire)
    from autoware_carla_scenario.examples import run

    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: None)
    # run_task unsets these and takes over the signals: not this process's.
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    import signal

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        assert worker.run_task(layout.manifest, 0, 0) == 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert seen["lock_dir"] == tmp_path / "locks"
    status = layout.read_task(0, 0)
    assert status is not None and "stop here" in status["error"]


def test_secrets_are_not_recorded_through_a_pattern() -> None:
    env = {
        "AUTOWARE_API_TOKEN": "t",
        "AUTOWARE_Secret_Path": "s",
        "CARLA_LICENSE_KEY": "k",
        "ONEPLANNER_DB_PASSWORD": "p",
        "ONEPLANNER_CREDENTIALS_FILE": "c",
        "AUTOWARE_ROOT": "/aw",
        "ONEPLANNER_ENGINE_TOKEN_OK": "explicit",
    }
    patterns = [*SlurmSettings().env_passthrough, "ONEPLANNER_ENGINE_TOKEN_OK"]
    assert snapshot_environment(patterns, env) == {
        "AUTOWARE_ROOT": "/aw",
        # Named exactly on env_passthrough: recorded.
        "ONEPLANNER_ENGINE_TOKEN_OK": "explicit",
    }
    # Set in launcher.env: recorded too.
    assert snapshot_environment(
        SlurmSettings().env_passthrough, env, explicit=["CARLA_LICENSE_KEY"]
    ) == {"AUTOWARE_ROOT": "/aw", "CARLA_LICENSE_KEY": "k"}


def test_the_manifest_and_job_scripts_are_the_users_alone(
    fake_slurm: SimpleNamespace,
) -> None:
    fake_slurm.monkeypatch.setenv("AUTOWARE_API_TOKEN", "do-not-record")
    _status, batch = _run_batch(fake_slurm, "slurm", _slurm())
    layout = BatchLayout(batch)
    assert (layout.manifest.stat().st_mode & 0o777) == 0o600
    script = layout.job_script(0)
    assert (script.stat().st_mode & 0o777) == 0o700
    assert "AUTOWARE_API_TOKEN" not in layout.manifest.read_text()
    assert "do-not-record" not in script.read_text()
    assert "AUTOWARE_CARLA_SCENARIO_TYPECHECK" in layout.read_manifest()["env"]


def test_port_slots_must_be_at_least_one() -> None:
    for slots in (0, -3):
        with pytest.raises(ValueError, match="port_slots must be at least 1"):
            SlurmSettings.from_mapping({"port_slots": slots})
    assert SlurmSettings.from_mapping({"port_slots": 1}).port_slots == 1


def test_start_stagger_without_flock_uses_a_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import fcntl
    import threading

    from autoware_carla_scenario.slurm.worker import wait_for_start_turn

    def no_flock(*_args: object) -> None:
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(fcntl, "flock", no_flock)
    lock = BatchLayout(tmp_path).start_lock
    spacing = 0.3
    started: list[float] = []

    def start() -> None:
        wait_for_start_turn(lock, spacing)
        started.append(time.time())

    threads = [threading.Thread(target=start) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    started.sort()
    gaps = [b - a for a, b in zip(started, started[1:])]
    assert len(gaps) == 2 and all(gap >= spacing - 0.05 for gap in gaps), gaps
    excl = lock.with_name(lock.name + ".excl")
    assert not excl.exists()
    # A lock file a killed task left is stale after a while, and removed.
    excl.write_text("")
    old = time.time() - 3600
    os.utime(excl, (old, old))
    time.sleep(spacing)
    assert wait_for_start_turn(lock, spacing) == 0.0
    assert not excl.exists()

    # Another error of flock is not swallowed.
    def broken(*_args: object) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(fcntl, "flock", broken)
    with pytest.raises(OSError, match="I/O error"):
        wait_for_start_turn(lock, spacing)


def test_a_failing_collect_job_is_reported_not_raised(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm import submit

    real = submit._run_sbatch

    def sbatch(command: list[str]) -> str:
        if any(arg.startswith("--dependency=") for arg in command):
            raise RuntimeError("sbatch failed (1): QOSMaxSubmitJobPerUserLimit")
        return real(command)

    fake_slurm.monkeypatch.setattr(submit, "_run_sbatch", sbatch)
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(), wait=False)
    captured = capsys.readouterr()
    assert status == 1
    submission = BatchLayout(batch).read_submission(0)
    assert submission["job_id"] and submission["collect_job_id"] is None
    assert "QOSMaxSubmitJobPerUserLimit" in submission["collect_error"]
    assert f"SLURM job {submission['job_id']}" in captured.err
    assert "scenario-slurm wait" in captured.err
    assert "scenario-slurm collect" in captured.err
    assert "Traceback" not in captured.err
    _wait_for(lambda: _jobs_done(fake_slurm), 120, "the array")


def test_interrupt_waits_for_the_cancelled_tasks_before_collecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.slurm import collect, submit

    layout = _batch_on_disk(tmp_path)
    events: list[str] = []
    polls = iter(
        [
            {0: "RUNNING"},  # follow's first poll, then Ctrl-C
            {0: "RUNNING"},  # after scancel: still stopping
            None,  # squeue hiccups
            {0: "COMPLETING"},
            {},  # gone
        ]
    )

    def squeue(_job: str) -> dict[int, str] | None:
        state = next(polls)
        events.append(f"squeue {state}")
        return state

    sleeps = iter([KeyboardInterrupt()])

    def sleep(_seconds: float) -> None:
        interrupt = next(sleeps, None)
        if interrupt is not None:
            raise interrupt

    monkeypatch.setattr(submit, "squeue_states", squeue)
    monkeypatch.setattr(submit.time, "sleep", sleep)
    monkeypatch.setattr(submit, "cancel_jobs", lambda jobs: events.append("scancel"))

    def collected(_layout: object) -> int:
        events.append("collect")
        return 0

    monkeypatch.setattr(collect, "collect", collected)
    settings = SlurmSettings(poll_seconds=0.1)
    assert submit.follow(layout, 0, settings) == 0
    assert events == [
        "squeue {0: 'RUNNING'}",
        "scancel",
        "squeue {0: 'RUNNING'}",
        "squeue None",
        "squeue {0: 'COMPLETING'}",
        "squeue {}",
        "collect",
    ]


def test_interrupt_collects_after_the_bounded_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.slurm import submit

    clock = [0.0]
    monkeypatch.setattr(submit.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        submit.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)
    )
    monkeypatch.setattr(submit, "squeue_states", lambda _job: {0: "COMPLETING"})
    assert submit._wait_until_gone("77", 60.0, 15.0) is False
    assert 60.0 <= clock[0] <= 61.0
    assert "still in the queue" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# A submit host that sees the nodes' file system at other paths (remote_cwd)
# ---------------------------------------------------------------------------

_REMOTE_MANIFEST: dict[str, Any] = {
    "python": "/submit/host/.venv/bin/python",
    "cwd": "/home/alice/mnt/node/OnePlanner",
    "batch_dir_relative": "outputs/2026-10-19/12-00-00",
    "env": {},
}


def _remote_layout() -> BatchLayout:
    return BatchLayout(
        Path("/home/alice/mnt/node/OnePlanner/outputs/2026-10-19/12-00-00")
    )


def test_remote_cwd_defaults_pass_nothing_of_the_submit_host() -> None:
    plain = SlurmSettings.from_mapping({"export": None, "env_passthrough": None})
    assert plain.remote_cwd is None
    assert plain.export == "ALL"
    assert plain.env_passthrough == SlurmSettings().env_passthrough
    remote = SlurmSettings.from_mapping(
        {
            "remote_cwd": "workspace/OnePlanner",
            "python": ".venv/bin/python",
            "export": None,
            "env_passthrough": None,
        }
    )
    assert remote.export == "NONE"
    assert remote.env_passthrough == ()
    # Given explicitly, they are what they are given.
    explicit = SlurmSettings.from_mapping(
        {
            "remote_cwd": "workspace/OnePlanner",
            "python": "python3",
            "export": "ALL",
            "env_passthrough": ["ONEPLANNER_*"],
        }
    )
    assert explicit.export == "ALL"
    assert explicit.env_passthrough == ("ONEPLANNER_*",)
    # Recorded as resolved: a resubmit reads the same settings back.
    assert SlurmSettings.from_mapping(remote.to_dict()) == remote
    assert remote.replace(wait=False).remote_cwd == "workspace/OnePlanner"


@pytest.mark.parametrize(
    ("launcher", "error"),
    [
        ({"remote_cwd": "workspace/OnePlanner"}, "needs launcher.python"),
        ({"remote_cwd": "~bob/work", "python": "python3"}, "not ~user"),
        ({"remote_cwd": "work", "python": "~bob/python"}, "not ~user"),
        ({"remote_cwd": "a\nb", "python": "python3"}, "one line"),
    ],
)
def test_remote_cwd_settings_it_cannot_use_are_refused(
    launcher: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        SlurmSettings.from_mapping(launcher)


@pytest.mark.parametrize(
    ("remote_cwd", "cd"),
    [
        ("workspace/OnePlanner", 'cd "$HOME"/workspace/OnePlanner || '),
        ("~/workspace/OnePlanner", 'cd "$HOME"/workspace/OnePlanner || '),
        ("~", 'cd "$HOME" || '),
        ("/shared/work dir", "cd '/shared/work dir' || "),
    ],
)
@pytest.mark.parametrize(
    ("python", "run"),
    [
        (".venv/bin/python", ".venv/bin/python -u -m "),
        ("python3", "python3 -u -m "),
        ("~/venv/bin/python", '"$HOME"/venv/bin/python -u -m '),
        ("/opt/py/bin/python", "/opt/py/bin/python -u -m "),
    ],
)
def test_remote_job_script_works_at_the_node_s_paths(
    remote_cwd: str, cd: str, python: str, run: str
) -> None:
    from autoware_carla_scenario.slurm.submit import job_script, sbatch_command

    layout = _remote_layout()
    settings = SlurmSettings.from_mapping(
        {
            "remote_cwd": remote_cwd,
            "python": python,
            "setup": ["source ~/workspace/opt/env.sh"],
            "env": {"ONEPLANNER_X": "x"},
        }
    )
    script = job_script(layout, _REMOTE_MANIFEST, settings, 2)
    lines = script.splitlines()
    # Nothing of the submit host's paths: neither its cwd nor its interpreter.
    assert "/home/alice" not in script and "/submit/host" not in script
    workdir = next(i for i, line in enumerate(lines) if line.startswith(cd))
    batch = lines.index('BATCH_DIR="$WORKDIR"/outputs/2026-10-19/12-00-00')
    log = lines.index(
        'exec >>"$BATCH_DIR"/slurm/logs/"${SLURM_ARRAY_JOB_ID}_${TASK}.out" 2>&1'
    )
    setup = lines.index("source ~/workspace/opt/env.sh")
    # setup may read unset variables: the job has little of an environment.
    assert lines[setup - 1] == "set +u" and lines[setup + 1] == "set -u"
    back = lines.index('cd "$WORKDIR" || exit 1')
    worker = next(i for i, line in enumerate(lines) if line.startswith(run))
    assert workdir < batch < log < setup < back < worker
    assert lines[worker].endswith(' "$BATCH_DIR"/slurm/manifest.json 2 "$TASK" &')
    assert 'PORT_FILE="$BATCH_DIR"/slurm/tasks/2-${TASK}.port' in lines
    completed = subprocess.run(["bash", "-n"], input=script, text=True, check=False)
    assert completed.returncode == 0

    command = sbatch_command(layout, settings, layout.job_script(2), 3)
    assert "--chdir=/tmp" in command
    assert "--output=/tmp/acs-slurm-%u-%A_%a.out" in command
    assert "--export=NONE" in command
    # sbatch reads the script on the submit host: its path is this host's.
    assert command[-1] == str(layout.job_script(2))


def test_remote_collect_job_runs_the_node_s_python_where_the_batch_is() -> None:
    from autoware_carla_scenario.slurm.submit import _collect_command, collect_script

    layout = _remote_layout()
    settings = SlurmSettings.from_mapping(
        {
            "remote_cwd": "workspace/OnePlanner",
            "python": ".venv/bin/python",
            "setup": ["source ~/workspace/opt/env.sh"],
        }
    )
    script = collect_script(_REMOTE_MANIFEST, settings)
    lines = script.splitlines()
    assert "/home/alice" not in script and "/submit/host" not in script
    assert 'cd "$HOME"/workspace/OnePlanner || ' in "\n".join(lines)
    assert 'exec >>"$BATCH_DIR"/slurm/logs/"collect-${SLURM_JOB_ID}.out" 2>&1' in lines
    assert lines.index("source ~/workspace/opt/env.sh") < lines.index(
        "exec .venv/bin/python -m autoware_carla_scenario.slurm.cli collect "
        '"$BATCH_DIR"'
    )
    completed = subprocess.run(["bash", "-n"], input=script, text=True, check=False)
    assert completed.returncode == 0
    path = layout.root / "jobs" / "0-collect.sh"
    command = _collect_command(layout, _REMOTE_MANIFEST, settings, "41", path)
    assert "--dependency=afterany:41" in command
    assert "--chdir=/tmp" in command
    assert "--output=/tmp/acs-slurm-%u-collect-%j.out" in command
    assert not any(arg.startswith("--wrap") for arg in command)
    assert command[-1] == str(path)


def test_without_remote_cwd_the_collect_job_runs_launcher_python() -> None:
    from autoware_carla_scenario.slurm.submit import _collect_command

    layout = BatchLayout(Path("/work/outputs/2026-10-19/12-00-00"))
    manifest = {"python": "/usr/bin/python3", "cwd": "/work", "env": {}}
    settings = SlurmSettings.from_mapping({"python": ".venv/bin/python"})
    command = _collect_command(layout, manifest, settings, "41")
    assert "--chdir=/work" in command
    wrap = next(arg for arg in command if arg.startswith("--wrap="))
    assert wrap.startswith("--wrap=.venv/bin/python -m ")


def test_the_batch_must_lie_under_the_submitting_directory(tmp_path: Path) -> None:
    from autoware_carla_scenario.slurm.submit import relative_batch_dir

    work = tmp_path / "work"
    (work / "outputs" / "d" / "t").mkdir(parents=True)
    assert relative_batch_dir(work / "outputs" / "d" / "t", str(work)) == "outputs/d/t"
    assert relative_batch_dir(Path("outputs/d/t"), str(work)) == "outputs/d/t"
    with pytest.raises(ValueError, match="not under the directory scenario runs in"):
        relative_batch_dir(tmp_path / "elsewhere", str(work))
    with pytest.raises(ValueError, match="must be a directory under it"):
        relative_batch_dir(work, str(work))


def test_manifest_versions_one_and_two_are_read(tmp_path: Path) -> None:
    from autoware_carla_scenario.slurm.manifest import MANIFEST_VERSIONS, write_json

    layout = BatchLayout(tmp_path / "batch")
    for version in MANIFEST_VERSIONS:
        write_json(layout.manifest, {"version": version, "cases": []})
        assert layout.read_manifest()["version"] == version
    write_json(layout.manifest, {"version": 99, "cases": []})
    with pytest.raises(ValueError, match="version 99 manifest"):
        layout.read_manifest()


def test_a_remote_job_script_refuses_a_manifest_without_the_relative_batch() -> None:
    from autoware_carla_scenario.slurm.submit import job_script

    settings = SlurmSettings.from_mapping({"remote_cwd": "work", "python": "python3"})
    manifest = {k: v for k, v in _REMOTE_MANIFEST.items() if k != "batch_dir_relative"}
    with pytest.raises(ValueError, match="batch_dir_relative"):
        job_script(_remote_layout(), manifest, settings, 0)


def test_a_batch_submitted_from_a_mount_runs_at_the_node_s_paths(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The submit host sees the work directory at ``mnt/node/OnePlanner``;
    the node at ``workspace/OnePlanner`` in its user's home, which is not the
    submit host's: each job finds it through ``remote_cwd``, relative to the
    home the node's passwd database gives it."""
    from autoware_carla_scenario.slurm import cli, submit

    local_status, local = _run_batch(fake_slurm, "local", list(_SWEEP))
    tmp: Path = fake_slurm.tmp
    mp = fake_slurm.monkeypatch
    submit_side = tmp / "mnt" / "node" / "OnePlanner"
    submit_side.mkdir(parents=True)
    node_home = tmp / "node_home"
    (node_home / "workspace").mkdir(parents=True)
    (node_home / "workspace" / "OnePlanner").symlink_to(submit_side)
    # The node's interpreter, relative to the work directory.
    venv_python = submit_side / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    venv_python.chmod(0o755)
    # The node's passwd database: the job user's home is node_home.
    getent = tmp / "bin" / "getent"
    getent.write_text(
        f'#!/bin/sh\necho "user:x:$(id -u):$(id -g)::{node_home}:/bin/sh"\n'
    )
    getent.chmod(0o755)
    node_tmp = tmp / "node_tmp"
    node_tmp.mkdir()
    mp.setattr(submit, "REMOTE_EARLY_OUTPUT_DIR", str(node_tmp))
    mp.setenv("HOME", str(tmp / "submit_home"))
    mp.chdir(submit_side)
    output = submit_side / "outputs" / "2026-10-19" / "12-00-00"
    output.mkdir(parents=True)
    mp.setattr(fake_slurm.run, "_make_batch_output_dir", lambda: output)

    overrides = _slurm(
        "launcher.remote_cwd=workspace/OnePlanner",
        "launcher.python=.venv/bin/python",
        "launcher.setup=['echo node setup in $PWD']",
    )
    with pytest.raises(SystemExit) as exited:
        fake_slurm.run.run_batch([_CONCRETE], overrides, expand=True, wait=False)
    assert int(exited.value.code or 0) == 0
    _wait_for(lambda: _jobs_done(fake_slurm), 120, "the array and its collect job")

    layout = BatchLayout(output)
    manifest = layout.read_manifest()
    assert manifest["version"] == 2
    assert manifest["batch_dir_relative"] == "outputs/2026-10-19/12-00-00"
    assert manifest["launcher"]["remote_cwd"] == "workspace/OnePlanner"
    # Nothing of the submit host's environment is handed to the node.
    assert manifest["env"] == {}
    assert manifest["launcher"]["export"] == "NONE"
    for script in (layout.root / "jobs").glob("*.sh"):
        assert str(submit_side) not in script.read_text()
        assert str(tmp / "submit_home") not in script.read_text()
    for line in (fake_slurm.state / "sbatch.log").open():
        options = [a for a in json.loads(line)["argv"] if a.startswith("--")]
        assert not any(str(submit_side) in a for a in options)

    assert (output / "batch_results.json").read_bytes() == (
        local / "batch_results.json"
    ).read_bytes()
    logs = {p.name: p.read_text() for p in layout.logs.iterdir()}
    collect_log = next(t for n, t in logs.items() if n.startswith("collect-"))
    assert "Batch Scenario Results" in collect_log
    assert f"node setup in {node_home / 'workspace' / 'OnePlanner'}" in collect_log
    task_logs = [t for n, t in logs.items() if not n.startswith("collect-")]
    assert task_logs and all(
        f"node setup in {node_home / 'workspace' / 'OnePlanner'}" in t
        for t in task_logs
    )

    # The submit host follows, collects and resubmits at its own path.
    capsys.readouterr()
    assert cli.main(["status", str(output)]) == 0
    assert "6 with a result" in capsys.readouterr().out
    assert cli.main(["resubmit", str(output), "--cases", "0", "--wait"]) == local_status
    assert layout.submissions() == [0, 1]
    again = layout.job_script(1).read_text()
    assert 'cd "$HOME"/workspace/OnePlanner || ' in again
    assert str(submit_side) not in again
    record = layout.read_result(0)
    assert record is not None and record["submission"] == 1


def _case_with_file(path: Path, **config: Any) -> dict[str, Any]:
    from autoware_carla_scenario.slurm.manifest import config_files

    resolved = {"odd": str(path), "maps": [str(path), "x"], **config}
    return {"label": "s#1", "config": resolved, "files": config_files(resolved)}


def test_the_manifest_records_the_files_a_config_names(tmp_path: Path) -> None:
    from autoware_carla_scenario.slurm.manifest import config_files, file_sha256

    odd = tmp_path / "odd.yaml"
    odd.write_text("odd: 1\n")
    files = config_files(
        {"odd": str(odd), "a": {"b": [1, str(odd), str(tmp_path / "gone")]}, "c": "x"}
    )
    assert files == {"odd": file_sha256(odd), "a.b[1]": file_sha256(odd)}


def test_a_file_at_another_path_with_the_same_content_is_the_same_config(
    tmp_path: Path,
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.slurm.worker import _check_same_config

    there = tmp_path / "submit" / "odd.yaml"
    here = tmp_path / "node" / "site-packages" / "odd.yaml"
    for path in (there, here):
        path.parent.mkdir(parents=True)
        path.write_text("odd: 1\n")
    case = _case_with_file(there)
    _check_same_config(
        case, OmegaConf.create({"odd": str(here), "maps": [str(here), "x"]})
    )

    here.write_text("odd: 2\n")
    with pytest.raises(RuntimeError, match=r"at odd \(.*content differs\)"):
        _check_same_config(
            case, OmegaConf.create({"odd": str(here), "maps": [str(there), "x"]})
        )
    gone = tmp_path / "node" / "gone.yaml"
    with pytest.raises(RuntimeError) as refused:
        _check_same_config(
            case, OmegaConf.create({"odd": str(gone), "maps": [str(there), "x"]})
        )
    assert f"at odd ({there} there, {gone} here: missing on the node)" in str(
        refused.value
    )
    # A string that names no file is compared as it is.
    with pytest.raises(RuntimeError, match="at name"):
        _check_same_config(
            {**case, "config": {**case["config"], "name": "a"}},
            OmegaConf.create(
                {"odd": str(there), "maps": [str(there), "x"], "name": "b"}
            ),
        )


def test_the_submitting_directory_seen_from_the_job_s_is_the_same_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.slurm.worker import _check_same_config

    node = tmp_path / "node" / "OnePlanner"
    node.mkdir(parents=True)
    monkeypatch.chdir(node)
    monkeypatch.delenv("PWD", raising=False)
    case = {"label": "s#1", "config": {"out": "/mnt/node/OnePlanner/data/x.bin"}}
    _check_same_config(
        case, OmegaConf.create({"out": f"{node}/data/x.bin"}), "/mnt/node/OnePlanner"
    )
    with pytest.raises(RuntimeError, match="at out"):
        _check_same_config(
            case,
            OmegaConf.create({"out": f"{node}/data/y.bin"}),
            "/mnt/node/OnePlanner",
        )


def test_a_remote_batch_collected_on_a_node_names_the_submit_host_s_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm.collect import collect
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = BatchLayout(tmp_path / "node" / "outputs" / "d" / "t")
    write_json(
        layout.manifest,
        {
            "version": 2,
            "batch_dir": "/home/alice/mnt/node/OnePlanner/outputs/d/t",
            "batch_dir_relative": "outputs/d/t",
            "cases": [
                {"index": 0, "label": "s#1", "case_overrides": [], "overrides": []}
            ],
            "launcher": {},
        },
    )
    assert collect(layout) == 1
    out = capsys.readouterr().out
    assert "scenario-slurm resubmit /home/alice/mnt/node/OnePlanner/outputs/d/t" in out
    assert f"resubmit {layout.batch_dir}" not in out
