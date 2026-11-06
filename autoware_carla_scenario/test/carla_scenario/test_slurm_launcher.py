"""The SLURM launcher (``launcher=slurm``, docs/slurm.md), against a fake SLURM.

``data/fake_slurm/fake_slurm.py`` stands in for sbatch, squeue, scancel and
scontrol, running an array's tasks as local subprocesses, and
``data/fake_slurm/fake_world.py`` for CARLA and the scenarios -- in this process
for a local batch, and in every task through a ``sitecustomize`` on
``PYTHONPATH``.  A case's outcome depends only on its config, so a batch run
under SLURM must report exactly what the same batch run here reports.
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import shlex
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
    kill_carla_on_port,
)
from autoware_carla_scenario.slurm.settings import (
    SlurmSettings,
    launcher_requested_in_argv,
    launcher_type,
    snapshot_environment,
)

from ._slurm_ports import RANGE_SLOTS, fake_carlas, port_range

_DATA = Path(__file__).parent / "data" / "fake_slurm"
#: A scenario without a sweep of its own: drawing from the ODD needs no map.
_CONCRETE = "intersection_passing/straight_3npc"
_SWEEP = ["+sweep.odd_sample.count=6", "+sweep.odd_sample.seed=1"]

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
def port_base() -> Iterator[int]:
    """The first of :data:`RANGE_SLOTS` slots no other test uses meanwhile."""
    with port_range() as base:
        yield base


@pytest.fixture
def fake_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port_base: int
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
    monkeypatch.delenv("FAKE_SCENARIO_OVERRIDES", raising=False)

    monkeypatch.syspath_prepend(str(_DATA))
    import fake_world  # type: ignore[import-not-found]

    monkeypatch.setattr(run, "build_scenario", fake_world.fake_build)
    monkeypatch.setattr(run, "ScenarioQueue", fake_world.FakeQueue)
    monkeypatch.setattr(run, "resolve_map_paths", fake_world.fake_map_paths)
    monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)
    monkeypatch.setattr(run, "use_canvas", fake_world.fake_use_canvas)
    monkeypatch.setattr(
        run,
        "_compose_config",
        fake_world.compose_with_scenario_overrides(run._compose_config),
    )

    dirs: list[Path] = []

    def batch_dir(name: str) -> Path:
        path = tmp_path / "outputs" / name
        path.mkdir(parents=True)
        dirs.append(path)
        return path

    # port_base, a fixture this one depends on, holds the test's ports until
    # the clean-up below is done.
    yield SimpleNamespace(
        run=run,
        tmp=tmp_path,
        state=state,
        record=record,
        batch_dir=batch_dir,
        monkeypatch=monkeypatch,
        port_base=port_base,
    )
    # Nothing a test left behind keeps running: its tasks, their fake CARLAs.
    for job_dir in (state / "jobs").glob("*") if (state / "jobs").is_dir() else []:
        subprocess.run([str(bin_dir / "scancel"), job_dir.name], check=False)
    for carla in record.glob("carla-*.json"):
        # Only this test's fake ones: not another test's, nor a real CARLA
        # that may listen on 2000.
        for pid in fake_carlas(record, json.loads(carla.read_text())["port"]):
            try:
                os.kill(pid, 9)
            except OSError:
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


def _slurm(env: SimpleNamespace, *extra: str) -> list[str]:
    """A SLURM batch's overrides, on the test's own ports (:func:`port_range`)."""
    return [
        *_SWEEP,
        "launcher=slurm",
        "launcher.array_parallelism=2",
        "launcher.poll_seconds=0.2",
        f"launcher.port_base={env.port_base}",
        f"launcher.port_slots={RANGE_SLOTS}",
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
    slurm_status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm))
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
        assert (carla["port"] - fake_slurm.port_base) % PORT_SLOT_WIDTH == 0
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
        assert fake_carlas(fake_slurm.record, port) == []


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
    slurm_status, batch = _run_batch(
        fake_slurm, "slurm", _slurm(fake_slurm, "rerun.enabled=true")
    )
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


#: A second concrete scenario on the same map, for a batch whose two cases
#: are set differently (``FAKE_SCENARIO_OVERRIDES``).
_SECOND = "adversity/brake_check"


def _two_case_batch(
    env: SimpleNamespace, name: str, overrides: list[str]
) -> tuple[int, Path]:
    """``run_batch`` of :data:`_CONCRETE` then :data:`_SECOND`; (exit status,
    directory)."""
    output = env.batch_dir(name)
    env.monkeypatch.setattr(env.run, "_make_batch_output_dir", lambda: output)
    with pytest.raises(SystemExit) as exited:
        env.run.run_batch([_CONCRETE, _SECOND], overrides)
    return int(exited.value.code or 0), output


def _queues(env: SimpleNamespace) -> list[dict[str, Any]]:
    """The settings of each queue built since the last call, in order."""
    path = env.record / "queues.jsonl"
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.open()]
    path.unlink()
    return [entry["settings"] for entry in entries]


@pytest.mark.parametrize(
    ("setting", "second_case"),
    [
        ("t4scene", ["++t4scene.enabled=true", "++t4scene.step_s=0.2"]),
        ("rerun", ["++rerun.enabled=true", "++rerun.strict=true"]),
        ("canvas", ["++canvas.config=/canvas.yaml"]),
        ("heavy_metrics", ["metrics.heavy=false"]),
    ],
)
def test_a_batch_s_settings_do_not_depend_on_how_its_cases_are_split_into_tasks(
    fake_slurm: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    setting: str,
    second_case: list[str],
) -> None:
    """#1225: only the batch's second case sets *setting*.  A local batch
    decides its queue (``t4scene``: the first case's), its Rerun converter
    (any case's), its canvas (the one any case names) and its heavy metrics
    (the first case setting ``metrics.heavy``) once, for every case; each
    SLURM task decides the same, whether it runs both cases or one."""
    if setting == "rerun":
        pytest.importorskip("autoware_carla_scenario_rerun")
    # The heavy metrics on unless a case turns them off.
    fake_slurm.monkeypatch.delenv(
        "AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS", raising=False
    )
    fake_slurm.monkeypatch.setenv(
        "FAKE_SCENARIO_OVERRIDES", json.dumps({_SECOND: second_case})
    )
    local_status, local = _two_case_batch(fake_slurm, "local", [])
    local_out = capsys.readouterr().out
    (decided,) = _queues(fake_slurm)
    if setting == "t4scene":
        assert decided["t4scene"] is None  # the first case's, which has none
    elif setting == "canvas":
        assert decided["canvas"] == "/canvas.yaml"
    elif setting == "heavy_metrics":
        assert decided["heavy_metrics"] is False
    else:
        assert decided["rerun"] == {
            "strict": True,
            "lidar_max_points": None,
            "cameras": False,
        }

    for per_task, tasks in ((1, 2), (2, 1)):
        name = f"slurm-{per_task}"
        status, batch = _two_case_batch(
            fake_slurm,
            name,
            [*_slurm(fake_slurm)[len(_SWEEP) :], f"launcher.cases_per_task={per_task}"],
        )
        out = capsys.readouterr().out
        assert BatchLayout(batch).read_submission(0)["chunks"] == (
            [[0], [1]] if per_task == 1 else [[0, 1]]
        )
        assert _queues(fake_slurm) == [decided] * tasks, name
        assert status == local_status, name
        assert sorted(_outputs(batch)) == sorted(_outputs(local)), name
        assert (batch / "batch_results.json").read_text().replace(
            str(batch), "<b>"
        ) == (local / "batch_results.json").read_text().replace(str(local), "<b>")
        assert _summary(out, batch) == _summary(local_out, local), name
        record = BatchLayout(batch).read_manifest()["batch_settings"]
        assert record["first_case"] == 0
        assert record["rerun_cases"] == ([0, 1] if setting == "rerun" else [])
        assert record["canvas_case"] == (1 if setting == "canvas" else None)
        assert record["heavy_metrics"] is (setting != "heavy_metrics")


def test_a_task_of_a_batch_without_the_record_decides_from_its_own_cases(
    fake_slurm: SimpleNamespace,
) -> None:
    """A manifest written before ``batch_settings`` (#1225): each task decides
    the batch-level settings from its own cases, as tasks did then."""
    fake_slurm.monkeypatch.setenv(
        "FAKE_SCENARIO_OVERRIDES",
        json.dumps({_SECOND: ["++t4scene.enabled=true", "++t4scene.step_s=0.2"]}),
    )
    fake_slurm.monkeypatch.setattr(
        fake_slurm.run, "batch_settings_cases", lambda *_a, **_k: None
    )
    _status, batch = _two_case_batch(
        fake_slurm,
        "slurm-old",
        [*_slurm(fake_slurm)[len(_SWEEP) :], "launcher.cases_per_task=1"],
    )
    assert "batch_settings" not in BatchLayout(batch).read_manifest()
    queues = sorted(
        _queues(fake_slurm), key=lambda settings: settings["t4scene"] is not None
    )
    assert [settings["t4scene"] for settings in queues] == [
        None,
        {"step_s": 0.2, "max_distance_m": 100.0},
    ]


def test_batch_settings_cases_name_a_case_for_each_setting(
    fake_slurm: SimpleNamespace,
) -> None:
    """The cases recorded decide what every case decides: one per distinct
    ``rerun`` block, the first naming a canvas."""
    pytest.importorskip("autoware_carla_scenario_rerun")
    # rerun.lidar records the sensors with the Rerun SDK itself.
    pytest.importorskip("rerun")
    run = fake_slurm.run
    lidar = ["rerun.enabled=true", "rerun.lidar=true", "rerun.lidar_max_points=7"]
    canvas = ["++canvas.config=/canvas.yaml"]
    off = run._compose_config(_CONCRETE, [])
    configs = [
        off,
        run._compose_config(_CONCRETE, lidar),
        off,
        run._compose_config(_CONCRETE, ["rerun.strict=true", *canvas]),
        run._compose_config(_CONCRETE, [*lidar, *canvas]),
    ]
    record = run.batch_settings_cases(configs, heavy_metrics=False)
    assert record == {
        "first_case": 0,
        "rerun_cases": [0, 1, 3],
        "canvas_case": 3,
        "heavy_metrics": False,
    }
    chosen = run.build_rerun_converter(*(configs[i] for i in record["rerun_cases"]))
    every = run.build_rerun_converter(*configs)
    assert (chosen._strict, chosen._lidar_max_points, chosen._cameras) == (
        every._strict,
        every._lidar_max_points,
        every._cameras,
    )
    assert run.batch_settings_cases([off, off], heavy_metrics=True) == {
        "first_case": 0,
        "rerun_cases": [],
        "canvas_case": None,
        "heavy_metrics": True,
    }


def test_the_manifest_rebuilds_each_case_exactly(
    fake_slurm: SimpleNamespace,
) -> None:
    from omegaconf import OmegaConf

    _status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm))
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

    status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm))
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
    _wait_for(
        lambda: fake_carlas(fake_slurm.record, port) == [], 20, "the orphaned CARLA"
    )
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
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm), wait=False)
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
    assert all(fake_carlas(fake_slurm.record, port) for port in ports)
    # The array alone, as a user's own scancel would: its collect job runs.
    subprocess.run(["scancel", str(submission["job_id"])], check=True)
    _wait_for(lambda: _jobs_done(fake_slurm), 30, "the cancelled tasks")
    for port in ports:
        # A worker that stops CARLA itself does so at once; one too slow for
        # the scheduler's grace period (5 s here) is killed, and the watchdog
        # then needs TERM, up to 5 s of grace, and KILL.  20 s covers both
        # with room for a loaded machine.
        _wait_for(lambda p=port: fake_carlas(fake_slurm.record, p) == [], 20, "CARLA")
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
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm), wait=False)
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
        _wait_for(lambda p=port: fake_carlas(fake_slurm.record, p) == [], 20, "CARLA")


def test_a_worker_killed_before_it_stopped_carla_leaves_none_running(
    fake_slurm: SimpleNamespace,
) -> None:
    """The SIGKILL after the grace period reaches the worker, not its CARLA (a
    session of its own); a worker too slow to stop CARLA by then (#375) must
    not leave it running."""
    from autoware_carla_scenario.slurm import cli

    fake_slurm.monkeypatch.setenv("FAKE_CASE_SECONDS", "60")
    fake_slurm.monkeypatch.setenv("FAKE_SLURM_KILL_WAIT", "1")
    _status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm), wait=False)
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
        _wait_for(lambda p=port: fake_carlas(fake_slurm.record, p) == [], 20, "CARLA")


def test_no_wait_submits_and_the_collect_job_writes_the_results(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.slurm import cli

    local_status, local = _run_batch(fake_slurm, "local", list(_SWEEP))
    status, batch = _run_batch(
        fake_slurm, "slurm", _slurm(fake_slurm, "launcher.cases_per_task=1"), wait=False
    )
    assert status == 0
    layout = BatchLayout(batch)
    submission = layout.read_submission(0)
    assert len(submission["chunks"]) == 6
    # The collect job runs a script beside the tasks' (#1159).
    collect_script = layout.root / "jobs" / "0-collect.sh"
    assert submission["collect_sbatch"][-1] == str(collect_script)
    assert collect_script.read_text().startswith("#!/bin/bash\n")
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
        fake_slurm.run.run_batch([_CONCRETE], _slurm(fake_slurm), expand=True)
    assert exited.value.code == 2
    assert "sbatch" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ("scenario.timeout_seconds=nan", "scenario.timeout_seconds"),
        ("scenario.timeout_seconds=inf", "scenario.timeout_seconds"),
        ("scenario.timeout_seconds=0", "scenario.timeout_seconds"),
        # #1188: not a number at all.
        ("scenario.timeout_seconds=fast", "scenario.timeout_seconds"),
        ("scenario.timeout_seconds=true", "scenario.timeout_seconds"),
        # The local queue's own settings, which a task reads only on its node.
        ("server.cooldown_max_retries=-1", "server.cooldown_max_retries"),
    ],
)
def test_slurm_refuses_what_a_local_batch_refuses_before_submitting(
    fake_slurm: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
    key: str,
) -> None:
    """#977: the whole batch is refused up front, naming the key."""
    from autoware_carla_scenario.slurm import submit

    def submitted(**_: object) -> int:
        raise AssertionError("submitted the batch")

    def output_dir() -> Path:
        raise AssertionError("made the batch's output directory")

    monkeypatch.setattr(submit, "submit_batch", submitted)
    monkeypatch.setattr(fake_slurm.run, "_make_batch_output_dir", output_dir)
    with pytest.raises(ValueError, match=re.escape(key)):
        fake_slurm.run.run_batch([_CONCRETE], ["launcher=slurm", override])
    # A local batch refuses the same value, before its output directory is
    # made too (#1430): run_batch raises it, and the CLI prints its Error line.
    with pytest.raises(fake_slurm.run.RunConfigError, match=re.escape(key)):
        fake_slurm.run.run_batch([_CONCRETE], [override])
    assert not (fake_slurm.state / "sbatch.log").exists()


def test_a_null_timeout_is_refused_by_the_config_not_the_run_timeout(
    fake_slurm: SimpleNamespace,
) -> None:
    """#1188: ``scenario.timeout_seconds=null`` (how a declarative scenario falls
    back to its document's timeout) is the default run timeout, not a
    TypeError.  A scenario whose config needs a number refuses it, naming the
    key, before the batch is submitted (#1818): every case would otherwise
    die in its builder with an error naming no key."""
    from autoware_carla_scenario.run_timeout import (
        DEFAULT_RUN_TIMEOUT_SECONDS,
        run_timeout_seconds,
    )

    run = fake_slurm.run
    cfg = run._compose_config(_CONCRETE, ["scenario.timeout_seconds=null"])
    assert cfg.scenario.timeout_seconds is None
    assert run_timeout_seconds(cfg) == DEFAULT_RUN_TIMEOUT_SECONDS
    with pytest.raises(
        run.RunConfigError,
        match=r"^scenario\.timeout_seconds must be a number, got None$",
    ):
        run.check_batch_configs([cfg])


def test_the_check_reads_the_t4scene_settings_as_the_batch_does(
    fake_slurm: SimpleNamespace,
) -> None:
    """#1233: the batch's T4Scenes are written with the first case's settings
    that enables them (build_t4scene_exporter, the queue's first case), so a
    later case's are not refused under SLURM when a local batch runs them."""
    run = fake_slurm.run
    good = run._compose_config(
        _CONCRETE, ["t4scene.enabled=true", "t4scene.step_s=0.1"]
    )
    bad = run._compose_config(
        _CONCRETE, ["t4scene.enabled=true", "t4scene.step_s=0.07"]
    )
    disabled = run._compose_config(_CONCRETE, [])
    run.check_batch_configs([good, bad])
    for configs in ([bad], [disabled, bad], [bad, good]):
        with pytest.raises(run.RunConfigError, match="t4scene.step_s"):
            run.check_batch_configs(configs)


def test_slurm_checks_every_case_not_only_the_first(
    fake_slurm: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task builds its queue from the first case of its own chunk."""
    from autoware_carla_scenario.slurm import submit

    compose = fake_slurm.run._compose_config

    def composed(name: str, overrides: list[str]) -> Any:
        cfg = compose(name, overrides)
        if calls:
            cfg.server.cooldown_max_retries = -1
        calls.append(name)
        return cfg

    calls: list[str] = []
    monkeypatch.setattr(fake_slurm.run, "_compose_config", composed)
    monkeypatch.setattr(
        submit, "submit_batch", lambda **_: pytest.fail("submitted the batch")
    )
    with pytest.raises(ValueError, match="server.cooldown_max_retries"):
        fake_slurm.run.run_batch([_CONCRETE, _CONCRETE], ["launcher=slurm"])
    assert len(calls) == 2


def test_slurm_checks_the_launcher_of_an_autoware_case_after_an_autopilot_one(
    fake_slurm: SimpleNamespace,
) -> None:
    """The ego's entity decides whether the Autoware launcher is read at all, so
    a case that differs from the one before only in it is checked again."""
    run = fake_slurm.run
    first = run._compose_config(_CONCRETE, ["autoware.launcher.type=bogus"])
    second = run._compose_config(_CONCRETE, ["autoware.launcher.type=bogus"])
    second.ego.entity = "autoware"
    run.check_batch_configs([first])
    with pytest.raises(ValueError, match="autoware.launcher.type"):
        run.check_batch_configs([first, second])


@pytest.mark.parametrize(
    ("kind", "override", "key"),
    [
        ("docker", "autoware.launcher.stop_timeout_s=nan", "stop_timeout_s"),
        ("command", "autoware.launcher.stop_timeout_s=nan", "stop_timeout_s"),
        ("docker", "autoware.launcher.stop_timeout_s=-5", "stop_timeout_s"),
        ("command", "autoware.launcher.stop_timeout_s=abc", "stop_timeout_s"),
        ("docker", "autoware.launcher.ros_domain_ids=[-3,999]", "ros_domain_ids"),
        ("docker", "autoware.launcher.ros_domain_ids=7", "ros_domain_ids"),
    ],
)
def test_a_batch_with_a_bad_launcher_setting_is_refused_up_front(
    fake_slurm: SimpleNamespace, kind: str, override: str, key: str
) -> None:
    """#1450: the launcher's stop timeout and ROS domain ids are refused with
    the batch, naming the key, not when a stack is stopped or started."""
    run = fake_slurm.run
    overrides = [
        "ego.entity=autoware",
        f"autoware.launcher.type={kind}",
        "autoware.launcher.launch=[autoware_launch,x.launch.xml]",
        "autoware.launcher.command=[ros2,launch]",
        override,
    ]
    with pytest.raises(
        run.RunConfigError, match=re.escape(f"autoware.launcher.{key} must be")
    ):
        run.check_batch_configs([run._compose_config(_CONCRETE, overrides)])


@pytest.mark.parametrize("kind", ["docker", "command"])
def test_a_batch_with_an_unknown_launcher_key_is_refused_before_anything_runs(
    fake_slurm: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """#1413: a typo under autoware.launcher refuses the whole batch up front,
    under SLURM and locally, naming the key."""
    from autoware_carla_scenario.slurm import submit

    run = fake_slurm.run
    overrides = [
        "ego.entity=autoware",
        f"autoware.launcher.type={kind}",
        "+autoware.launcher.stop_timout_s=60",
    ]
    with pytest.raises(
        run.RunConfigError, match=re.escape("autoware.launcher.stop_timout_s")
    ):
        run.check_batch_configs([run._compose_config(_CONCRETE, overrides)])

    monkeypatch.setattr(
        submit, "submit_batch", lambda **_: pytest.fail("submitted the batch")
    )
    monkeypatch.setattr(
        run, "_make_batch_output_dir", lambda: pytest.fail("made the output dir")
    )
    with pytest.raises(ValueError, match=re.escape("autoware.launcher.stop_timout_s")):
        run.run_batch([_CONCRETE], ["launcher=slurm", *overrides])
    local = fake_slurm.batch_dir("local")
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: local)
    with pytest.raises(ValueError, match=re.escape("autoware.launcher.stop_timout_s")):
        run.run_batch([_CONCRETE], overrides)
    assert not (fake_slurm.state / "sbatch.log").exists()


#: One ``map`` setting a batch's second case sets differently: each decides
#: the map its run is on (#1241).
_MAP_SETTINGS = [
    pytest.param("name", "map.name=OtherTown", id="name"),
    pytest.param("source", "++map.source='git+https://x.invalid/m.git'", id="source"),
    pytest.param("directory", "++map.directory=/elsewhere/map", id="directory"),
    pytest.param("lanelet2_path", "++map.lanelet2_path=/other.osm", id="lanelet2"),
    pytest.param("xodr_path", "++map.xodr_path=/other.xodr", id="xodr"),
    pytest.param("projector_type", "++map.projector_type=utm", id="projector"),
    pytest.param("overwrite_xodr", "++map.overwrite_xodr=false", id="overwrite"),
]


@pytest.mark.parametrize(("key", "override"), _MAP_SETTINGS)
def test_the_check_refuses_cases_on_different_maps(
    fake_slurm: SimpleNamespace, key: str, override: str
) -> None:
    """#1241: the batch's queue is built on its first case's map, so a case
    resolving to another -- the same ``map.name`` with another source or file
    included -- is refused, naming the key and the cases."""
    run = fake_slurm.run
    first = run._compose_config(_CONCRETE, [])
    second = run._compose_config(_SECOND, [override])
    run.check_batch_configs([first, first])
    with pytest.raises(run.RunConfigError, match=rf"map\.{key} differs") as refused:
        run.check_batch_configs([first, second], ["one", "two"])
    assert "in one" in str(refused.value)
    assert "in two" in str(refused.value)


def test_the_check_passes_cases_with_the_same_map_set_alike(
    fake_slurm: SimpleNamespace,
) -> None:
    """The same settings, however each case came by them, are one map; what a
    sweep reads per case (``no_3d_model_lanelet_ids``) is not compared."""
    run = fake_slurm.run
    same = ["++map.lanelet2_path=/m.osm", "++map.projector_type=utm"]
    first = run._compose_config(_CONCRETE, [*same, "++map.directory="])
    second = run._compose_config(
        _SECOND, [*same, "++map.no_3d_model_lanelet_ids=[1,2]"]
    )
    run.check_batch_configs([first, second])


@pytest.mark.parametrize(("key", "override"), _MAP_SETTINGS)
@pytest.mark.parametrize("launcher", ["local", "slurm"])
def test_a_batch_on_two_maps_is_refused_before_anything_runs(
    fake_slurm: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    key: str,
    override: str,
    launcher: str,
) -> None:
    """#1241: a batch, local or SLURM, whose second case is on another map is
    refused before its queue is built or anything is submitted."""
    fake_slurm.monkeypatch.setenv(
        "FAKE_SCENARIO_OVERRIDES", json.dumps({_SECOND: [override]})
    )
    overrides = [] if launcher == "local" else _slurm(fake_slurm)[len(_SWEEP) :]
    status, _batch = _two_case_batch(fake_slurm, launcher, overrides)
    assert status == 1
    err = capsys.readouterr().err
    assert f"map.{key} differs" in err
    assert _SECOND in err
    assert _queues(fake_slurm) == []
    assert not (fake_slurm.state / "sbatch.log").exists()


def test_the_worker_refuses_a_timeout_that_is_not_positive_and_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#977: a case submitted with a NaN timeout fails its task, naming the key,
    before anything is built."""
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm import worker
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path, cases=1)
    manifest = layout.read_manifest()
    manifest["cases"][0].update(
        scenario=_CONCRETE,
        overrides=["scenario.timeout_seconds=nan"],
        stem="S",
        scenario_class="X",
    )
    write_json(layout.manifest, manifest)
    released: list[bool] = []
    slot = SimpleNamespace(
        rpc=31500,
        to_dict=dict,
        release=lambda keep_lock=False: released.append(keep_lock),
    )

    def no_watchdog(*_args: object) -> None:
        raise OSError("no watchdog here")

    def built(*_args: object, **_kw: object) -> None:
        raise AssertionError("built the scenario")

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    monkeypatch.setattr(worker, "acquire_slot", lambda *_a, **_k: slot)
    monkeypatch.setattr(worker, "CarlaWatchdog", no_watchdog)
    monkeypatch.setattr(
        worker, "worker_overrides", lambda _cfg, _slot, _domains=None: {}
    )
    monkeypatch.setattr(worker, "kill_carla_on_port", lambda _port: [])
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: None)
    monkeypatch.setattr(run, "build_scenario", built)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        assert worker.run_task(layout.manifest, 0, 0) == 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    status = layout.read_task(0, 0)
    assert status is not None and status["state"] == "failed"
    assert "ValueError: scenario.timeout_seconds" in status["error"]
    assert "nan" in status["error"]
    assert released == [False]


def test_the_worker_refuses_a_case_off_its_documents_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1424: a case whose scenario's document names a map of its own, composed
    on another map group, fails its task naming ``map=<id>_map``, before
    anything is built -- for a manifest the submitting process did not check."""
    from autoware_carla_scenario import registry
    from autoware_carla_scenario.authoring.starter import new_document
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm import worker
    from autoware_carla_scenario.slurm.manifest import write_json

    document = new_document()
    document.map.source = (
        "git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps"
        "@5058c6683a1077a99aa5fdca5e6ecff9d71fcfb1#autoware_maps/Town10HD_Opt"
    )
    document.map.name = "Town10HD_Opt"
    name = str(run._compose_config(_CONCRETE, []).scenario.name)
    monkeypatch.setitem(registry._SCENARIO_DOCUMENTS, name, lambda: document)

    layout = _batch_on_disk(tmp_path, cases=1)
    manifest = layout.read_manifest()
    manifest["cases"][0].update(
        scenario=_CONCRETE, overrides=[], stem="S", scenario_class="X"
    )
    write_json(layout.manifest, manifest)
    released: list[bool] = []
    slot = SimpleNamespace(
        rpc=31500,
        to_dict=dict,
        release=lambda keep_lock=False: released.append(keep_lock),
    )

    def no_watchdog(*_args: object) -> None:
        raise OSError("no watchdog here")

    def built(*_args: object, **_kw: object) -> None:
        raise AssertionError("built the scenario")

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    monkeypatch.setattr(worker, "acquire_slot", lambda *_a, **_k: slot)
    monkeypatch.setattr(worker, "CarlaWatchdog", no_watchdog)
    monkeypatch.setattr(
        worker, "worker_overrides", lambda _cfg, _slot, _domains=None: {}
    )
    monkeypatch.setattr(worker, "kill_carla_on_port", lambda _port: [])
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: None)
    monkeypatch.setattr(run, "build_scenario", built)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        assert worker.run_task(layout.manifest, 0, 0) == 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    status = layout.read_task(0, 0)
    assert status is not None and status["state"] == "failed"
    assert "RunConfigError: s#1: " in status["error"]
    assert "map=my_cut_in_map" in status["error"]
    assert released == [False]


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
    assert "--no-wait is for launcher=slurm only" in capsys.readouterr().err


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


@pytest.mark.parametrize("key", ["start_stagger_seconds", "poll_seconds"])
@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e400", -1, "-0.5"])
def test_settings_refuse_seconds_that_are_not_finite_or_negative(
    key: str, value: Any
) -> None:
    # inf made a task's start wait overflow (and the manifest's JSON hold
    # Infinity), nan read as 0, a negative stagger as 0 (#784).
    with pytest.raises(ValueError, match=rf"launcher\.{key} must be a finite"):
        SlurmSettings.from_mapping({key: value})


def test_settings_read_the_seconds_they_take() -> None:
    settings = SlurmSettings.from_mapping(
        {"start_stagger_seconds": "2.5", "poll_seconds": 3}
    )
    assert (settings.start_stagger_seconds, settings.poll_seconds) == (2.5, 3.0)
    # null: no stagger; a poll of 0 would spin, so it is 0.1 s at least.
    settings = SlurmSettings.from_mapping(
        {"start_stagger_seconds": None, "poll_seconds": 0}
    )
    assert (settings.start_stagger_seconds, settings.poll_seconds) == (0.0, 0.1)
    assert SlurmSettings.from_mapping(settings.to_dict()) == settings


@pytest.mark.parametrize(
    "key", ["cpus_per_task", "array_parallelism", "cases_per_task"]
)
@pytest.mark.parametrize(
    "value", [2.7, True, False, "2.5", "two", "", float("nan"), float("inf"), [2]]
)
def test_settings_refuse_counts_that_are_not_whole_numbers(
    key: str, value: Any
) -> None:
    # int() read 2.7 as 2 and true as 1 (#1485).
    with pytest.raises(ValueError, match=rf"launcher\.{key} must be"):
        SlurmSettings.from_mapping({key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("signal_grace_seconds", -30),
        ("signal_grace_seconds", 1.5),
        ("signal_grace_seconds", True),
        ("signal_grace_seconds", None),
        ("port_base", 20000.5),
        ("port_base", True),
        ("port_base", 80),
        ("port_base", "x"),
        ("port_slots", 2.5),
        ("port_slots", True),
        ("port_slots", 0),
    ],
)
def test_settings_refuse_grace_and_ports_out_of_range(key: str, value: Any) -> None:
    # A negative grace became --signal=B:TERM@-30, which only sbatch refused,
    # after the batch was written (#1485).
    with pytest.raises(ValueError, match=rf"launcher\.{key} must be"):
        SlurmSettings.from_mapping({key: value})


def test_settings_read_whole_numbers_as_ints() -> None:
    settings = SlurmSettings.from_mapping(
        {
            "cpus_per_task": 4.0,
            "array_parallelism": "3",
            "cases_per_task": 2,
            "signal_grace_seconds": 0,
            "port_base": 3e4,
            "port_slots": "10",
        }
    )
    assert (
        settings.cpus_per_task,
        settings.array_parallelism,
        settings.cases_per_task,
        settings.signal_grace_seconds,
        settings.port_base,
        settings.port_slots,
    ) == (4, 3, 2, 0, 30000, 10)
    assert all(
        type(v) is int
        for v in (settings.cpus_per_task, settings.port_base, settings.port_slots)
    )
    assert SlurmSettings.from_mapping({"cases_per_task": None}).cases_per_task is None


@pytest.mark.parametrize(
    "key", ["wait", "headless", "collect_job", "cancel_on_interrupt"]
)
def test_settings_read_flags_strictly(key: str) -> None:
    # bool() read "false" as true (#1485).
    for value in (False, "false", "False", " no ", "off", "0", 0):
        assert getattr(SlurmSettings.from_mapping({key: value}), key) is False
    for value in (True, "true", "yes", "on", "1", 1):
        assert getattr(SlurmSettings.from_mapping({key: value}), key) is True
    for bad in ("ture", "", None, 2, 0.0, [True]):
        with pytest.raises(ValueError, match=rf"launcher\.{key} must be true or false"):
            SlurmSettings.from_mapping({key: bad})


@pytest.mark.parametrize("key", ["start_stagger_seconds", "poll_seconds"])
@pytest.mark.parametrize("value", [True, False, [1], "soon"])
def test_settings_refuse_seconds_that_are_not_numbers(key: str, value: Any) -> None:
    # float(True) read a flag as 1 s (#1485).
    with pytest.raises(ValueError, match=rf"launcher\.{key} must be a finite"):
        SlurmSettings.from_mapping({key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("partition", True),
        ("mem", ["12G"]),
        ("export", {"ALL": 1}),
        ("extra_sbatch_args", 5),
        ("setup", {"a": "b"}),
        ("env", ["A=1"]),
    ],
)
def test_settings_refuse_text_and_lists_of_another_type(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"launcher\.{key} must be"):
        SlurmSettings.from_mapping({key: value})


def test_shipped_launcher_config_is_read() -> None:
    import yaml

    import autoware_carla_scenario.examples as examples

    path = Path(examples.__file__).parent / "conf" / "launcher" / "slurm.yaml"
    assert SlurmSettings.from_mapping(yaml.safe_load(path.read_text())) == (
        SlurmSettings()
    )


class _FrozenClock:
    """``worker.time`` with ``time()`` fixed at *now* and ``sleep`` recorded."""

    def __init__(self, now: float) -> None:
        self.now = now
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        ("1000.0", 0.0),  # long ago
        ("1998.0", 3.0),  # 2 s ago: the rest of the spacing
        ("2000.0", 5.0),  # just now: the whole spacing
        ("2600.0", 5.0),  # a node whose clock runs 600 s ahead: no more
        ("inf", 0.0),  # no finite time: as if no task had started
        ("-inf", 0.0),
        ("nan", 0.0),
        ("garbage", 0.0),
        ("", 0.0),
    ],
)
def test_start_turn_waits_at_most_the_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str, expected: float
) -> None:
    from autoware_carla_scenario.slurm import worker

    clock = _FrozenClock(2000.0)
    monkeypatch.setattr(worker, "time", clock)
    lock = tmp_path / "start.lock"
    lock.write_text(recorded + "\n", encoding="utf-8")
    assert worker.wait_for_start_turn(lock, 5.0) == expected
    assert clock.slept == ([expected] if expected else [])
    assert float(lock.read_text(encoding="utf-8")) == 2000.0


@pytest.mark.parametrize("spacing", [math.inf, math.nan, -1.0])
def test_start_turn_refuses_a_spacing_that_is_not_finite_or_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spacing: float
) -> None:
    from autoware_carla_scenario.slurm import worker

    clock = _FrozenClock(2000.0)
    monkeypatch.setattr(worker, "time", clock)
    lock = tmp_path / "start.lock"
    lock.write_text("2000.0\n", encoding="utf-8")
    # Not an OverflowError out of time.sleep(inf) (#784).
    with pytest.raises(ValueError, match="launcher.start_stagger_seconds"):
        worker.wait_for_start_turn(lock, spacing)
    assert clock.slept == []


def _recorded_starts(monkeypatch: pytest.MonkeyPatch) -> Any:
    """``wait_for_start_turn``, returning the start it recorded in the lock.

    Its last ``time.time()`` is the start it writes, so each thread keeps its
    own last one. A stamp taken after the call returns is late when the thread
    is scheduled late, making the next gap look short; reading the lock back
    races the next task, which may already have overwritten (or truncated) it
    (#726).
    """
    import threading

    from autoware_carla_scenario.slurm import worker

    local = threading.local()

    class _Clock:
        def __getattr__(self, name: str) -> Any:
            return getattr(time, name)

        def time(self) -> float:
            local.last = now = time.time()
            return now

    monkeypatch.setattr(worker, "time", _Clock())

    def start(lock: Path, spacing: float) -> float:
        worker.wait_for_start_turn(lock, spacing)
        return float(local.last)

    return start


def test_start_stagger_spaces_every_task_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from autoware_carla_scenario.slurm.worker import wait_for_start_turn

    lock = BatchLayout(tmp_path).start_lock
    spacing = 0.4
    started: list[float] = []

    recorded_start = _recorded_starts(monkeypatch)

    def start() -> None:
        started.append(recorded_start(lock, spacing))

    # Three tasks that start together (a first wave)...
    threads = [threading.Thread(target=start) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # ...and one that starts later, once a slot frees up, right after them.
    start()
    # The recorded start is the one in the lock (nothing else runs now).
    assert started[-1] == float(lock.read_text())
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


def test_port_slots_are_exclusive_and_skip_ports_in_use(
    tmp_path: Path, port_base: int
) -> None:
    base, slots = port_base, RANGE_SLOTS
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
    tmp_path: Path, port_base: int
) -> None:
    from autoware_carla_scenario.slurm.ports import CarlaWatchdog

    base = port_base
    slot = acquire_slot(base, 1, start=0, lock_dir=tmp_path)
    watchdog = CarlaWatchdog(slot)
    slot.release(keep_lock=True)
    with pytest.raises(RuntimeError):
        acquire_slot(base, 1, start=0, lock_dir=tmp_path)
    watchdog.finish()  # no CARLA runs: it ends, and with it the lock
    again = acquire_slot(base, 1, start=0, lock_dir=tmp_path)
    again.release()


def test_the_watchdog_removes_the_port_file_of_a_worker_that_died(
    tmp_path: Path, port_base: int
) -> None:
    """A SIGKILLed worker never removes its ``.port`` file (#525): the watchdog,
    which sees the pipe close without ``done``, does."""
    from autoware_carla_scenario.slurm.ports import CarlaWatchdog

    slot = acquire_slot(port_base, 1, start=0, lock_dir=tmp_path)
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
    slot = acquire_slot(port_base, 1, start=0, lock_dir=tmp_path)
    port_file.write_text("x\n", encoding="utf-8")
    watchdog = CarlaWatchdog(slot, port_file)
    slot.release(keep_lock=True)
    watchdog.finish()
    assert port_file.exists()


#: A stand-in CARLA: says ``ready`` once its SIGTERM handling is set, then waits.
_STAND_IN = (
    "import signal, sys, time\n"
    "if sys.argv[1] == 'ignore':\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print(None, flush=True)\n"
    "time.sleep(120)\n"
)
#: Starts the stand-in in a session of its own, as CarlaServerManager starts
#: CARLA, prints its pid and exits: the stand-in is not the test's child.
_ORPHANING = (
    "import subprocess, sys\n"
    "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], *sys.argv[2:]],\n"
    "                         stdout=subprocess.PIPE, start_new_session=True)\n"
    "assert child.stdout is not None\n"
    "child.stdout.readline()\n"
    "print(child.pid, flush=True)\n"
)


class _StandIns:
    """Starts stand-in CARLAs on a port of the test's own range."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.children: list[subprocess.Popen[str]] = []

    def start(self, on_term: str, *, child: bool) -> int:
        """Start one and wait until it is ready; its pid."""
        args = [on_term, f"-carla-rpc-port={self.port}"]
        if not child:
            output = subprocess.run(
                [sys.executable, "-c", _ORPHANING, _STAND_IN, *args],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
                timeout=30,
            ).stdout
            return int(output.strip())
        process = subprocess.Popen(
            [sys.executable, "-c", _STAND_IN, *args], stdout=subprocess.PIPE, text=True
        )
        self.children.append(process)
        assert process.stdout is not None
        process.stdout.readline()
        return process.pid


@pytest.fixture
def stand_ins(port_base: int) -> Iterator[_StandIns]:
    """Stand-in CARLAs on a spare port of the test's first slot, all killed
    and reaped after the test."""
    stand_ins = _StandIns(port_base + 7)
    try:
        yield stand_ins
    finally:
        for pid in carla_processes_on_port(stand_ins.port):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        for process in stand_ins.children:
            process.kill()
            process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()


@pytest.mark.parametrize("child", [True, False], ids=["child", "not-a-child"])
@pytest.mark.parametrize("on_term", ["ignore", "exit"])
def test_kill_carla_on_port_returns_only_what_still_runs(
    stand_ins: _StandIns, on_term: str, child: bool
) -> None:
    """A CARLA that ignores SIGTERM and is SIGKILLed is not reported as still
    running (#771), nor one that ends on SIGTERM, whether or not it is the
    caller's child (a child stays a zombie until it is reaped)."""
    pid = stand_ins.start(on_term, child=child)
    assert carla_processes_on_port(stand_ins.port) == [pid]
    assert kill_carla_on_port(stand_ins.port, grace_s=0.5) == []
    assert carla_processes_on_port(stand_ins.port) == []


def test_kill_carla_on_port_with_nothing_running(stand_ins: _StandIns) -> None:
    assert kill_carla_on_port(stand_ins.port, grace_s=0.5) == []


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
    ("overrides", "key"),
    [
        (
            [
                "autoware.launcher.type=docker",
                "autoware.launcher.ros_domain_ids=[10,11]",
            ],
            "autoware.launcher.ros_domain_ids",
        ),
        (["autoware.launcher.type=docker"], "autoware.launcher.ros_domain_ids"),
        (["autoware.launcher.type=command"], "autoware.launcher.env"),
        (["ego.entity=ros2", "ros2.domain_id=5"], "ros2.domain_id"),
    ],
)
def test_two_tasks_on_a_node_get_disjoint_ros_domains(
    tmp_path: Path, overrides: list[str], key: str
) -> None:
    """Two tasks (two slots) on one node never share a DDS domain (#1862);
    each keeps its own for every case."""
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm.ports import PortSlot, RosDomains
    from autoware_carla_scenario.slurm.worker import worker_overrides

    def domains_of(changes: dict[str, Any]) -> set[int]:
        value = changes[key]
        if key == "autoware.launcher.env":
            return {int(value["ROS_DOMAIN_ID"])}
        if isinstance(value, int):
            return {value}
        return set(value)

    first, second = RosDomains(tmp_path), RosDomains(tmp_path)
    try:
        taken = []
        for index, domains in ((0, first), (1, second)):
            slot = PortSlot(index=index, base=20000 + 10 * index)
            cfg = run._compose_config(_CONCRETE, overrides)
            changes = worker_overrides(cfg, slot, domains)
            got = domains_of(changes)
            # Applied to the config, and the same for the task's next case.
            assert _select(cfg, key) == changes[key]
            again = worker_overrides(
                run._compose_config(_CONCRETE, overrides), slot, domains
            )
            assert domains_of(again) == got
            taken.append(got)
        assert not taken[0] & taken[1]
        if "autoware.launcher.ros_domain_ids=[10,11]" in overrides:
            # The first task gets the ids it asked for.
            assert taken[0] == {10, 11}
            assert taken[1] == {12, 13}
    finally:
        first.release()
        second.release()
    # Once let go, the domains are free again.
    third = RosDomains(tmp_path)
    try:
        assert third.take(10) == 10
    finally:
        third.release()


def _select(cfg: Any, key: str) -> Any:
    """The dotted *key* of *cfg*, as plain values."""
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, key)
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


def test_worker_leaves_the_ros_domains_alone_without_a_holder() -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm.ports import PortSlot
    from autoware_carla_scenario.slurm.worker import worker_overrides

    cfg = run._compose_config(_CONCRETE, ["ego.entity=ros2", "ros2.domain_id=5"])
    changes = worker_overrides(cfg, PortSlot(index=3, base=20030))
    assert "ros2.domain_id" not in changes
    assert cfg.ros2.domain_id == 5


def test_ros_domains_refuse_an_id_out_of_range_and_run_out(tmp_path: Path) -> None:
    from autoware_carla_scenario.slurm.ports import RosDomains

    domains = RosDomains(tmp_path, highest=2)
    others = RosDomains(tmp_path, highest=2)
    try:
        with pytest.raises(ValueError, match="from 0 to 2"):
            domains.take(3)
        # From the id asked for on, then round from 0.
        assert [domains.take(i) for i in (1, 2)] == [1, 2]
        assert others.take(1) == 0
        with pytest.raises(RuntimeError, match="No free ROS domain"):
            others.take(2)
    finally:
        domains.release()
        others.release()


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


def test_settings_keep_the_array_within_max_array_size() -> None:
    """#1347: one case a task made an array SLURM's MaxArraySize refused."""
    from autoware_carla_scenario.slurm.settings import (
        DEFAULT_MAX_ARRAY_SIZE,
        parse_max_array_size,
    )

    assert DEFAULT_MAX_ARRAY_SIZE == 1001
    # Up to the limit, one case a task as before; past it, fewer tasks.
    assert SlurmSettings().chunk_size(1001) == 1
    assert SlurmSettings().chunk_size(1002) == 2
    assert SlurmSettings().chunk_size(2500) == 3
    assert SlurmSettings().chunk_size(2500, max_tasks=100) == 25
    assert SlurmSettings(array_parallelism=2000).chunk_size(5000) == 5
    assert SlurmSettings(array_parallelism=10).chunk_size(5000) == 500
    # cases_per_task is taken as it is.
    assert SlurmSettings(cases_per_task=1).chunk_size(5000) == 1
    config = "Configuration data as of ...\nMaxArraySize            = 40001\n"
    assert parse_max_array_size(config) == 40001
    assert parse_max_array_size("MaxJobCount = 10000\n") is None
    assert parse_max_array_size("MaxArraySize = (null)\n") is None
    assert parse_max_array_size("MaxArraySize = 0\n") is None
    assert parse_max_array_size("MaxArraySize = ²\n") is None


def _scontrol_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str | None
) -> None:
    """PATH holds only an ``scontrol`` running *script* (none for ``None``)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if script is not None:
        scontrol = bin_dir / "scontrol"
        scontrol.write_text(f"#!/bin/sh\n{script}\n")
        scontrol.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("echo 'MaxArraySize            = 300'", 300),
        ("echo 'MaxJobCount = 10000'", 1001),
        ("echo 'slurm_load_ctl_conf error' >&2; exit 1", 1001),
        (None, 1001),
        ("printf 'ClusterName = \\377\\nMaxArraySize = 300\\n'", 300),
        ("printf 'MaxArraySize = \\377\\n'", 1001),
    ],
    ids=["reported", "unnamed", "failing", "missing", "not-utf8", "garbled"],
)
def test_max_array_size_asks_scontrol_and_falls_back_to_slurm_s_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str | None,
    expected: int,
) -> None:
    from autoware_carla_scenario.slurm import submit

    _scontrol_on_path(tmp_path, monkeypatch, script)
    assert submit.max_array_size() == expected


def _submit_many(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cases: int,
    settings: SlurmSettings,
    limit: int | None,
) -> tuple[BatchLayout, list[list[str]]]:
    """Submit *cases* cases of a batch on disk, sbatch faked; its commands."""
    from autoware_carla_scenario.slurm import submit

    layout = _batch_on_disk(tmp_path, cases=cases)
    commands: list[list[str]] = []

    def run_sbatch(command: list[str]) -> str:
        commands.append(command)
        return "88"

    monkeypatch.setattr(submit, "_run_sbatch", run_sbatch)
    monkeypatch.setattr(submit, "job_script", lambda *_a, **_k: "#!/bin/sh\n")
    if limit is not None:
        monkeypatch.setattr(submit, "max_array_size", lambda: limit)
    submit.submit_cases(layout, layout.read_manifest(), settings, list(range(cases)))
    return layout, commands


@pytest.mark.parametrize(
    ("cases", "limit", "tasks"),
    [(2500, None, 834), (2500, 300, 278), (1001, None, 1001), (3, None, 3)],
    ids=["default-limit", "scontrol-limit", "at-the-limit", "small"],
)
def test_submit_spreads_a_large_batch_within_max_array_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cases: int,
    limit: int | None,
    tasks: int,
) -> None:
    """#1347: every case once, in no more tasks than MaxArraySize."""
    if limit is None:  # scontrol fails: SLURM's default, 1001
        _scontrol_on_path(tmp_path, monkeypatch, "exit 1")
    layout, commands = _submit_many(
        tmp_path,
        monkeypatch,
        cases,
        SlurmSettings(wait=False, collect_job=False),
        limit,
    )
    submission = layout.read_submission(1)
    chunks = submission["chunks"]
    assert len(chunks) == tasks <= (limit or 1001)
    assert [case for chunk in chunks for case in chunk] == list(range(cases))
    if cases <= (limit or 1001):
        assert all(len(chunk) == 1 for chunk in chunks)
    assert f"--array=0-{tasks - 1}" in commands[0]
    assert f"in {tasks} task(s)" in capsys.readouterr().out


def test_submit_refuses_a_cases_per_task_that_exceeds_max_array_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1347: a cases_per_task set too small is named before sbatch runs."""
    layout = None
    with pytest.raises(RuntimeError) as raised:
        layout, _ = _submit_many(
            tmp_path,
            monkeypatch,
            2500,
            SlurmSettings(cases_per_task=2, wait=False, collect_job=False),
            1001,
        )
    message = str(raised.value)
    assert "MaxArraySize (1001)" in message
    assert "launcher.cases_per_task to at least 3" in message
    assert layout is None
    assert BatchLayout(tmp_path / "batch").submissions() == [0]


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (
            "sbatch failed (1): sbatch: error: Batch job submission failed: "
            "Invalid job array specification",
            True,
        ),
        ("sbatch failed (1): sbatch: error: array size exceeds MaxArraySize", True),
        (
            "sbatch failed (1): sbatch: error: Unable to open file "
            "/scratch/array_runs/job.sh",
            False,
        ),
    ],
    ids=["invalid-array", "max-array-size", "unrelated-path"],
)
def test_an_array_sbatch_refuses_names_cases_per_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str, hint: bool
) -> None:
    """#1347: should sbatch refuse the array all the same, say what to set;
    any other refusal is raised as it was."""
    from autoware_carla_scenario.slurm import submit

    layout = _batch_on_disk(tmp_path, cases=5)

    def run_sbatch(_command: list[str]) -> str:
        raise RuntimeError(error)

    monkeypatch.setattr(submit, "_run_sbatch", run_sbatch)
    monkeypatch.setattr(submit, "job_script", lambda *_a, **_k: "#!/bin/sh\n")
    monkeypatch.setattr(submit, "max_array_size", lambda: 1001)
    with pytest.raises(RuntimeError) as raised:
        submit.submit_cases(
            layout, layout.read_manifest(), SlurmSettings(), list(range(5))
        )
    message = str(raised.value)
    assert message.startswith(error)
    assert ("launcher.cases_per_task" in message) is hint
    assert layout.submissions() == [0]


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
        _layout: object,
        _manifest: object,
        _settings: object,
        chosen: list[int],
        **_kwargs: object,
    ) -> int:
        submitted.append(chosen)
        return 1

    monkeypatch.setattr(cli, "submit_cases", submit_cases)
    monkeypatch.setattr(cli, "after_submit", lambda *_: 0)
    argv = ["resubmit", str(layout.batch_dir), "--cases", "2,s#3, 0,s#1,2"]
    assert cli.main(argv) == 0
    assert "Resubmitting 2 case(s): s#1, s#3" in capsys.readouterr().out
    assert submitted == [[0, 2]]


@pytest.mark.parametrize("item", ["²", "٣"])
def test_resubmit_refuses_a_non_ascii_digit_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    item: str,
) -> None:
    """#1612: ``--cases ²`` raised (``"²".isdigit()`` but not ``int()``)."""
    from autoware_carla_scenario.slurm import cli

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    argv = ["resubmit", str(layout.batch_dir), "--cases", item, "--dry-run"]
    assert cli.main(argv) == 1
    assert f"Error: no case {item!r} in the batch" in capsys.readouterr().err


def test_non_ascii_digits_are_no_submission_and_no_task(tmp_path: Path) -> None:
    """#1612: a stray ``².json`` and an ``squeue`` line ``7_²`` are skipped."""
    from autoware_carla_scenario.slurm.scripts import squeue_task_states

    layout = _batch_on_disk(tmp_path)
    layout.submission(0).with_name("².json").write_text("{}")
    assert layout.submissions() == [0]
    assert squeue_task_states("7", "7_0 RUNNING\n7_² RUNNING\n") == {0: "RUNNING"}


def test_concurrent_submissions_get_indices_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1352: two submit_cases at once took the same index; the second
    record replaced the first, whose job id was lost."""
    import threading

    from autoware_carla_scenario.slurm import submit

    layout = _batch_on_disk(tmp_path)
    jobs = iter(["1000", "1001"])
    lock = threading.Lock()

    def run_sbatch(_command: list[str]) -> str:
        with lock:
            job = next(jobs)
        time.sleep(0.5)
        return job

    monkeypatch.setattr(submit, "_run_sbatch", run_sbatch)
    monkeypatch.setattr(submit, "job_script", lambda *_a, **_k: "#!/bin/sh\n")
    monkeypatch.setattr(submit, "max_array_size", lambda: 1001)
    settings = SlurmSettings(wait=False, collect_job=False)
    indices: list[int] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def submit_one() -> None:
        barrier.wait()
        try:
            indices.append(
                submit.submit_cases(layout, layout.read_manifest(), settings, [0, 1])
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=submit_one) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(indices) == [1, 2]
    assert layout.submissions() == [0, 1, 2]
    records = [layout.read_submission(i) for i in (1, 2)]
    assert [r["index"] for r in records] == [1, 2]
    assert sorted(r["job_id"] for r in records) == ["1000", "1001"]
    # No temporary file of the reservation is left.
    assert sorted(p.name for p in (layout.root / "submissions").iterdir()) == [
        "0.json",
        "1.json",
        "2.json",
    ]


def test_reserve_submission_passes_over_a_taken_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1352: an index another submitter created is never written over."""
    layout = _batch_on_disk(tmp_path)
    taken = {"index": 1, "job_id": "5"}
    layout.submission(1).write_text(json.dumps(taken))
    # Told index 1, as a submitter racing the one that created 1.json was.
    told = iter([1])
    real = BatchLayout.next_submission
    monkeypatch.setattr(
        BatchLayout, "next_submission", lambda self: next(told, None) or real(self)
    )
    record: dict[str, Any] = {"index": None, "job_id": None}
    assert layout.reserve_submission(record) == 2
    assert layout.read_submission(1) == taken
    assert layout.read_submission(2) == {"index": 2, "job_id": None}


def test_a_failed_sbatch_removes_the_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1352: whatever fails before the job id is known frees the index."""
    from autoware_carla_scenario.slurm import submit

    layout = _batch_on_disk(tmp_path)
    seen: list[dict[str, Any]] = []

    def run_sbatch(_command: list[str]) -> str:
        # The reservation holds a whole record while sbatch runs.
        seen.append(layout.read_submission(1))
        raise RuntimeError("sbatch failed (1): down")

    monkeypatch.setattr(submit, "_run_sbatch", run_sbatch)
    monkeypatch.setattr(submit, "job_script", lambda *_a, **_k: "#!/bin/sh\n")
    monkeypatch.setattr(submit, "max_array_size", lambda: 1001)
    with pytest.raises(RuntimeError, match="down"):
        submit.submit_cases(layout, layout.read_manifest(), SlurmSettings(), [0])
    assert seen[0]["index"] == 1 and seen[0]["job_id"] is None
    assert layout.submissions() == [0]

    def refuse(_index: int) -> None:
        raise RuntimeError("busy")

    with pytest.raises(RuntimeError, match="busy"):
        submit.submit_cases(
            layout, layout.read_manifest(), SlurmSettings(), [0], before_sbatch=refuse
        )
    assert layout.submissions() == [0]
    assert len(seen) == 1  # sbatch never ran


def test_a_timed_out_sbatch_keeps_the_reservation_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1786: an sbatch that timed out may have been taken, so its record
    stays (job_id None), its tasks would find their chunks, its index is not
    reused, and resubmits are refused while it is in flight.  #1791: the
    record says that sbatch timed out, and when, and the refusal says so."""
    from autoware_carla_scenario.slurm import cli, queue, submit
    from autoware_carla_scenario.slurm.manifest import (
        sbatch_in_flight,
        sbatch_timed_out,
    )

    layout = _batch_on_disk(tmp_path)
    sbatch = _hanging_tool(tmp_path / "bin", "sbatch")
    monkeypatch.setattr(queue, "require_sbatch", lambda: str(sbatch))
    monkeypatch.setattr(queue, "SBATCH_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(submit, "job_script", lambda *_a, **_k: "#!/bin/sh\n")
    monkeypatch.setattr(submit, "max_array_size", lambda: 1001)
    started = time.monotonic()
    with pytest.raises(submit.SbatchTimedOut, match="record is kept"):
        submit.submit_cases(layout, layout.read_manifest(), SlurmSettings(), [0, 2])
    assert time.monotonic() - started < 10.0
    assert layout.submissions() == [0, 1]
    record = layout.read_submission(1)
    assert record["job_id"] is None and record["chunks"] == [[0], [2]]
    assert record["sbatch_timed_out"] is True
    assert isinstance(record["sbatch_timed_out_at"], str)
    assert sbatch_in_flight(record) and sbatch_timed_out(record)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    capsys.readouterr()
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "the sbatch of submission(s) 1 of this batch timed out" in err
    assert "squeue --me" in err and "--force" in err
    assert "being submitted" not in err and "has not returned" not in err


def test_resubmit_refuses_while_a_submission_is_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1352: a record whose sbatch has not returned (job_id None) is in
    flight, not inactive; one left by a submitter that died is ignored once
    SBATCH_IN_FLIGHT_SECONDS have passed."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.manifest import (
        SBATCH_IN_FLIGHT_SECONDS,
        write_json,
    )

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    pending: dict[str, Any] = {
        "index": 1,
        "job_id": None,
        "submitted_at": time.time(),
        "chunks": [],
    }
    write_json(layout.submission(1), pending)
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "submission(s) 1 of this batch are being submitted" in err
    assert "--force" in err
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run", "--force"]) == 0
    capsys.readouterr()
    pending["submitted_at"] = time.time() - SBATCH_IN_FLIGHT_SECONDS - 1
    write_json(layout.submission(1), pending)
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 0
    assert "Resubmitting 3 case(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "content", ["", "{not json", "[1, 2]", "null"], ids=["empty", "bad", "list", "null"]
)
def test_an_unreadable_submission_record_refuses_only_while_young(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    content: str,
) -> None:
    """#1352: a record that cannot be read (empty, not JSON, not an object)
    may be one being written: it refuses a resubmit for
    SBATCH_IN_FLIGHT_SECONDS after it was last modified, then is ignored,
    so a corrupt one never blocks the viewer, which cannot pass --force."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.manifest import SBATCH_IN_FLIGHT_SECONDS

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    path = layout.submission(1)
    path.write_text(content)
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert f"submission record(s) {path} of this batch are unreadable" in err
    assert "being submitted" not in err and "--force" in err
    old = time.time() - SBATCH_IN_FLIGHT_SECONDS - 1
    os.utime(path, (old, old))
    with caplog.at_level("WARNING", logger=cli.__name__):
        assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 0
    assert "Resubmitting 3 case(s)" in capsys.readouterr().out
    assert any(str(path) in r.getMessage() for r in caplog.records)


_UNREADABLE = pytest.mark.parametrize(
    "content", ["", "{not json", "[1, 2]", "null"], ids=["empty", "bad", "list", "null"]
)


@_UNREADABLE
def test_read_submission_record_is_none_for_an_unreadable_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    """#1356: the shared reader names the file it cannot read; a missing
    record is None without a warning."""
    from autoware_carla_scenario.slurm import manifest

    layout = _batch_on_disk(tmp_path)
    path = layout.submission(1)
    path.write_text(content)
    with caplog.at_level("WARNING", logger=manifest.__name__):
        assert manifest.read_submission_record(path) is None
        assert layout.submission_record(1) is None
    assert any(str(path) in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING", logger=manifest.__name__):
        assert layout.submission_record(2) is None
    assert not caplog.records
    assert layout.submission_record(0) == layout.read_submission(0)


@_UNREADABLE
def test_status_lists_an_unreadable_submission_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    from autoware_carla_scenario.slurm import cli

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    path = layout.submission(1)
    path.write_text(content)
    assert cli.main(["status", str(layout.batch_dir)]) == 0
    out = capsys.readouterr().out
    assert "submission 0: job 77, 1 task(s)" in out
    assert f"submission 1: unreadable record {path}" in out


@_UNREADABLE
def test_cancel_skips_an_unreadable_submission_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    from autoware_carla_scenario.slurm import cli

    layout = _batch_on_disk(tmp_path)
    cancelled: list[list[str]] = []

    def cancel_jobs(jobs: list[str]) -> bool:
        cancelled.append(jobs)
        return True

    monkeypatch.setattr(cli, "cancel_jobs", cancel_jobs)
    path = layout.submission(1)
    path.write_text(content)
    assert cli.main(["cancel", str(layout.batch_dir)]) == 0
    assert cancelled == [["77"]]
    captured = capsys.readouterr()
    assert f"Skipping {path}" in captured.err
    assert "Cancelling job(s) 77" in captured.out


@_UNREADABLE
def test_collect_counts_an_unreadable_record_s_cases_as_not_submitted_by_it(
    tmp_path: Path, content: str
) -> None:
    """#1356: a case's missing message names the last submission that can be
    read; with none, the case was never submitted."""
    from autoware_carla_scenario.slurm.collect import collect

    layout = _batch_on_disk(tmp_path)
    layout.submission(1).write_text(content)
    assert collect(layout, quiet_hint=True) == 1
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert ["task 0 of job 77 never started" in e["message"] for e in summary] == [
        True
    ] * 3
    layout.submission(0).write_text(content)
    assert collect(layout, quiet_hint=True) == 1
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert ["it was never submitted" in e["message"] for e in summary] == [True] * 3


@_UNREADABLE
def test_wait_and_the_worker_refuse_an_unreadable_submission_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    from autoware_carla_scenario.slurm import submit, worker

    layout = _batch_on_disk(tmp_path)
    path = layout.submission(1)
    path.write_text(content)
    assert submit.follow(layout, 1, SlurmSettings()) == 1
    assert f"the submission record {path} is missing or cannot be read" in (
        capsys.readouterr().err
    )
    assert worker.run_task(layout.manifest, 1, 0) == 2
    assert f"the submission record {path} is missing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    ["[]", "null", "2", '"manifest"', '{"version": true}', '{"version": "1"}'],
    ids=["list", "null", "number", "string", "bool-version", "str-version"],
)
def test_a_manifest_that_is_not_a_readable_object_is_a_value_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    """#1630: JSON that is not an object, or a version that is a bool
    (``True == 1``), is the ValueError read_manifest documents, which the
    CLI reports, not an AttributeError."""
    from autoware_carla_scenario.slurm import cli

    layout = _batch_on_disk(tmp_path)
    layout.manifest.write_text(content)
    with pytest.raises(ValueError, match=re.escape(str(layout.manifest))):
        layout.read_manifest()
    for command in ("status", "collect"):
        assert cli.main([command, str(layout.batch_dir)]) == 1
        assert f"Error: {layout.manifest}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    ["", "{not json", "[1, 2]", "null", "3", '"done"'],
    ids=["empty", "bad", "list", "null", "number", "string"],
)
def test_a_result_or_task_status_that_is_not_an_object_is_none(
    tmp_path: Path, content: str
) -> None:
    """#1630: read_result and read_task return a JSON object or None, as
    their callers expect: such a case counts as missing, its task as never
    started."""
    from autoware_carla_scenario.slurm.collect import collect

    layout = _batch_on_disk(tmp_path)
    result = layout.result(1)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(content)
    task = layout.task(0, 0)
    task.parent.mkdir(parents=True, exist_ok=True)
    task.write_text(content)
    assert layout.read_result(1) is None
    assert layout.read_task(0, 0) is None
    assert missing_cases(layout) == [0, 1, 2]
    assert unfinished_cases(layout) == [0, 1, 2]
    assert collect(layout, quiet_hint=True) == 1
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert ["task 0 of job 77 never started" in e["message"] for e in summary] == [
        True
    ] * 3


def _recorded(passed: Any = True, **result: Any) -> dict[str, Any]:
    """A case's outcome as a task records it, from submission 0."""
    return {
        "submission": 0,
        "errored": False,
        "result": {
            "passed": passed,
            "message": "",
            "elapsed_seconds": 1.0,
            "condition_statuses": [],
            **result,
        },
    }


_NO_VERDICT = pytest.mark.parametrize(
    "record",
    [
        _recorded("false"),
        _recorded("true"),
        _recorded(1),
        _recorded(None),
        {"submission": 0, "errored": False},
        {"submission": 0, "errored": False, "result": "passed"},
        {"submission": 0, "errored": False, "result": None},
        {"submission": 0, "errored": False, "result": [True]},
        # #1726: a verdict in a result that cannot be reported.
        {"submission": 0, "errored": False, "result": {"passed": True}},
        _recorded(True, message=None),
        _recorded(True, message=3),
        _recorded(True, elapsed_seconds="soon"),
        _recorded(True, elapsed_seconds=None),
        _recorded(True, elapsed_seconds=True),
        _recorded(True, elapsed_seconds=10**400),
        _recorded(True, condition_statuses="none"),
        _recorded(True, condition_statuses=["ok"]),
        _recorded(True, condition_statuses=[{"label": "a", "satisfied": True}]),
        _recorded(
            True,
            condition_statuses=[{"label": "a", "satisfied": "no", "message": ""}],
        ),
        _recorded(False, failure="collided"),
        _recorded(False, failure={"at": "later"}),
    ],
    ids=[
        "str-false",
        "str-true",
        "int",
        "null",
        "no-result",
        "str-result",
        "null-result",
        "list-result",
        "passed-only",
        "null-message",
        "int-message",
        "str-elapsed",
        "null-elapsed",
        "bool-elapsed",
        "overflowing-elapsed",
        "str-statuses",
        "str-status",
        "status-without-message",
        "str-satisfied",
        "str-failure",
        "str-failure-at",
    ],
)


def _with_case_1(tmp_path: Path, record: dict[str, Any]) -> BatchLayout:
    """A batch whose cases 0 and 2 passed and whose case 1 recorded *record*."""
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path)
    layout.result(0).parent.mkdir(parents=True, exist_ok=True)
    write_json(layout.result(0), _recorded(True))
    write_json(layout.result(1), record)
    write_json(layout.result(2), _recorded(True))
    return layout


def test_recorded_verdict_is_a_boolean_passed_or_none() -> None:
    """#1718: the one reader of a recorded verdict."""
    from autoware_carla_scenario.slurm.collect import recorded_verdict

    assert recorded_verdict(_recorded(True)) is True
    assert recorded_verdict(_recorded(False)) is False
    assert recorded_verdict(None) is None
    for value in ("false", "true", 1, 0, None, 1.0):
        assert recorded_verdict(_recorded(value)) is None
    assert recorded_verdict({"result": "x"}) is None
    assert recorded_verdict({}) is None


def test_what_the_worker_records_is_readable() -> None:
    """#1726: every result ``ScenarioResult.to_dict()`` writes keeps its
    verdict -- with or without its optional fields, a whole number of seconds
    included."""
    from autoware_carla_scenario.conditions.base import (
        ConditionStatus,
        ScenarioResult,
    )
    from autoware_carla_scenario.failure import Failure, FailureCategory
    from autoware_carla_scenario.slurm.collect import recorded_verdict
    from autoware_carla_scenario.slurm.manifest import readable_result

    status = ConditionStatus("pass[0](t)", True, "ok", "TimeoutCondition", "pass")
    failure = Failure("fail[0](c)", FailureCategory.OTHER, 3.0, "collided")
    for result in (
        ScenarioResult(True, "", 0.0),
        ScenarioResult(True, "ok", 12.5, [status], rerun_recording="r.rrd"),
        ScenarioResult(False, "collided", 3.0, [status], failure=failure),
    ):
        data = json.loads(json.dumps(result.to_dict()))
        assert readable_result(data)
        record = {"submission": 0, "errored": False, "result": data}
        assert recorded_verdict(record) is result.passed
    assert recorded_verdict(_recorded(True, elapsed_seconds=4)) is True
    assert recorded_verdict(_recorded(True, condition_statuses=None)) is True
    without = _recorded(False)
    del without["result"]["condition_statuses"]
    assert recorded_verdict(without) is False


_UNWRITABLE = pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EROFS, "Read-only file system"),
    ],
    ids=["permission", "read-only"],
)


def _one_error_line(err: str) -> str:
    """The one line *err* holds, which is no traceback."""
    assert "Traceback" not in err
    lines = err.splitlines()
    assert len(lines) == 1, err
    return lines[0]


@_UNWRITABLE
@pytest.mark.parametrize("output", ["batch_results.json", "collected.json"])
def test_collect_into_an_unwritable_batch_reports_one_error_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: OSError,
    output: str,
) -> None:
    """#1740: a batch directory ``collect`` cannot write (shared storage, a
    read-only mount) is one ``Error:`` line, after the summary."""
    from autoware_carla_scenario import batch_report
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm import collect as collect_module

    layout = _with_case_1(tmp_path, _recorded(True))
    # Monkeypatched, as a chmod would not stop root.
    target = layout.batch_dir / output
    written: list[Path] = []

    def refuse(path: Path, *args: Any, **kwargs: Any) -> None:
        written.append(Path(path))
        raise type(error)(error.errno, error.strerror, str(path))

    if output == "batch_results.json":
        monkeypatch.setattr(batch_report, "write_text_atomically", refuse)
    else:
        monkeypatch.setattr(collect_module, "write_json", refuse)
        target = layout.root / output
    assert cli.main(["collect", str(layout.batch_dir)]) == 1
    captured = capsys.readouterr()
    assert written == [target]
    line = _one_error_line(captured.err)
    assert line == f"Error: [Errno {error.errno}] {error.strerror}: '{target}'"
    # The verdict is printed before the error.
    assert "[PASS] s#1" in captured.out
    assert "3/3 scenarios passed" in captured.out
    assert ("Result JSON:" in captured.out) == (output == "collected.json")


def test_collect_still_prints_its_hints_before_collected_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1740: ``collected.json`` is written last, so the hint to resubmit the
    missing cases is printed though it cannot be."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm import collect as collect_module

    layout = _batch_on_disk(tmp_path)

    def refuse(path: Path, data: Any) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(collect_module, "write_json", refuse)
    assert cli.main(["collect", str(layout.batch_dir)]) == 1
    captured = capsys.readouterr()
    assert _one_error_line(captured.err).startswith("Error: [Errno 13]")
    assert "0/3 scenarios passed, 3 errored" in captured.out
    assert "3 case(s) have no result" in captured.out
    assert (layout.batch_dir / "batch_results.json").is_file()


def _advised_layout(tmp_path: Path, situation: str) -> BatchLayout:
    """A batch of three cases, none with a result, its one submission in
    *situation* (#1803)."""
    from autoware_carla_scenario.slurm.manifest import (
        SBATCH_IN_FLIGHT_SECONDS,
        write_json,
    )

    layout = _batch_on_disk(tmp_path)
    chunks = [[0, 1, 2]]
    now = time.time()
    if situation in ("running", "queued", "finished"):
        if situation != "queued":
            state = "running" if situation == "running" else "failed"
            write_json(layout.task(0, 0), {"task": 0, "state": state})
        return layout
    record: dict[str, Any] = {"index": 0, "job_id": None, "chunks": chunks}
    if situation == "in_flight":
        record["submitted_at"] = now
    elif situation in ("timed_out_young", "timed_out_old"):
        old = situation == "timed_out_old"
        record["submitted_at"] = now - (SBATCH_IN_FLIGHT_SECONDS + 60 if old else 0)
        record["sbatch_timed_out"] = True
        record["sbatch_timed_out_at"] = "2026-11-06T12:00:00"
    else:
        raise AssertionError(situation)
    write_json(layout.submission(0), record)
    return layout


@pytest.mark.parametrize(
    "situation",
    ["running", "queued", "in_flight", "timed_out_young", "timed_out_old", "finished"],
)
def test_collect_advises_only_what_resubmit_would_take(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], situation: str
) -> None:
    """#1803: a missing case's message, and the hint on the missing cases,
    advise a resubmit only where ``resubmit`` takes it: they say to wait for a
    task that may still be queued or running, to try again once an in-flight
    sbatch returns, and that a resubmit is refused while a timed-out sbatch's
    record is young."""
    from autoware_carla_scenario.slurm import cli

    layout = _advised_layout(tmp_path, situation)
    target = str(layout.batch_dir)
    assert cli.main(["collect", target]) == 1
    out = capsys.readouterr().out
    results = json.loads((layout.batch_dir / "batch_results.json").read_text())
    messages = [entry["message"] for entry in results]
    assert len(set(messages)) == 1
    message = messages[0]
    hint = out[out.index("3 case(s) have no result") :]
    wait = f"scenario-slurm wait {target}"
    resubmit = f"scenario-slurm resubmit {target}"
    refused = "a resubmit is refused until 600 s after the record was made"
    if situation in ("running", "queued"):
        state = "running" if situation == "running" else "never started"
        assert f"(task 0 of job 77 {state})" in message
        for text in (message, hint):
            assert f"wait for it with: {wait}" in text or (
                f"wait for them with: {wait}" in text
            )
            # Only once the task has ended, as resubmit would refuse before.
            assert re.search(r"once (it has|they have) ended, run", text)
            assert resubmit in text
    elif situation == "in_flight":
        assert "no job id yet (sbatch in flight)" in message
        for text in (message, hint):
            assert f"try again once sbatch returns: {wait}" in text
            assert "resubmit" not in text
    elif situation == "timed_out_young":
        for text in (message, hint):
            assert f"scenario-slurm collect {target}" in text
            assert f"{resubmit} ({refused}, unless with --force)" in text
    else:
        if situation == "finished":
            assert "(task 0 of job 77 failed)" in message
        else:
            assert "sbatch timed out" in message
        for text in (message, hint):
            assert text.rstrip().endswith(
                f"Run {'it' if text is message else 'them'} again with: {resubmit}"
            )
            assert "wait" not in text and "refused" not in text


def test_collect_hint_puts_a_running_task_before_a_resubmit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1803: resubmit refuses the whole batch while a task of it may still
    run, so of cases needing either, the hint says to wait first."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path)
    write_json(
        layout.submission(0),
        {"index": 0, "job_id": "77", "chunks": [[0, 1], [2]]},
    )
    write_json(layout.task(0, 0), {"task": 0, "state": "done"})
    write_json(layout.task(0, 1), {"task": 1, "state": "building"})
    target = str(layout.batch_dir)
    assert cli.main(["collect", target]) == 1
    out = capsys.readouterr().out
    results = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert results[0]["message"].endswith(
        f"Run it again with: scenario-slurm resubmit {target}"
    )
    assert f"wait for it with: scenario-slurm wait {target}" in results[2]["message"]
    hint = out[out.index("3 case(s) have no result") :]
    assert f"wait for them with: scenario-slurm wait {target}" in hint


def test_collect_names_an_older_submission_still_running_to_wait_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1803: ``wait`` follows the latest submission unless told which, so
    where an older one may still be running and the latest has ended, the
    message and the hint name it (``--submission 0``)."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path)
    write_json(layout.task(0, 0), {"task": 0, "state": "running"})
    write_json(
        layout.submission(1),
        {"index": 1, "job_id": "78", "chunks": [[2]]},
    )
    write_json(layout.task(1, 0), {"task": 0, "state": "failed"})
    target = str(layout.batch_dir)
    assert cli.main(["collect", target]) == 1
    out = capsys.readouterr().out
    results = json.loads((layout.batch_dir / "batch_results.json").read_text())
    older = f"scenario-slurm wait --submission 0 {target}"
    for entry in results[:2]:
        assert f"wait for it with: {older};" in entry["message"]
    assert results[2]["message"].endswith(
        f"Run it again with: scenario-slurm resubmit {target}"
    )
    hint = out[out.index("3 case(s) have no result") :]
    assert f"wait for them with: {older};" in hint
    assert f"scenario-slurm wait {target}" not in hint


@pytest.mark.parametrize("command", ["status", "collect", "wait", "resubmit"])
def test_an_unreadable_manifest_is_one_error_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """#1740: a manifest that cannot be read is one ``Error:`` line, and a
    batch directory with none keeps its message."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm import manifest as manifest_module

    layout = _batch_on_disk(tmp_path)
    read_json = manifest_module.read_json

    def unreadable(path: Path) -> Any:
        if Path(path) == layout.manifest:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return read_json(path)

    monkeypatch.setattr(manifest_module, "read_json", unreadable)
    assert cli.main([command, str(layout.batch_dir)]) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line == f"Error: [Errno 13] Permission denied: '{layout.manifest}'"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main([command, str(empty)]) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line == (
        f"Error: {empty} holds no SLURM batch (no {BatchLayout(empty).manifest})"
    )


def test_a_result_with_no_verdict_keeps_its_rerun_error(tmp_path: Path) -> None:
    """#1726: the Rerun recording a task failed to make is reported, in
    ``batch_results.json``, for a case whose result cannot be, too."""
    from autoware_carla_scenario.slurm.collect import collect

    record: dict[str, Any] = {
        "submission": 0,
        "errored": False,
        "result": {"passed": True},
        "rerun_error": "rerun died",
    }
    layout = _with_case_1(tmp_path, record)
    assert collect(layout, quiet_hint=True) == 1
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert summary[1].get("error") is True
    assert summary[1]["rerun_error"] == "rerun died"


@_NO_VERDICT
def test_a_result_with_no_verdict_is_an_error_not_a_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], record: dict[str, Any]
) -> None:
    """#1718: a recorded result whose ``passed`` is no boolean, or that has
    no ``result`` object, did not pass: collect reports it as ``ERROR`` naming
    its file and exits 1, resubmit picks it, status counts it as no pass;
    none of them raises."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.collect import collect

    layout = _with_case_1(tmp_path, record)
    assert missing_cases(layout) == []
    assert unfinished_cases(layout) == [1]
    assert unfinished_cases(layout, failed=True) == [1]
    assert collect(layout) == 1
    out = capsys.readouterr().out
    assert "[ERROR] s#2" in out
    assert str(layout.result(1)) in out
    assert "1 case(s) have a result with no verdict: s#2" in out
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert [e["passed"] for e in summary] == [True, False, True]
    assert summary[1].get("error") is True
    assert str(layout.result(1)) in summary[1]["message"]
    collected = json.loads((layout.root / "collected.json").read_text())
    assert collected["passed"] == 2
    assert collected["errored"] == ["s#2"]
    assert cli.main(["status", str(layout.batch_dir)]) == 0
    assert "3 with a result (2 passed, 0 errored)" in capsys.readouterr().out


def test_a_failed_result_is_picked_only_with_failed(tmp_path: Path) -> None:
    """#1718: a boolean ``false`` is a verdict: a failure, not an error."""
    from autoware_carla_scenario.slurm.collect import collect

    layout = _with_case_1(tmp_path, _recorded(False))
    assert unfinished_cases(layout) == []
    assert unfinished_cases(layout, failed=True) == [1]
    assert collect(layout, quiet_hint=True) == 1
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert [e["passed"] for e in summary] == [True, False, True]
    assert "error" not in summary[1]


@_NO_VERDICT
def test_follow_counts_a_result_with_no_verdict_as_no_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    record: dict[str, Any],
) -> None:
    """#1718: follow's progress line counts only a boolean ``true``."""
    from autoware_carla_scenario.slurm import submit

    layout = _with_case_1(tmp_path, record)
    monkeypatch.setattr(submit, "squeue_states", lambda _job: {})
    assert submit.follow(layout, 0, SlurmSettings(poll_seconds=0.1)) == 1
    assert "3/3 case(s) done (2 passed)" in capsys.readouterr().out


def test_sbatch_that_prints_no_job_id_is_a_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1630: sbatch exiting 0 without output is the RuntimeError callers
    catch, not an IndexError."""
    from autoware_carla_scenario.slurm import queue

    monkeypatch.setattr(queue, "require_sbatch", lambda: "sbatch")
    stdout = ""

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(queue.subprocess, "run", run)
    for stdout in ("", "\n \n", ";cluster\n"):
        with pytest.raises(RuntimeError, match="sbatch printed no job id"):
            queue._run_sbatch(["sbatch", "--parsable", "job.sh"])
    stdout = "warning: something\n4242;cluster\n"
    assert queue._run_sbatch(["sbatch", "--parsable", "job.sh"]) == "4242"


def _hanging_tool(bin_dir: Path, name: str) -> Path:
    """A fake SLURM tool *name* that hangs (one process: run() kills it)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    tool = bin_dir / name
    tool.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    tool.chmod(0o755)
    return tool


def test_an_sbatch_that_hangs_is_a_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1786: a hung sbatch is given up on, as the RuntimeError callers
    report, not waited for."""
    from autoware_carla_scenario.slurm import queue

    sbatch = _hanging_tool(tmp_path / "bin", "sbatch")
    monkeypatch.setattr(queue, "require_sbatch", lambda: str(sbatch))
    monkeypatch.setattr(queue, "SBATCH_TIMEOUT_SECONDS", 0.5)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"sbatch timed out after 0\.5 s"):
        queue._run_sbatch(["sbatch", "--parsable", "job.sh"])
    assert time.monotonic() - started < 10.0


def test_a_scancel_that_hangs_is_reported_and_not_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1786: a hung scancel ends cancel_jobs with one line and False."""
    from autoware_carla_scenario.slurm import queue

    bin_dir = tmp_path / "bin"
    _hanging_tool(bin_dir, "scancel")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(queue, "SCANCEL_TIMEOUT_SECONDS", 0.5)
    started = time.monotonic()
    assert queue.cancel_jobs(["41", "42"]) is False
    assert time.monotonic() - started < 10.0
    err = capsys.readouterr().err
    assert err == "[slurm] scancel 41 42 timed out after 0.5 s\n"


@pytest.mark.parametrize("timed_out", [False, True], ids=["", "timed-out"])
@pytest.mark.parametrize("stale", [False, True], ids=["in-flight", "stale"])
def test_wait_neither_follows_nor_collects_a_submission_with_no_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stale: bool,
    timed_out: bool,
) -> None:
    """#1417: a record whose sbatch has not returned (job_id None) is not a
    job that ended: wait asks squeue nothing, collects nothing, and says
    whether to retry (in flight) or that the record is stale.  #1791: one
    whose sbatch timed out says so, in the window or past it, and where to
    look for the job, never to wait for sbatch to return."""
    from autoware_carla_scenario.slurm import cli, collect, submit
    from autoware_carla_scenario.slurm.manifest import (
        SBATCH_IN_FLIGHT_SECONDS,
        write_json,
    )

    layout = _batch_on_disk(tmp_path)
    asked: list[str] = []

    def squeue_states(job: str) -> dict[str, str]:
        asked.append(job)
        return {}

    def no_collect(*_a: Any, **_k: Any) -> int:
        raise AssertionError("collect must not run")

    monkeypatch.setattr(submit, "squeue_states", squeue_states)
    monkeypatch.setattr(collect, "collect", no_collect)
    submitted_at = time.time() - (SBATCH_IN_FLIGHT_SECONDS + 1 if stale else 0)
    record: dict[str, Any] = {
        "index": 1,
        "job_id": None,
        "submitted_at": submitted_at,
        "chunks": [[0]],
    }
    if timed_out:
        record["sbatch_timed_out"] = True
        record["sbatch_timed_out_at"] = "2026-11-06T12:00:00+09:00"
    write_json(layout.submission(1), record)
    for argv in (["wait"], ["wait", "--submission", "1"]):
        assert cli.main([*argv, str(layout.batch_dir)]) == 1
        err = capsys.readouterr().err
        if timed_out:
            assert "submission 1 of this batch has no job id to follow" in err
            assert "sbatch timed out at 2026-11-06T12:00:00+09:00" in err
            assert "squeue --me" in err and "scenario-slurm collect" in err
            assert "try again once it has" not in err and "stale" not in err
            # #1796: resubmit is refused only while the record is in flight.
            assert ("refused until" in err) is not stale
        elif stale:
            assert "submission 1 of this batch has no job" in err
            assert "stale" in err
        else:
            assert "submission 1 of this batch is being submitted" in err
            assert "try again once it has" in err
    assert asked == []
    assert not (layout.batch_dir / "batch_results.json").exists()


@pytest.mark.parametrize("kind", ["in-flight", "timed-out", "stale"])
def test_status_names_no_job_for_a_submission_with_no_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    """#1791: status asks squeue nothing for a record with no job id, never
    prints "job None" nor "ended", and says why it has none."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.manifest import (
        SBATCH_IN_FLIGHT_SECONDS,
        write_json,
    )

    layout = _batch_on_disk(tmp_path)
    asked: list[str] = []

    def squeue_states(job: str) -> dict[str, str]:
        asked.append(job)
        return {}

    monkeypatch.setattr(cli, "squeue_states", squeue_states)
    record: dict[str, Any] = {
        "index": 1,
        "job_id": None,
        "submitted_at": time.time()
        - (SBATCH_IN_FLIGHT_SECONDS + 1 if kind == "stale" else 0),
        "cases": [1, 2],
        "chunks": [[1], [2]],
    }
    if kind == "timed-out":
        record["sbatch_timed_out"] = True
        record["sbatch_timed_out_at"] = "2026-11-06T12:00:00+09:00"
    write_json(layout.submission(1), record)
    assert cli.main(["status", str(layout.batch_dir)]) == 0
    out = capsys.readouterr().out
    assert asked == ["77"]
    (line,) = [x for x in out.splitlines() if x.startswith("  submission 1:")]
    assert "None" not in line and "ended" not in line
    assert line.startswith("  submission 1: 2 task(s), 2 case(s); no job id")
    if kind == "in-flight":
        assert line.endswith("no job id yet (sbatch in flight)")
    elif kind == "timed-out":
        assert "sbatch timed out at 2026-11-06T12:00:00+09:00" in line
        assert "squeue --me" in line
    else:
        assert "stale" in line
    # The submission with a job reads as before.
    assert "  submission 0: job 77, 1 task(s), 0 case(s); in the queue: ended" in out


@pytest.mark.parametrize("kind", ["in-flight", "timed-out", "stale"])
def test_collect_names_no_job_for_a_submission_with_no_job_id(
    tmp_path: Path, kind: str
) -> None:
    """#1796: a case last given to a submission with no job id is reported
    with why it has none (no_job_note), never as "job None"."""
    from autoware_carla_scenario.slurm.collect import collect
    from autoware_carla_scenario.slurm.manifest import (
        SBATCH_IN_FLIGHT_SECONDS,
        no_job_note,
        write_json,
    )

    layout = _batch_on_disk(tmp_path)
    record: dict[str, Any] = {
        "index": 1,
        "job_id": None,
        "submitted_at": time.time()
        - (SBATCH_IN_FLIGHT_SECONDS + 1 if kind == "stale" else 0),
        "chunks": [[0, 1, 2]],
    }
    if kind == "timed-out":
        record["sbatch_timed_out"] = True
        record["sbatch_timed_out_at"] = "2026-11-06T12:00:00+09:00"
    write_json(layout.submission(1), record)
    assert collect(layout, quiet_hint=True) == 1
    summary = json.loads((layout.batch_dir / "batch_results.json").read_text())
    assert len(summary) == 3
    for entry in summary:
        message = entry["message"]
        assert "job None" not in message
        assert "task 0 of submission 1 never started; no job id" in message
        assert no_job_note(record) in message
        if kind == "in-flight":
            assert "sbatch in flight" in message
        elif kind == "timed-out":
            assert "sbatch timed out at 2026-11-06T12:00:00+09:00" in message
        else:
            assert "stale" in message


@pytest.mark.parametrize("old", [False, True], ids=["young", "old"])
def test_follow_says_resubmit_is_refused_only_while_a_timed_out_record_is_young(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], old: bool
) -> None:
    """#1796: past SBATCH_IN_FLIGHT_SECONDS a plain resubmit is taken, so
    follow no longer says it is refused."""
    from autoware_carla_scenario.slurm import submit
    from autoware_carla_scenario.slurm.manifest import (
        SBATCH_IN_FLIGHT_SECONDS,
        sbatch_in_flight,
        write_json,
    )

    layout = _batch_on_disk(tmp_path)
    record: dict[str, Any] = {
        "index": 1,
        "job_id": None,
        "submitted_at": time.time() - (SBATCH_IN_FLIGHT_SECONDS + 60 if old else 1),
        "chunks": [[0]],
        "sbatch_timed_out": True,
        "sbatch_timed_out_at": "2026-11-06T12:00:00+09:00",
    }
    write_json(layout.submission(1), record)
    assert sbatch_in_flight(record) is not old
    assert submit.follow(layout, 1, SlurmSettings()) == 1
    err = capsys.readouterr().err
    assert "its sbatch timed out at 2026-11-06T12:00:00+09:00" in err
    assert f"scenario-slurm resubmit {layout.batch_dir}" in err
    if old:
        assert "refused until" not in err and "--force" not in err
    else:
        assert (
            f"refused until {SBATCH_IN_FLIGHT_SECONDS:.0f} s after the record "
            "was made, unless with --force" in err
        )


def test_of_two_resubmits_at_once_the_later_index_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1352: both passed the check before either reserved an index; the
    one that reserved second sees the first's record and submits nothing."""
    from autoware_carla_scenario.slurm import cli, submit
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path)
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    monkeypatch.setattr(cli, "after_submit", lambda *_: 0)
    monkeypatch.setattr(submit, "job_script", lambda *_a, **_k: "#!/bin/sh\n")
    monkeypatch.setattr(submit, "max_array_size", lambda: 1001)
    sbatch: list[list[str]] = []

    def run_sbatch(command: list[str]) -> str:
        sbatch.append(command)
        return "99"

    monkeypatch.setattr(submit, "_run_sbatch", run_sbatch)
    real_submit = cli.submit_cases

    def submit_cases(*args: Any, **kwargs: Any) -> int:
        # The other resubmit reserved index 1 after this one's check.
        write_json(
            layout.submission(1),
            {"index": 1, "job_id": None, "submitted_at": time.time(), "chunks": []},
        )
        return real_submit(*args, **kwargs)

    monkeypatch.setattr(cli, "submit_cases", submit_cases)
    assert cli.main(["resubmit", str(layout.batch_dir), "--no-wait"]) == 1
    assert "submission(s) 1 of this batch are being submitted" in (
        capsys.readouterr().err
    )
    assert sbatch == []
    assert layout.submissions() == [0, 1]
    assert layout.read_submission(1)["job_id"] is None


def test_batches_started_in_the_same_second_get_directories_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1223: two batches in one second shared ``outputs/<date>/<time>``."""
    import datetime as datetime_module

    from autoware_carla_scenario.examples import run

    class _Frozen(datetime_module.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> _Frozen:  # type: ignore[override]
            return cls(2026, 3, 13, 12, 0, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(datetime_module, "datetime", _Frozen)
    made = [run._make_batch_output_dir() for _ in range(3)]
    day = tmp_path / "outputs" / "2026-03-13"
    # The first keeps the name it always had; the others are numbered.
    assert made == [day / "12-00-00", day / "12-00-00-1", day / "12-00-00-2"]
    assert all(path.is_dir() for path in made)
    # A directory left from an earlier batch is never reused either.
    (made[0] / "batch_results.json").write_text("[]")
    assert run._make_batch_output_dir() == day / "12-00-00-3"


def test_a_new_batch_is_refused_a_directory_that_holds_a_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1223: the second batch overwrote the first's manifest, whose tasks
    then ran the second's cases and deleted each other's outputs."""
    from autoware_carla_scenario.slurm import cli, submit

    layout = _batch_on_disk(tmp_path)
    before = layout.manifest.read_bytes()
    sbatch: list[list[str]] = []

    def run_sbatch(command: list[str]) -> str:
        sbatch.append(command)
        return "1"

    monkeypatch.setattr(submit, "_run_sbatch", run_sbatch)
    with pytest.raises(FileExistsError, match="already holds a SLURM batch"):
        submit.submit_batch(
            scenario_names=["s"],
            cases=[("s", [])],
            labels=["s#1"],
            case_overrides=[[]],
            configs=[],
            scenario_classes=["Scenario"],
            overrides=[],
            expand=False,
            output_dir=layout.batch_dir,
        )
    assert layout.manifest.read_bytes() == before
    assert sbatch == []
    assert layout.submissions() == [0]
    # A resubmit adds a submission to that same manifest, as before.
    monkeypatch.setattr(cli, "squeue_states", lambda _job: {})
    assert cli.main(["resubmit", str(layout.batch_dir), "--dry-run"]) == 0
    assert "Resubmitting 3 case(s)" in capsys.readouterr().out


def test_port_locks_go_to_a_node_wide_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port_base: int
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
    slot = acquire_slot(port_base, RANGE_SLOTS, start=0)
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


def test_a_failing_task_port_propagates_and_releases_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm import worker
    from autoware_carla_scenario.slurm.manifest import write_json

    layout = _batch_on_disk(tmp_path, cases=1)
    manifest = layout.read_manifest()
    manifest["cases"][0].update(scenario="x", overrides=[], stem="S")
    write_json(layout.manifest, manifest)
    released: list[bool] = []
    slot = SimpleNamespace(
        rpc=31500, release=lambda keep_lock=False: released.append(keep_lock)
    )

    def task_port(*_args: object) -> Path:
        raise OSError("no port file")

    monkeypatch.setattr(worker, "acquire_slot", lambda *_a, **_k: slot)
    monkeypatch.setattr(BatchLayout, "task_port", lambda self, *a: task_port(*a))
    monkeypatch.setattr(worker, "kill_carla_on_port", lambda _port: [])
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        assert worker.run_task(layout.manifest, 0, 0) == 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    # The original error is recorded, not an UnboundLocalError from the cleanup.
    status = layout.read_task(0, 0)
    assert status is not None
    assert status["state"] == "failed"
    assert "OSError: no port file" in status["error"]
    assert "UnboundLocalError" not in status["traceback"]
    assert released == [False]


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
    _status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm))
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

    recorded_start = _recorded_starts(monkeypatch)

    def start() -> None:
        started.append(recorded_start(lock, spacing))

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


def _stale_lock(path: Path, token: str = "dead-task") -> None:
    """A lock file a task killed an hour ago left."""
    path.write_text(token)
    old = time.time() - 3600
    os.utime(path, (old, old))


def _takeover_mutex(path: Path, generation: int) -> Path:
    """Generation *generation* of the takeover mutex of the lock at *path*."""
    st = path.stat()
    return path.with_name(
        f"{path.name}.takeover.{st.st_ino}-{st.st_mtime_ns}.{generation}"
    )


def _herd(path: Path, waiters: int, worker: Any) -> tuple[int, list[str]]:
    """*waiters* tasks taking the lock at *path* at once: the most that held
    it together, and their tokens."""
    import threading

    order = threading.Lock()
    holding: list[str] = []
    most = [0]
    tokens: list[str] = []
    go = threading.Barrier(waiters, timeout=10)

    def hold() -> None:
        go.wait()
        with worker._exclusive_file(path, stale_after=60.0):
            name = threading.current_thread().name
            with order:
                holding.append(name)
                most[0] = max(most[0], len(holding))
            tokens.append(path.read_text())
            time.sleep(0.02)
            with order:
                holding.remove(name)

    threads = [threading.Thread(target=hold, name=f"t{i}") for i in range(waiters)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not any(thread.is_alive() for thread in threads)
    return most[0], tokens


def _count_removals(monkeypatch: pytest.MonkeyPatch, worker: Any) -> list[str]:
    """The tokens of the locks taken over as stale, as they are."""
    real = worker._remove_stale
    removed: list[str] = []

    def remove(lock: Path, stale: Any) -> bool:
        done = bool(real(lock, stale))
        if done:
            removed.append(stale[3])
        return done

    monkeypatch.setattr(worker, "_remove_stale", remove)
    return removed


def test_waiters_taking_over_one_stale_lock_do_not_both_hold_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1360: A and B both judge the lock stale before either takes it over;
    B takes over only once A holds a fresh lock, and holds its takeover
    mutex while a third task C polls.  B leaves A's lock alone, and no two
    tasks hold the lock at once."""
    import threading

    from autoware_carla_scenario.slurm import worker

    path = tmp_path / "start.lock.excl"
    _stale_lock(path)
    real_take_over = worker._take_over
    real_create = worker._create_lock
    judged = threading.Barrier(2, timeout=10)
    a_holds = threading.Event()
    c_polls = threading.Event()
    b_done = threading.Event()
    arrivals: list[str] = []
    order = threading.Lock()
    local = threading.local()

    def take_over(*args: Any) -> bool:
        if threading.current_thread().name == "c" or getattr(local, "done", False):
            return bool(real_take_over(*args))
        local.done = True
        judged.wait()  # both judged the same stale lock
        with order:
            arrivals.append(threading.current_thread().name)
            second = len(arrivals) == 2
        if not second:
            return bool(real_take_over(*args))
        assert a_holds.wait(10) and c_polls.wait(10)
        try:
            return bool(real_take_over(*args))
        finally:
            b_done.set()

    def create(lock: Path, token: str) -> bool:
        made = bool(real_create(lock, token))
        if made and ".takeover." in lock.name and arrivals[1:] == ["b"]:
            time.sleep(0.3)  # B mid-takeover, holding the mutex: C polls
        return made

    monkeypatch.setattr(worker, "_take_over", take_over)
    monkeypatch.setattr(worker, "_create_lock", create)
    removed = _count_removals(monkeypatch, worker)
    holding: list[str] = []
    most = [0]
    tokens: dict[str, str] = {}

    def hold() -> None:
        name = threading.current_thread().name
        with worker._exclusive_file(path, stale_after=60.0):
            with order:
                holding.append(name)
                most[0] = max(most[0], len(holding))
            tokens[name] = path.read_text()
            if name == arrivals[0]:
                a_holds.set()
                assert b_done.wait(10)
                time.sleep(0.1)
                # Still in place, still this task's.
                assert path.read_text() == tokens[name]
            with order:
                holding.remove(name)

    threads = [threading.Thread(target=hold, name=name) for name in "ab"]
    for thread in threads:
        thread.start()
    assert a_holds.wait(10)
    c = threading.Thread(target=hold, name="c")
    c.start()
    time.sleep(0.1)
    c_polls.set()
    for thread in [*threads, c]:
        thread.join(30)
    assert not any(thread.is_alive() for thread in [*threads, c])
    assert sorted(arrivals) == ["a", "b"]
    assert most[0] == 1
    assert sorted(tokens) == ["a", "b", "c"] and len(set(tokens.values())) == 3
    assert removed == ["dead-task"]  # the stale lock, and nothing else
    assert list(tmp_path.iterdir()) == []  # no mutex left


def test_a_takeover_waits_for_a_live_mutex_and_clears_a_stale_one(
    tmp_path: Path,
) -> None:
    import threading

    from autoware_carla_scenario.slurm import worker

    path = tmp_path / "start.lock.excl"
    _stale_lock(path)
    mutex = _takeover_mutex(path, 1)
    mutex.write_text("other-host:1:taking-over")  # another waiter's, live
    acquired = threading.Event()

    def hold() -> None:
        with worker._exclusive_file(path, stale_after=60.0):
            acquired.set()

    thread = threading.Thread(target=hold)
    thread.start()
    assert not acquired.wait(0.5)
    assert path.read_text() == "dead-task"
    assert mutex.read_text() == "other-host:1:taking-over"
    # Its holder was killed: stale after a while, and superseded.
    old = time.time() - 3600
    os.utime(mutex, (old, old))
    assert acquired.wait(10)
    thread.join(10)
    assert list(tmp_path.iterdir()) == []


def test_a_herd_on_the_mutex_of_a_killed_waiter_takes_the_lock_over_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1360: a waiter killed holding the takeover mutex; the waiters behind
    it all find the mutex stale at the same moment, and only one of them
    takes the lock over."""
    from autoware_carla_scenario.slurm import worker

    monkeypatch.setattr(worker, "_TAKEOVER_STALE_AFTER", 0.5)
    path = tmp_path / "start.lock.excl"
    _stale_lock(path)
    _takeover_mutex(path, 1).write_text("killed-host:1:waiter")  # fresh
    removed = _count_removals(monkeypatch, worker)
    most, tokens = _herd(path, 6, worker)
    assert most == 1
    assert len(set(tokens)) == 6 and "dead-task" not in tokens
    assert removed == ["dead-task"]
    assert list(tmp_path.iterdir()) == []


def test_a_resubmission_into_a_stale_lock_and_aged_mutexes_takes_it_over_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1360: scancel during a takeover left a stale lock and its mutex, a
    generation already superseded once; a resubmission's tasks all arrive
    to find them an hour old."""
    from autoware_carla_scenario.slurm import worker

    path = tmp_path / "start.lock.excl"
    _stale_lock(path)
    old = time.time() - 3600
    for generation in (1, 2):
        mutex = _takeover_mutex(path, generation)
        mutex.write_text(f"killed-host:{generation}:waiter")
        os.utime(mutex, (old, old))
    removed = _count_removals(monkeypatch, worker)
    most, tokens = _herd(path, 8, worker)
    assert most == 1
    assert len(set(tokens)) == 8
    assert removed == ["dead-task"]
    assert list(tmp_path.iterdir()) == []


def test_a_gap_below_a_dead_takeover_generation_is_climbed_past(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1360: generation 1's winner stalled, found generation 2 and backed
    off; generation 2's winner was killed.  A stale lock, a dead generation
    2 and no generation 1: a waiter neither hangs nor spins on them."""
    import threading

    from autoware_carla_scenario.slurm import worker

    monkeypatch.setattr(worker, "_TAKEOVER_STALE_AFTER", 0.3)
    path = tmp_path / "start.lock.excl"
    _stale_lock(path)
    dead = _takeover_mutex(path, 2)
    dead.write_text("killed-host:2:waiter")
    old = time.time() - 3600
    os.utime(dead, (old, old))
    real_create = worker._create_lock
    created: list[str] = []

    def create(lock: Path, token: str) -> bool:
        created.append(lock.name)
        return bool(real_create(lock, token))

    monkeypatch.setattr(worker, "_create_lock", create)
    acquired = threading.Event()

    def hold() -> None:
        with worker._exclusive_file(path, stale_after=60.0):
            acquired.set()

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(10)
    thread.join(10)
    # Generation 1 is won once, left while it is live, then climbed past.
    attempts = [name for name in created if ".takeover." in name]
    assert len(attempts) < 20, attempts
    assert list(tmp_path.iterdir()) == []


def test_releasing_a_lock_taken_over_leaves_the_new_holders(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.slurm import worker

    path = tmp_path / "start.lock.excl"
    with worker._exclusive_file(path, stale_after=60.0):
        mine = path.read_text()
        # Taken over as stale meanwhile, and held by another task now.
        path.unlink()
        path.write_text("other-host:1:another-task")
    assert path.read_text() == "other-host:1:another-task"
    assert mine != "other-host:1:another-task"
    # Its own lock is removed.
    path.unlink()
    with worker._exclusive_file(path, stale_after=60.0):
        assert path.exists()
    assert not path.exists()


@pytest.mark.parametrize("skew", [3600.0, -3600.0])
def test_a_lock_is_judged_stale_by_the_file_servers_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skew: float
) -> None:
    """#1360: this node's clock an hour ahead (or behind) the file server's
    neither breaks a live lock nor keeps a stale one."""
    import threading

    from autoware_carla_scenario.slurm import worker

    class _SkewedClock:
        def __getattr__(self, name: str) -> Any:
            return getattr(time, name)

        def time(self) -> float:
            return time.time() + skew

    monkeypatch.setattr(worker, "time", _SkewedClock())
    path = tmp_path / "start.lock.excl"
    # A live lock, just made: not taken over however far ahead this clock is.
    path.write_text("other-host:1:live-task")
    acquired = threading.Event()

    def hold() -> None:
        with worker._exclusive_file(path, stale_after=5.0):
            acquired.set()

    thread = threading.Thread(target=hold)
    thread.start()
    assert not acquired.wait(0.5)
    assert path.read_text() == "other-host:1:live-task"
    path.unlink()  # its holder releases it
    assert acquired.wait(10)
    thread.join(10)
    # A lock an hour old is stale, however far behind this clock is.
    _stale_lock(path)
    with worker._exclusive_file(path, stale_after=5.0):
        assert path.read_text() != "dead-task"
    assert list(tmp_path.iterdir()) == []


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
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm), wait=False)
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


def test_an_unwritable_collect_script_is_reported_not_raised(
    fake_slurm: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The array is submitted before the collect job's script is written: a
    failure to write it is reported like a refused sbatch (#1159)."""
    from autoware_carla_scenario.slurm import submit

    real = submit._write_script

    def write(path: Path, text: str) -> None:
        if path.name.endswith("-collect.sh"):
            raise OSError(28, "No space left on device")
        real(path, text)

    fake_slurm.monkeypatch.setattr(submit, "_write_script", write)
    status, batch = _run_batch(fake_slurm, "slurm", _slurm(fake_slurm), wait=False)
    captured = capsys.readouterr()
    assert status == 1
    submission = BatchLayout(batch).read_submission(0)
    assert submission["job_id"] and submission["collect_job_id"] is None
    assert "No space left on device" in submission["collect_error"]
    assert "scenario-slurm collect" in captured.err
    assert "Traceback" not in captured.err
    _wait_for(lambda: _jobs_done(fake_slurm), 120, "the array")


def test_interrupt_waits_for_the_cancelled_tasks_before_collecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.slurm import collect, queue, submit

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
    # The wait after scancel polls through slurm.queue's own (#1634).
    monkeypatch.setattr(queue, "squeue_states", squeue)
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
    from autoware_carla_scenario.slurm import queue

    clock = [0.0]
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        queue.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)
    )
    monkeypatch.setattr(queue, "squeue_states", lambda _job: {0: "COMPLETING"})
    assert queue._wait_until_gone("77", 60.0, 15.0) is False
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
    from autoware_carla_scenario.slurm.submit import _collect_command, collect_script

    layout = BatchLayout(Path("/work/outputs/2026-10-19/12-00-00"))
    manifest = {"python": "/usr/bin/python3", "cwd": "/work", "env": {}}
    settings = SlurmSettings.from_mapping({"python": ".venv/bin/python"})
    script = collect_script(manifest, settings, layout).splitlines()
    assert script[-1] == (
        "exec .venv/bin/python -m autoware_carla_scenario.slurm.cli collect "
        "/work/outputs/2026-10-19/12-00-00"
    )
    assert "cd /work || exit 1" in script
    path = layout.root / "jobs" / "0-collect.sh"
    command = _collect_command(layout, manifest, settings, "41", path)
    assert "--chdir=/work" in command
    assert not any(arg.startswith("--wrap") for arg in command)


def _environment_lines(script: str) -> list[str]:
    """The lines of *script* that set the job's environment up, in order."""
    return [
        line
        for line in script.splitlines()
        if line.startswith(("export ", "module ", "source ", "set +u", "set -u"))
        or line.startswith("# launcher.")
    ]


def test_without_remote_cwd_the_collect_job_sets_up_the_environment_as_a_task(
    tmp_path: Path,
) -> None:
    """#1159: the collect job ran launcher.python without launcher.setup and
    launcher.env, which a cluster that loads modules or activates a venv in
    setup needs to find it."""
    from autoware_carla_scenario.slurm.submit import (
        _collect_command,
        collect_script,
        job_script,
    )

    batch = tmp_path / "work dir" / "outputs" / "d" / "t"
    layout = BatchLayout(batch)
    manifest = {
        "python": "/submit/python",
        "cwd": str(tmp_path / "work dir"),
        "env": {"PATH": "/usr/bin", "CARLA_ROOT": "/opt/carla's root"},
    }
    settings = SlurmSettings.from_mapping(
        {
            "python": "python3",
            "export": "NONE",
            "setup": [
                "module load python/3.12",
                "source '/shared/my venv/bin/activate'",
            ],
            "env": {"ONEPLANNER_X": 'say "hi"', "A": "1"},
        }
    )
    task = job_script(layout, manifest, settings, 0)
    script = collect_script(manifest, settings, layout)
    assert _environment_lines(script) == _environment_lines(task)
    lines = script.splitlines()
    assert "module load python/3.12" in lines
    assert "source '/shared/my venv/bin/activate'" in lines
    assert "export ONEPLANNER_X='say \"hi\"'" in lines
    assert lines.index("export ONEPLANNER_X='say \"hi\"'") < lines.index(
        f"exec python3 -m autoware_carla_scenario.slurm.cli collect '{batch}'"
    )
    assert f"cd '{tmp_path / 'work dir'}' || exit 1" in lines
    completed = subprocess.run(["bash", "-n"], input=script, text=True, check=False)
    assert completed.returncode == 0

    path = layout.root / "jobs" / "0-collect.sh"
    command = _collect_command(layout, manifest, settings, "41", path)
    assert "--dependency=afterany:41" in command
    assert f"--output={layout.logs}/collect-%j.out" in command
    assert "--export=NONE" in command
    assert command[-1] == str(path)
    with pytest.raises(ValueError, match="runs a script"):
        _collect_command(layout, manifest, settings, "41")


def test_without_remote_cwd_the_collect_script_runs_setup_before_python(
    tmp_path: Path,
) -> None:
    """The collect script, run as the job runs it, reaches the interpreter only
    once launcher.setup and launcher.env have run."""
    from autoware_carla_scenario.slurm.submit import collect_script

    work = tmp_path / "work dir"
    bin_dir = tmp_path / "setup's bin"
    bin_dir.mkdir()
    record = tmp_path / "record"
    fake_python = bin_dir / "fake-python"
    fake_python.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$PWD" "$ONEPLANNER_X" "$CARLA_ROOT" "$@" > {shlex.quote(str(record))}\n'
    )
    fake_python.chmod(0o755)
    layout = BatchLayout(work / "outputs" / "d" / "t")
    layout.root.mkdir(parents=True)
    manifest = {"python": "/x", "cwd": str(work), "env": {"CARLA_ROOT": "/c"}}
    settings = SlurmSettings.from_mapping(
        {
            "python": "fake-python",
            "setup": [f'export PATH={shlex.quote(str(bin_dir))}:"$PATH"'],
            "env": {"ONEPLANNER_X": "it's x"},
        }
    )
    script = layout.root / "collect.sh"
    script.write_text(collect_script(manifest, settings, layout))
    completed = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert record.read_text().splitlines() == [
        str(work),
        "it's x",
        "/c",
        "-m",
        "autoware_carla_scenario.slurm.cli",
        "collect",
        str(layout.batch_dir),
    ]


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
    from autoware_carla_scenario.slurm import cli, scripts

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
    mp.setattr(scripts, "REMOTE_EARLY_OUTPUT_DIR", str(node_tmp))
    mp.setenv("HOME", str(tmp / "submit_home"))
    mp.chdir(submit_side)
    output = submit_side / "outputs" / "2026-10-19" / "12-00-00"
    output.mkdir(parents=True)
    mp.setattr(fake_slurm.run, "_make_batch_output_dir", lambda: output)

    overrides = _slurm(
        fake_slurm,
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


@pytest.mark.parametrize("dry_run", [False, True])
def test_resubmit_with_nothing_to_resubmit_collects_only_without_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    """``resubmit --wait`` with nothing to resubmit collects at once; with
    ``--dry-run`` too it collects nothing and writes nothing (#1768)."""
    from autoware_carla_scenario.slurm import cli
    from autoware_carla_scenario.slurm.manifest import MANIFEST_VERSION

    tmp_path = tmp_path.resolve()
    layout = BatchLayout(tmp_path)
    layout.manifest.parent.mkdir(parents=True)
    layout.manifest.write_text(
        json.dumps({"version": MANIFEST_VERSION, "cases": [], "launcher": {}})
    )
    collected: list[Path] = []

    def fake_collect(target: BatchLayout, **_: Any) -> int:
        collected.append(target.batch_dir)
        return 1

    monkeypatch.setattr(cli, "collect", fake_collect)
    argv = ["resubmit", str(tmp_path), "--wait", *(["--dry-run"] if dry_run else [])]
    code = cli.main(argv)
    assert "Nothing to resubmit" in capsys.readouterr().out
    if dry_run:
        assert code == 0
        assert collected == []
        assert not (tmp_path / "batch_results.json").exists()
        assert not (layout.root / "collected.json").exists()
    else:
        assert code == 1
        assert collected == [tmp_path]
