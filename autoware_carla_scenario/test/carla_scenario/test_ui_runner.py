"""ui.runner: its helpers, as spelled for the Codon check (#132), and how a
run starts (#664)."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner
from autoware_carla_scenario.ui.models import RunProgress


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 10, 12, 9, 5, 3),
        datetime(2026, 1, 2, 0, 0, 0),
        datetime(1999, 12, 31, 23, 59, 59, 999999),
        datetime(1000, 1, 1),
        datetime(9999, 12, 31, 23, 59, 59),
        datetime.now(),  # noqa: DTZ005
    ],
)
def test_the_multirun_dir_is_what_strftime_makes(now: datetime) -> None:
    # The f-string the runner used to build it with: the same string, with
    # zero-padded fields (years before 1000 differ between Pythons' strftime).
    assert runner._multirun_dir(now) == f"multirun/{now:%Y-%m-%d}/{now:%H-%M-%S}"


@pytest.mark.parametrize(
    ("stdout", "stderr", "written"),
    [
        ("out", "err", "out\n\nerr"),
        ("out", "", "out"),
        (None, "err", "err"),
        ("", None, None),
    ],
)
def test_the_raw_output_joins_the_streams_written(
    stdout: str | None, stderr: str | None, written: str | None, tmp_path: Path
) -> None:
    result = subprocess.CompletedProcess(["true"], 0, stdout, stderr)
    job_dir = tmp_path / "multirun" / "0"
    runner._save_raw_output(result, job_dir)
    log = job_dir / runner.RAW_OUTPUT_LOG
    if written is None:
        assert not log.exists()
    else:
        assert log.read_text(encoding="utf-8") == written


def test_a_run_appends_the_extra_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner._run_worker([["scenario=a"], ["scenario=b", "x=1"]], None, ["y=2"], False)
    assert commands == [
        ["uv", "run", "scenario", "scenario=a", "y=2"],
        ["uv", "run", "scenario", "scenario=b", "x=1", "y=2"],
    ]
    progress = runner.get_progress()
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (2, 2, "done")
    assert not runner.is_running()


# ── Starting a run (#664) ──────────────────────────────────────────────────


class _UnstartedThread:
    """A worker thread the OS has not scheduled yet: start() runs nothing.

    The test runs the worker itself, with :meth:`run`, when it chooses to.
    """

    made: list[_UnstartedThread] = []

    def __init__(self, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
        self.target = target
        self.args = args
        self.started = False
        _UnstartedThread.made.append(self)

    def start(self) -> None:
        self.started = True

    def run(self) -> None:
        self.target(*self.args)


@pytest.fixture
def unstarted_threads(monkeypatch: pytest.MonkeyPatch) -> list[_UnstartedThread]:
    """The runner's worker threads, none of them running until the test says.

    The previous run is over and its progress is ``done``.
    """
    monkeypatch.setattr(_UnstartedThread, "made", [])
    monkeypatch.setattr(runner.threading, "Thread", _UnstartedThread)
    monkeypatch.setattr(runner, "_running", False)
    monkeypatch.setattr(
        runner,
        "_progress",
        RunProgress(current=1, total=1, scenario_name="", status="done"),
    )

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    return _UnstartedThread.made


def test_a_started_run_is_running_before_its_worker_is(
    unstarted_threads: list[_UnstartedThread],
) -> None:
    started = runner.start_run([["scenario=a"], ["scenario=b"]])
    assert [t.started for t in unstarted_threads] == [True]
    running, progress = runner.get_status()
    # What /api/run/progress streams now: the new run, not the last one's done.
    assert running
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (0, 2, "running")
    assert started is True


def test_a_second_run_is_refused_until_the_first_ends(
    unstarted_threads: list[_UnstartedThread],
) -> None:
    first = runner.start_run([["scenario=a"]])
    second = runner.start_run([["scenario=b"]])
    assert len(unstarted_threads) == 1
    assert (first, second) == (True, False)
    progress = runner.get_progress()
    assert progress is not None and progress.status == "running"

    unstarted_threads[0].run()
    assert not runner.is_running()
    progress = runner.get_progress()
    assert progress is not None and progress.status == "done"

    assert runner.start_run([["scenario=b"]]) is True
    assert len(unstarted_threads) == 2


def test_the_api_answers_409_to_a_refused_run(
    unstarted_threads: list[_UnstartedThread],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    client = TestClient(app_module.app)

    first = client.post("/api/run", json={"scenario": "a"})
    assert first.status_code == 200
    assert first.json() == {"status": "started"}

    second = client.post("/api/run", json={"scenario": "b"})
    assert second.status_code == 409
    assert second.json()["status"] == "busy"
    assert len(unstarted_threads) == 1
    assert unstarted_threads[0].args[0] == [["scenario=a"]]

    status = client.get("/api/run/status").json()
    assert status["running"] is True
    assert status["progress"]["status"] == "running"


def test_a_run_whose_thread_does_not_start_is_over(
    unstarted_threads: list[_UnstartedThread],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def start(self: _UnstartedThread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(_UnstartedThread, "start", start)
    with pytest.raises(RuntimeError):
        runner.start_run([["scenario=a"]])
    running, progress = runner.get_status()
    # The stream ends instead of following a run that never runs.
    assert not running
    assert progress is not None and progress.status == "done"


def test_a_worker_that_raises_ends_its_run(
    unstarted_threads: list[_UnstartedThread],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Outside a job's launch and collection, which fail only their job.
    def extract(overrides: list[str]) -> str:
        raise ValueError("boom")

    monkeypatch.setattr(runner, "_extract_scenario_name", extract)
    assert runner.start_run([["scenario=a"], ["scenario=b"]]) is True
    with pytest.raises(ValueError):
        unstarted_threads[0].run()
    running, progress = runner.get_status()
    assert not running
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (0, 2, "done")
    assert runner.start_run([["scenario=b"]]) is True


# ── Output that is not UTF-8 (#1073) ───────────────────────────────────────


@pytest.fixture
def fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``uv`` on PATH that prints bytes that are not UTF-8 and logs its jobs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "launched.log"
    script = bin_dir / "uv"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$3" >> "{log}"\n'
        "printf 'out \\377\\376 end\\n'\n"
        "printf 'err \\377\\n' >&2\n"
        'case "$3" in scenario=b) exit 1;; esac\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def _record_progress(monkeypatch: pytest.MonkeyPatch) -> list[RunProgress]:
    published: list[RunProgress] = []
    set_progress = runner._set_progress

    def record(p: RunProgress) -> None:
        published.append(p)
        set_progress(p)

    monkeypatch.setattr(runner, "_set_progress", record)
    return published


def test_output_that_is_not_utf8_runs_every_job(
    fake_uv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _record_progress(monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    runner._run_worker([["scenario=a"], ["scenario=b"], ["scenario=c"]], base, [], True)
    launched = fake_uv.read_text(encoding="utf-8").splitlines()
    assert launched == ["scenario=a", "scenario=b", "scenario=c"]
    finished = [(p.scenario_name, p.status) for p in published if p.status != "running"]
    assert finished == [
        ("a", "passed"),
        ("b", "failed"),
        ("c", "passed"),
        ("", "done"),
    ]
    assert not runner.is_running()
    # The undecodable bytes are replaced in the raw output kept for each job.
    logs = sorted((base / "multirun").glob(f"*/*/*/{runner.RAW_OUTPUT_LOG}"))
    assert len(logs) == 3
    assert logs[0].read_text(encoding="utf-8") == (
        "out \ufffd\ufffd end\n\n\nerr \ufffd\n"
    )


def test_a_job_that_cannot_be_collected_fails_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _record_progress(monkeypatch)
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if "scenario=a" in cmd:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner._run_worker([["scenario=a"], ["scenario=b"]], None, [], False)
    assert len(commands) == 2
    finished = [(p.scenario_name, p.status) for p in published if p.status != "running"]
    assert finished == [("a", "failed"), ("b", "passed"), ("", "done")]
    assert not runner.is_running()
