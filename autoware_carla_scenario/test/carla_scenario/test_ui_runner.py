"""ui.runner: its helpers, as spelled for the Codon check (#132), how a run
starts (#664), and how it stops (#1351)."""

from __future__ import annotations

import os
import time
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


def test_a_job_writes_both_streams_to_its_log_in_order(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    code = runner._run_job(
        ["sh", "-c", "echo out; echo err >&2; echo more; exit 3"], None, log
    )
    assert code == 3
    assert log.read_text(encoding="utf-8") == "out\nerr\nmore\n"


def test_a_run_appends_the_extra_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], cwd: str | None, log: Path) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(runner, "_run_job", run)
    runner._run_worker([["scenario=a"], ["scenario=b", "x=1"]], None, ["y=2"], False)
    assert commands == [
        ["uv", "run", "scenario", "scenario=a", "y=2"],
        ["uv", "run", "scenario", "scenario=b", "x=1", "y=2"],
    ]
    progress = runner.get_progress()
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (2, 2, "done")
    assert not runner.is_running()


# ── A grouped run's own multirun directory (#1359) ─────────────────────────


def test_a_multirun_dir_is_never_shared(tmp_path: Path) -> None:
    now = datetime(2026, 3, 13, 12, 0, 0)
    made = [runner._reserve_multirun_dir(tmp_path, now) for _ in range(3)]
    # The first keeps the name it always had; the others are numbered, as a
    # batch's output directory is.
    assert made == [
        "multirun/2026-03-13/12-00-00",
        "multirun/2026-03-13/12-00-00-1",
        "multirun/2026-03-13/12-00-00-2",
    ]
    assert all((tmp_path / name).is_dir() for name in made)
    # A directory a CLI sweep made in the same second is never reused either.
    (tmp_path / "multirun/2026-03-13/12-00-00-3/0").mkdir(parents=True)
    assert runner._reserve_multirun_dir(tmp_path, now) == (
        "multirun/2026-03-13/12-00-00-4"
    )


def test_grouped_runs_in_one_second_get_their_own_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> _Frozen:  # type: ignore[override]
            return cls(2026, 3, 13, 12, 0, 0)

    commands: list[list[str]] = []

    def run(cmd: list[str], cwd: str | None, log: Path) -> int:
        commands.append(cmd)
        log.write_text("out", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "datetime", _Frozen)
    monkeypatch.setattr(runner, "_run_job", run)
    runner._run_worker([["scenario=a"]], tmp_path, [], True)
    runner._run_worker([["scenario=b"]], tmp_path, [], True)
    assert [cmd[-1] for cmd in commands] == [
        "hydra.run.dir=multirun/2026-03-13/12-00-00/0",
        "hydra.run.dir=multirun/2026-03-13/12-00-00-1/0",
    ]
    for session in ("12-00-00", "12-00-00-1"):
        log = tmp_path / "multirun/2026-03-13" / session / "0" / runner.RAW_OUTPUT_LOG
        assert log.read_text(encoding="utf-8") == "out"


def test_the_scanner_lists_a_numbered_multirun_dir(tmp_path: Path) -> None:
    from autoware_carla_scenario.ui import scanner

    now = datetime(2026, 3, 13, 12, 0, 0)
    for _ in range(2):
        job = tmp_path / runner._reserve_multirun_dir(tmp_path, now) / "0"
        job.mkdir()
        (job / "Demo_result.json").write_text(
            '{"passed": true, "scenario": "Demo"}', encoding="utf-8"
        )
    scanner.clear_cache()
    sessions = scanner.scan_sessions(tmp_path)
    assert sorted((s.session_type, s.time) for s in sessions) == [
        ("multirun", "12-00-00"),
        ("multirun", "12-00-00-1"),
    ]


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

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return self.started


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

    def run(cmd: list[str], cwd: str | None, log: Path) -> int:
        return 0

    monkeypatch.setattr(runner, "_run_job", run)
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
    client = TestClient(app_module.app, base_url="http://127.0.0.1")

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


def test_a_run_whose_multirun_dir_cannot_be_made_fails(
    unstarted_threads: list[_UnstartedThread],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The `done` says why, so the viewer shows a failure, not "Completed!"
    # (#2183); no job is launched, and the next run may start.
    def reserve(base: Path, now: datetime) -> str:
        raise PermissionError(13, "Permission denied", str(base / "multirun"))

    def launch(cmd: list[str], cwd: str | None, output: Any) -> None:
        raise AssertionError("no job runs")

    monkeypatch.setattr(runner, "_reserve_multirun_dir", reserve)
    monkeypatch.setattr(runner, "_launch", launch)
    assert runner.start_run([["scenario=a"], ["scenario=b"]], tmp_path, None, True)
    with caplog.at_level("ERROR", logger=runner.__name__):
        unstarted_threads[0].run()
    running, progress = runner.get_status()
    assert not running
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (0, 2, "done")
    assert progress.message.startswith("The run failed: could not create")
    assert "Permission denied" in progress.message
    assert any("multirun directory" in r.getMessage() for r in caplog.records)
    assert runner.start_run([["scenario=b"]]) is True


def test_a_completed_run_has_no_message(
    unstarted_threads: list[_UnstartedThread],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_launch", lambda cmd, cwd, output: None)
    assert runner.start_run([["scenario=a"]]) is True
    unstarted_threads[0].run()
    progress = runner.get_progress()
    assert progress is not None and progress.status == "done"
    assert progress.message == ""


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
    # The raw output is kept for each job as it was written (both streams,
    # in order), its undecodable bytes replaced as the viewer reads it.
    logs = sorted((base / "multirun").glob(f"*/*/*/{runner.RAW_OUTPUT_LOG}"))
    assert len(logs) == 3
    assert logs[0].read_bytes() == b"out \xff\xfe end\nerr \xff\n"
    assert runner._read_raw_log(logs[0].parent) == "out \ufffd\ufffd end\nerr \ufffd\n"


def test_a_job_that_cannot_be_collected_fails_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _record_progress(monkeypatch)
    commands: list[list[str]] = []

    def run(cmd: list[str], cwd: str | None, log: Path) -> int:
        commands.append(cmd)
        if "scenario=a" in cmd:
            raise OSError("the job cannot be launched")
        return 0

    monkeypatch.setattr(runner, "_run_job", run)
    runner._run_worker([["scenario=a"], ["scenario=b"]], None, [], False)
    assert len(commands) == 2
    finished = [(p.scenario_name, p.status) for p in published if p.status != "running"]
    assert finished == [("a", "failed"), ("b", "passed"), ("", "done")]
    assert not runner.is_running()


# ── Stopping a run (#1351) ─────────────────────────────────────────────────


def test_a_stop_between_jobs_starts_no_further_job(
    unstarted_threads: list[_UnstartedThread], monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _record_progress(monkeypatch)
    commands: list[list[str]] = []

    def run(cmd: list[str], cwd: str | None, log: Path) -> int:
        commands.append(cmd)
        # The server shuts down while the first job runs.
        runner.stop_run()
        return 0

    monkeypatch.setattr(runner, "_run_job", run)
    assert runner.start_run([["scenario=a"], ["scenario=b"], ["scenario=c"]])
    unstarted_threads[0].run()
    assert [c[3] for c in commands] == ["scenario=a"]
    finished = [(p.scenario_name, p.status) for p in published if p.status != "running"]
    assert finished == [("a", "passed")]
    running, progress = runner.get_status()
    assert not running
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (1, 3, "done")
    # The next run is not stopped by the last one's stop.
    commands.clear()

    def run_all(cmd: list[str], cwd: str | None, log: Path) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(runner, "_run_job", run_all)
    assert runner.start_run([["scenario=d"], ["scenario=e"]])
    unstarted_threads[1].run()
    assert [c[3] for c in commands] == ["scenario=d", "scenario=e"]


def test_stopping_without_a_run_does_nothing() -> None:
    assert not runner.is_running()
    runner.stop_run(timeout=0.1)
    assert not runner._stop_requested()


def _alive(pid: int) -> bool:
    """Whether process *pid* still runs (a zombie, exited unreaped, does not)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_for(condition: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def test_a_stop_terminates_the_running_jobs_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `uv` whose job leaves a child that ignores SIGTERM, as a CARLA or an
    # Autoware it started might: the group is killed after the timeout.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launched = tmp_path / "launched.log"
    child_pid = tmp_path / "child.pid"
    script = bin_dir / "uv"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$3" >> "{launched}"\n'
        'sh -c \'trap "" TERM; echo $$ > "$0"; exec sleep 30\' '
        f'"{child_pid}" &\n'
        "wait\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(runner, "_running", False)
    monkeypatch.setattr(runner, "_progress", None)
    published = _record_progress(monkeypatch)

    assert runner.start_run([["scenario=a"], ["scenario=b"]], tmp_path)
    try:
        assert _wait_for(lambda: child_pid.is_file() and child_pid.read_text())
        pid = int(child_pid.read_text())
        assert _alive(pid)
        started = time.monotonic()
        runner.stop_run(timeout=1.0)
        assert time.monotonic() - started < 5.0
        assert _wait_for(lambda: not _alive(pid))
        assert _wait_for(lambda: not runner.is_running())
    finally:
        runner.stop_run(timeout=1.0)
    assert launched.read_text(encoding="utf-8").splitlines() == ["scenario=a"]
    finished = [(p.scenario_name, p.status) for p in published if p.status != "running"]
    assert finished == [("a", "failed")]
    progress = runner.get_progress()
    assert progress is not None
    assert (progress.current, progress.total, progress.status) == (1, 2, "done")


def test_the_progress_stream_ends_when_the_server_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def get_progress() -> RunProgress:
        calls.append(1)
        if len(calls) == 2:
            monkeypatch.setattr(app_module, "_shutting_down", True)
        return RunProgress(current=1, total=3, scenario_name="a", status="running")

    monkeypatch.setattr(app_module, "_shutting_down", False)
    monkeypatch.setattr(runner, "get_progress", get_progress)
    response = TestClient(app_module.app, base_url="http://127.0.0.1").get(
        "/api/run/progress"
    )
    events = [line for line in response.text.splitlines() if line.startswith("data:")]
    # The run is not done, but the stream ends with the server.
    assert len(events) == 2


def test_the_apps_shutdown_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[bool] = []
    monkeypatch.setattr(runner, "stop_run", lambda: stopped.append(True))
    monkeypatch.setattr(app_module, "_shutting_down", False)
    with TestClient(app_module.app, base_url="http://127.0.0.1"):
        assert not app_module._shutting_down
        assert stopped == []
    # Cleared once the shutdown is over: the app, served again, streams.
    assert not app_module._shutting_down
    assert stopped == [True]


def test_the_progress_streams_after_a_served_app_shut_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As a test that enters TestClient(app) leaves the app for the next one.
    monkeypatch.setattr(runner, "stop_run", lambda: None)
    monkeypatch.setattr(app_module, "_shutting_down", False)
    monkeypatch.setattr(
        runner,
        "get_progress",
        lambda: RunProgress(current=1, total=1, scenario_name="", status="done"),
    )

    def events(client: TestClient) -> list[str]:
        text = client.get("/api/run/progress").text
        return [line for line in text.splitlines() if line.startswith("data:")]

    for _ in range(2):
        with TestClient(app_module.app, base_url="http://127.0.0.1") as client:
            assert len(events(client)) == 1
    assert len(events(TestClient(app_module.app, base_url="http://127.0.0.1"))) == 1


def test_the_flag_is_set_while_the_app_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    during: list[bool] = []
    monkeypatch.setattr(
        runner, "stop_run", lambda: during.append(app_module._shutting_down)
    )
    monkeypatch.setattr(app_module, "_shutting_down", False)
    with TestClient(app_module.app, base_url="http://127.0.0.1"):
        pass
    assert during == [True]


def test_the_viewer_bounds_its_shutdown_and_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from autoware_carla_scenario import ui

    served: list[dict[str, Any]] = []
    stopped: list[bool] = []

    def serve(app: object, **kwargs: Any) -> None:
        served.append(kwargs)
        # A second Ctrl-C: the app's shutdown does not run.
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn, "run", serve)
    monkeypatch.setattr(runner, "stop_run", lambda: stopped.append(True))
    monkeypatch.setattr(app_module, "BASE_PATH", app_module.BASE_PATH)
    with pytest.raises(KeyboardInterrupt):
        ui.main([])
    assert served[0]["timeout_graceful_shutdown"] == ui.SHUTDOWN_TIMEOUT_SECONDS
    assert stopped == [True]


# ── A job's output goes to a file, not memory (#2239) ──────────────────────


@pytest.fixture
def chatty_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``uv`` on PATH that prints 4 MB to stdout and a last line to
    stderr, and fails; and a ``subprocess.Popen`` that records its streams."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "uv"
    script.write_text(
        "#!/bin/sh\n"
        "head -c 4194304 /dev/zero | tr '\\0' 'x'\n"
        "echo\n"
        "echo 'the last words' >&2\n"
        "exit 2\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _record_streams(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    streams: list[dict[str, Any]] = []
    popen = runner.subprocess.Popen

    def record(cmd: list[str], **kwargs: Any) -> Any:
        streams.append({k: kwargs.get(k) for k in ("stdout", "stderr")})
        return popen(cmd, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", record)
    return streams


def test_a_grouped_jobs_whole_output_is_streamed_to_its_log(
    chatty_uv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    streams = _record_streams(monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    with caplog.at_level("WARNING", logger=runner.__name__):
        runner._run_worker([["scenario=a"]], base, [], True)
    # Written to a file as the job runs: no pipe is read into memory.
    assert len(streams) == 1
    assert streams[0]["stdout"] is not runner.subprocess.PIPE
    assert hasattr(streams[0]["stdout"], "fileno")
    assert streams[0]["stderr"] is runner.subprocess.STDOUT
    [log] = (base / "multirun").glob(f"*/*/0/{runner.RAW_OUTPUT_LOG}")
    assert log.stat().st_size == 4 * 1024 * 1024 + len("\nthe last words\n")
    # The page shows its end; the warning logs a few KB of it.
    shown = runner._read_raw_log(log.parent)
    assert shown.startswith("... (truncated")
    assert shown.endswith("the last words\n")
    [warning] = [r.getMessage() for r in caplog.records if "failed" in r.getMessage()]
    assert "the last words" in warning
    assert len(warning) < 8 * 1024


def test_an_ungrouped_jobs_output_leaves_no_file(
    chatty_uv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setattr(runner.tempfile, "tempdir", str(temporary))
    base = tmp_path / "base"
    base.mkdir()
    with caplog.at_level("WARNING", logger=runner.__name__):
        runner._run_worker([["scenario=a"]], base, [], False)
    assert list(temporary.iterdir()) == []
    assert list(base.iterdir()) == []
    assert any("the last words" in r.getMessage() for r in caplog.records)


def test_a_job_that_prints_nothing_leaves_no_raw_log(tmp_path: Path) -> None:
    # The scenario page then shows the job's run.log instead.
    job = tmp_path / "0"
    (job / ".hydra").mkdir(parents=True)  # what Hydra wrote
    log = job / runner.RAW_OUTPUT_LOG
    assert runner._run_job(["true"], None, log) == 0
    runner._settle_log(log, temporary=False)
    assert not log.exists()
    assert job.is_dir()  # Hydra's, kept
    (job / "run.log").write_text("logged\n", encoding="utf-8")
    assert runner._read_raw_log(job) == "logged\n"


def test_a_job_that_never_ran_leaves_no_job_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The run is stopped before its first job starts: the directory made for
    # its log is not left behind, to be listed as a case with no result.
    monkeypatch.setattr(runner, "_stopping", True)
    monkeypatch.setattr(runner, "_stop_requested", lambda: False)
    base = tmp_path / "base"
    base.mkdir()
    runner._run_worker([["scenario=a"], ["scenario=b"]], base, [], True)
    [session] = (base / "multirun").glob("*/*")
    assert list(session.iterdir()) == []


def test_a_job_that_could_not_be_launched_leaves_no_job_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))  # no uv
    base = tmp_path / "base"
    base.mkdir()
    runner._run_worker([["scenario=a"]], base, [], True)
    [session] = (base / "multirun").glob("*/*")
    assert list(session.iterdir()) == []


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("a" * 100 + "\nlast line\n", "last line\n"),
        # One line longer than the tail: its end, rather than nothing.
        ("b" * 100, "b" * 16),
        # Cut mid-character: the partial character is replaced, then dropped
        # with the partial line it starts.
        ("é" * 60 + "\nend\n", "end\n"),
    ],
)
def test_the_tail_of_a_long_log(tmp_path: Path, text: str, shown: str) -> None:
    log = tmp_path / "log"
    log.write_text(text, encoding="utf-8")
    assert runner._tail(log, 16).endswith("...\n" + shown)


def test_a_raw_log_that_is_a_symlink_is_not_read(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    job = tmp_path / "job"
    job.mkdir()
    (job / runner.RAW_OUTPUT_LOG).symlink_to(outside)
    assert runner._read_raw_log(job) == ""
    (job / "run.log").write_text("logged\n", encoding="utf-8")
    assert runner._read_raw_log(job) == "logged\n"


@pytest.mark.parametrize(
    ("overrides", "name"),
    [
        (["x=1", "scenario=a/b"], "a/b"),
        (["x=1"], "scenario"),
        (["scenario="], ""),
    ],
)
def test_the_progress_names_a_job_by_its_scenario_override(
    overrides: list[str], name: str
) -> None:
    assert runner._extract_scenario_name(overrides) == name
