"""What ui.slurm_launch does without SLURM: the command it builds, the
submission it records, how it runs and stops the submitting process, and how
it runs ``scenario-slurm`` and tells a collect that collected.

Each test replaces the process (``subprocess.Popen``, ``subprocess.run``,
``os.killpg``, ``threading.Thread``) with a fake that records how it was
asked for, so the exact arguments, strings and order of calls are pinned.
``test_ui_slurm.py`` runs the same functions with real processes.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.ui import slurm_launch
from autoware_carla_scenario.ui.slurm_launch import SlurmOptions, Submission

# ── The command ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("configured", "command"),
    [
        (None, ["uv", "run", "scenario"]),
        ("", ["uv", "run", "scenario"]),
        ("  \t\n ", ["uv", "run", "scenario"]),
        (".venv/bin/scenario", [".venv/bin/scenario"]),
        ("  uv  run\tscenario \n", ["uv", "run", "scenario"]),
        (
            "'/opt/my env/bin/scenario' --flag \"a b\" c\\ d",
            ["/opt/my env/bin/scenario", "--flag", "a b", "c d"],
        ),
        ("a''b '' \"\"", ["ab", "", ""]),
        ('"x\\"y" \'x\\y\' x\\\\y', ['x"y', "x\\y", "x\\y"]),
        ('"a\\$b" a#b # c', ["a\\$b", "a#b", "#", "c"]),
        ("ä 'ö ü'", ["ä", "ö ü"]),
    ],
    ids=[
        "unset",
        "empty",
        "blank",
        "one-word",
        "whitespace",
        "quotes-and-escapes",
        "empty-quotes",
        "escapes-in-quotes",
        "no-comments",
        "non-ascii",
    ],
)
def test_the_scenario_command_is_split_as_a_posix_shell_splits_it(
    configured: str | None, command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    if configured is None:
        monkeypatch.delenv("VIEWER_SCENARIO_COMMAND", raising=False)
    else:
        monkeypatch.setenv("VIEWER_SCENARIO_COMMAND", configured)
    assert slurm_launch.scenario_command() == command


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        ("scenario 'unclosed", "No closing quotation"),
        ('scenario "unclosed', "No closing quotation"),
        ("scenario \\", "No escaped character"),
    ],
)
def test_a_scenario_command_no_shell_could_split_is_refused(
    configured: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIEWER_SCENARIO_COMMAND", configured)
    with pytest.raises(ValueError, match=f"^{message}$"):
        slurm_launch.scenario_command()


def test_the_command_starts_as_the_viewer_starts_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIEWER_SCENARIO_COMMAND", "'/opt/a b/scenario' -X dev")
    options = SlurmOptions(partition="main", time="01:00:00", env={"A": "x y"})
    assert slurm_launch.build_command(
        "cut_in/*", ["map=town10hd_opt", "seed=1"], "lanelet_constraint", options
    ) == [
        "/opt/a b/scenario",
        "-X",
        "dev",
        "--multirun",
        "scenario=cut_in/*",
        "map=town10hd_opt",
        "seed=1",
        "launcher=slurm",
        "launcher.partition=main",
        "launcher.time=01:00:00",
        "++launcher.env.A='x y'",
        "--no-wait",
    ]
    monkeypatch.delenv("VIEWER_SCENARIO_COMMAND")
    assert slurm_launch.build_command("x", [], "", SlurmOptions()) == [
        "uv",
        "run",
        "scenario",
        "scenario=x",
        "launcher=slurm",
        "--no-wait",
    ]


# ── The submission record ──────────────────────────────────────────────────

_STARTED = re.compile(r"\A\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00\Z")


def test_a_submission_starts_submitting_at_the_utc_second_it_was_made() -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    record = Submission(command=["scenario"])
    after = datetime.now(timezone.utc)
    assert record.command == ["scenario"]
    assert record.state == "submitting"
    assert record.batch_dir is None
    assert record.output == ""
    assert record.returncode is None
    assert _STARTED.match(record.started), record.started
    started = datetime.fromisoformat(record.started)
    assert started.tzinfo is not None
    assert started.utcoffset() == timedelta(0)
    assert before <= started <= after


def test_each_submission_reads_the_clock_and_a_given_start_is_kept() -> None:
    assert Submission(command=[], started="then").started == "then"
    first = Submission(command=[])
    second = Submission(command=[])
    assert _STARTED.match(first.started) and _STARTED.match(second.started)
    assert first.started <= second.started


def test_a_submission_is_sent_with_its_command_quoted_and_its_output_s_end() -> None:
    record = Submission(
        command=["uv", "run", "scenario", "scenario=a b", "x'y", "", "ä"],
        state="failed",
        batch_dir="/runs/outputs/2026-01-01/00-00-00",
        output="x" * 8000 + "tail",
        returncode=2,
        started="2026-01-01T00:00:00+00:00",
    )
    assert record.to_dict() == {
        "command": "uv run scenario 'scenario=a b' 'x'\"'\"'y' '' 'ä'",
        "state": "failed",
        "batch_dir": "/runs/outputs/2026-01-01/00-00-00",
        "output": "x" * 7996 + "tail",
        "returncode": 2,
        "started": "2026-01-01T00:00:00+00:00",
    }
    assert list(record.to_dict()) == [
        "command",
        "state",
        "batch_dir",
        "output",
        "returncode",
        "started",
    ]
    short = Submission(command=[], output="short", started="s").to_dict()
    assert short == {
        "command": "",
        "state": "submitting",
        "batch_dir": None,
        "output": "short",
        "returncode": None,
        "started": "s",
    }


def test_the_submission_is_the_last_one_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(slurm_launch, "_submission", None)
    assert slurm_launch.submission() is None
    record = Submission(command=[])
    monkeypatch.setattr(slurm_launch, "_submission", record)
    assert slurm_launch.submission() is record


# ── The manifests a submit wrote ───────────────────────────────────────────


def _manifest(base: Path, date: str, time: str) -> Path:
    path = base / "outputs" / date / time / "slurm" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    return path


def test_new_manifests_are_those_not_seen_before_by_date_then_session(
    tmp_path: Path,
) -> None:
    old = _manifest(tmp_path, "2026-11-02", "00-00-00")
    _manifest(tmp_path, "2026-11-02", "12-00-00-10")
    _manifest(tmp_path, "2026-11-01", "23-59-59")
    _manifest(tmp_path, "2026-11-02", "12-00-00-2")
    _manifest(tmp_path, "2026-11-02", "12-00-00")
    (tmp_path / "outputs" / "2026-11-03" / "00-00-00" / "slurm").mkdir(parents=True)
    found = slurm_launch._new_manifests(tmp_path, {old})
    assert [(p.parents[2].name, p.parents[1].name) for p in found] == [
        ("2026-11-01", "23-59-59"),
        ("2026-11-02", "12-00-00"),
        ("2026-11-02", "12-00-00-2"),
        ("2026-11-02", "12-00-00-10"),
    ]
    assert all(p.name == "manifest.json" for p in found)
    assert slurm_launch._new_manifests(tmp_path / "nothing", set()) == []


# ── Stopping the submitting process ────────────────────────────────────────


class _Process:
    pid = 4242


def test_the_process_group_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    slurm_launch._kill_group(_Process())  # type: ignore[arg-type]
    assert calls == [(4242, signal.SIGKILL)]


def test_a_group_that_has_ended_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def killpg(pgid: int, sig: int) -> None:
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "killpg", killpg)
    slurm_launch._kill_group(_Process())  # type: ignore[arg-type]


def test_a_group_that_cannot_be_killed_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def killpg(pgid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", killpg)
    with pytest.raises(PermissionError):
        slurm_launch._kill_group(_Process())  # type: ignore[arg-type]


# ── Running the submitting command ─────────────────────────────────────────


class _FakePopen:
    """A Popen whose communicate() calls give, in turn, *replies*: an
    ``(stdout, stderr)`` pair, or an exception to raise."""

    def __init__(self, replies: list[Any], returncode: int = 0) -> None:
        self.replies = list(replies)
        self.returncode = returncode
        self.pid = 777
        self.calls: list[Any] = []

    def __call__(self, command: list[str], **kwargs: Any) -> _FakePopen:
        self.calls.append(("Popen", command, kwargs))
        return self

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.calls.append(("communicate", timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def wait(self) -> int:
        self.calls.append(("wait",))
        return self.returncode


def _fake_process(
    monkeypatch: pytest.MonkeyPatch, replies: list[Any], returncode: int = 0
) -> _FakePopen:
    fake = _FakePopen(replies, returncode)
    monkeypatch.setattr(subprocess, "Popen", fake)
    monkeypatch.setattr(
        slurm_launch,
        "_kill_group",
        lambda process: fake.calls.append(("kill", process)),
    )
    return fake


_POPEN_KWARGS = {
    "cwd": "/the/base",
    "stdin": subprocess.DEVNULL,
    "stdout": subprocess.PIPE,
    "stderr": subprocess.PIPE,
    "encoding": "utf-8",
    "errors": "replace",
    "start_new_session": True,
}


@pytest.mark.parametrize(
    ("stdout", "stderr", "output"),
    [
        ("out\n", "err\n", "out\n\nerr\n"),
        ("out", "", "out"),
        ("", "err", "err"),
        ("", "", ""),
    ],
)
def test_a_submit_gives_its_status_and_output(
    stdout: str, stderr: str, output: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_process(monkeypatch, [(stdout, stderr)], returncode=3)
    result = slurm_launch._run_submit(["scenario", "x"], Path("/the/base"))
    assert result == (3, output)
    assert fake.calls == [
        ("Popen", ["scenario", "x"], _POPEN_KWARGS),
        ("communicate", 1800.0),
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory", "scenario"),
        PermissionError(13, "Permission denied"),
        ValueError("embedded null byte"),
    ],
)
def test_a_submit_that_cannot_start_says_why(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def popen(command: list[str], **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert slurm_launch._run_submit(["scenario", "x"], Path("/b")) == (
        None,
        f"Could not run scenario: {error}",
    )


_REASON = (
    "scenario did not end within {} s and was stopped: the batch's checks or "
    "sbatch hung (is the SLURM controller answering?). Stopped during sbatch, "
    "it keeps the batch's submission record, as the controller may have taken "
    "the job: if `squeue --me` lists it, it runs and the batch can be "
    "collected once it ends (or scancel it), and resubmits of that batch are "
    "refused until 600 s after the record was made. A new submit (a new "
    "batch) is taken now."
)


@pytest.mark.parametrize(
    ("timeout", "shown"), [(1800.0, "1800"), (1.5, "1.5"), (0.25, "0.25")]
)
@pytest.mark.parametrize(
    ("stdout", "stderr", "before"),
    [
        ("checking\n", "warn\n", "checking\n\nwarn\n\n"),
        ("checking", "", "checking\n"),
        ("", "", ""),
    ],
)
def test_a_submit_that_hangs_is_stopped_with_what_it_wrote(
    timeout: float,
    shown: str,
    stdout: str,
    stderr: str,
    before: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(slurm_launch, "SUBMIT_TIMEOUT_SECONDS", timeout)
    expired = subprocess.TimeoutExpired(["scenario"], timeout)
    fake = _fake_process(monkeypatch, [expired, (stdout, stderr)])
    code, output = slurm_launch._run_submit(["scenario"], Path("/the/base"))
    assert code == slurm_launch.TIMED_OUT == 124
    assert output == before + _REASON.format(shown)
    assert fake.calls == [
        ("Popen", ["scenario"], _POPEN_KWARGS),
        ("communicate", timeout),
        ("kill", fake),
        ("communicate", 10.0),
    ]


def test_a_stopped_submit_whose_pipes_stay_open_is_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_process(
        monkeypatch,
        [
            subprocess.TimeoutExpired(["scenario"], 1800.0),
            subprocess.TimeoutExpired(["scenario"], 10.0),
        ],
    )
    code, output = slurm_launch._run_submit(["scenario"], Path("/the/base"))
    assert (code, output) == (124, _REASON.format("1800"))
    assert fake.calls[1:] == [
        ("communicate", 1800.0),
        ("kill", fake),
        ("communicate", 10.0),
        ("wait",),
    ]


@pytest.mark.parametrize("raised", [KeyboardInterrupt(), RuntimeError("boom")])
def test_a_submit_interrupted_while_waiting_is_killed_and_reaped(
    raised: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_process(monkeypatch, [raised])
    with pytest.raises(type(raised)) as caught:
        slurm_launch._run_submit(["scenario"], Path("/the/base"))
    assert caught.value is raised
    assert fake.calls[1:] == [("communicate", 1800.0), ("kill", fake), ("wait",)]


# ── What a submit records ──────────────────────────────────────────────────


def _submit_with(
    monkeypatch: pytest.MonkeyPatch,
    base: Path,
    returncode: int | None,
    output: str,
    write: tuple[tuple[str, str], ...] = (),
) -> Submission:
    asked: list[Any] = []

    def run_submit(command: list[str], at: Path) -> tuple[int | None, str]:
        asked.append((command, at))
        for date, time in write:
            _manifest(base, date, time)
        return returncode, output

    monkeypatch.setattr(slurm_launch, "_run_submit", run_submit)
    record = Submission(command=["scenario", "y"])
    slurm_launch._submit(["scenario", "y"], base, record)
    assert asked == [(["scenario", "y"], base)]
    assert record.output == output
    assert record.returncode == returncode
    return record


@pytest.mark.parametrize(
    ("output", "batch_dir"),
    [
        ("SLURM job 1: 2 case(s); outputs in /r/outputs/d/t\n", "/r/outputs/d/t"),
        ("outputs in /r/a b/c  \t\nmore\n", "/r/a b/c"),
        ("outputs in /first\nlog\noutputs in /second\nend", "/second"),
        ("x outputs in /one outputs in /two", "/one outputs in /two"),
        ("outputs in \n", None),
        ("outputs in   x  ", "  x"),  # the blanks after "in " are kept
        ("OUTPUTS IN /upper", None),
        ("outputsin /no", None),
    ],
    ids=[
        "one",
        "spaces-and-trailing-blanks",
        "last-line-wins",
        "rest-of-the-line",
        "nothing-after",
        "blanks-before",
        "case-sensitive",
        "no-space",
    ],
)
def test_the_batch_directory_is_the_last_one_the_output_names(
    output: str, batch_dir: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _submit_with(monkeypatch, tmp_path, 0, output)
    assert record.batch_dir == batch_dir
    assert record.state == ("submitted" if batch_dir else "failed")


def test_without_one_named_the_batch_is_the_newest_new_manifest_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest(tmp_path, "2026-12-31", "23-59-59")  # there before: not this one
    record = _submit_with(
        monkeypatch,
        tmp_path,
        0,
        "no batch named",
        write=(("2026-11-01", "12-00-00-10"), ("2026-11-01", "12-00-00-9")),
    )
    assert record.batch_dir == str(tmp_path / "outputs" / "2026-11-01" / "12-00-00-10")
    assert record.state == "submitted"


@pytest.mark.parametrize(
    ("returncode", "output", "state", "batch_dir"),
    [
        (0, "outputs in /b", "submitted", "/b"),
        (1, "outputs in /b", "failed", "/b"),
        (124, "outputs in /b", "failed", "/b"),
        (None, "Could not run scenario: x", "failed", None),
        (0, "nothing", "failed", None),
    ],
)
def test_a_submit_is_submitted_only_with_status_0_and_a_batch(
    returncode: int | None,
    output: str,
    state: str,
    batch_dir: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _submit_with(monkeypatch, tmp_path, returncode, output)
    assert (record.state, record.batch_dir) == (state, batch_dir)


class _FakeThread:
    made: list[_FakeThread] = []

    def __init__(self, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
        self.target, self.args, self.daemon = target, args, daemon
        self.started = False
        _FakeThread.made.append(self)

    def start(self) -> None:
        self.started = True


def test_a_submit_runs_in_a_daemon_thread_and_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(slurm_launch, "_submission", None)
    monkeypatch.setattr(_FakeThread, "made", [])
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    command = ["scenario", "z"]
    record = slurm_launch.start_submit(command, Path("/base"))
    assert slurm_launch.submission() is record
    assert record.state == "submitting"
    assert record.command == command and record.command is not command
    [thread] = _FakeThread.made
    assert thread.target is slurm_launch._submit
    assert thread.args == (command, Path("/base"), record)
    assert thread.args[0] is command
    assert thread.daemon is True and thread.started
    with pytest.raises(RuntimeError, match=r"^A batch is still being submitted$"):
        slurm_launch.start_submit(["other"], Path("/base"))
    assert slurm_launch.submission() is record and len(_FakeThread.made) == 1
    for ended in ("submitted", "failed", "anything"):
        record.state = ended
        again = slurm_launch.start_submit(["again"], Path("/b2"))
        assert again is not record and slurm_launch.submission() is again
        record = again


# ── scenario-slurm ─────────────────────────────────────────────────────────


class _Done:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.mark.parametrize(
    ("stdout", "stderr", "output"),
    [("o\n", "e\n", "o\n\ne\n"), ("o", "", "o"), ("", "e", "e"), ("", "", "")],
)
def test_scenario_slurm_is_run_by_this_interpreter(
    stdout: str, stderr: str, output: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []

    def run(command: list[str], **kwargs: Any) -> _Done:
        calls.append((command, kwargs))
        return _Done(5, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", run)
    assert slurm_launch.run_cli(["status", "/b"]) == (5, output)
    assert slurm_launch.run_cli(["cancel"], timeout=2.5) == (5, output)
    module = "autoware_carla_scenario.slurm.cli"
    kwargs = {
        "capture_output": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
    }
    assert calls == [
        ([sys.executable, "-m", module, "status", "/b"], {**kwargs, "timeout": 300.0}),
        ([sys.executable, "-m", module, "cancel"], {**kwargs, "timeout": 2.5}),
    ]


@pytest.mark.parametrize(
    ("raised", "result"),
    [
        (
            subprocess.TimeoutExpired(["python", "-m", "x"], 300.0),
            (124, "Command '['python', '-m', 'x']' timed out after 300.0 seconds"),
        ),
        (
            FileNotFoundError(2, "No such file or directory", "python"),
            (127, "[Errno 2] No such file or directory: 'python'"),
        ),
        (OSError("no fork"), (127, "no fork")),
    ],
    ids=["timeout", "not-found", "oserror"],
)
def test_a_scenario_slurm_that_cannot_run_says_why(
    raised: Exception, result: tuple[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(subprocess, "run", run)
    assert slurm_launch.run_cli(["collect", "/b"]) == result


def test_a_scenario_slurm_that_raises_otherwise_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        slurm_launch.run_cli(["collect", "/b"])


def test_a_file_is_told_from_the_next_by_inode_time_and_size(tmp_path: Path) -> None:
    path = tmp_path / "collected.json"
    assert slurm_launch._stat_key(path) is None
    assert slurm_launch._stat_key(tmp_path / "no" / "such" / "file") is None
    path.write_text("{}")
    stat = path.stat()
    assert slurm_launch._stat_key(path) == (stat.st_ino, stat.st_mtime_ns, 2)


@pytest.mark.parametrize(
    ("code", "writes", "ok"),
    [
        (0, False, True),
        (0, True, True),
        (1, True, True),
        (1, False, False),
        (2, True, False),
        (124, True, False),
        (127, False, False),
    ],
)
@pytest.mark.parametrize("there_before", [False, True])
def test_a_collect_collected_on_status_0_or_1_with_a_new_record(
    code: int,
    writes: bool,
    ok: bool,
    there_before: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = tmp_path / "outputs" / "2026-01-01" / "00-00-00"
    collected = batch / "slurm" / "collected.json"
    collected.parent.mkdir(parents=True)
    if there_before:
        collected.write_text("{}")
    asked: list[Any] = []

    def run_cli(args: list[str], timeout: float = 300.0) -> tuple[int, str]:
        asked.append((args, timeout))
        if writes:
            new = collected.with_name("collected.json.tmp")
            new.write_text('{"collected": true}')
            new.replace(collected)
        return code, "the output"

    monkeypatch.setattr(slurm_launch, "run_cli", run_cli)
    assert slurm_launch.run_collect(batch, 7.0) == (ok, code, "the output")
    assert slurm_launch.run_collect(batch) == (ok, code, "the output")
    assert asked == [(["collect", str(batch)], 7.0), (["collect", str(batch)], 300.0)]


def test_a_collect_that_leaves_the_record_as_it_was_did_not_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "b"
    collected = batch / "slurm" / "collected.json"
    collected.parent.mkdir(parents=True)
    collected.write_text("{}")
    monkeypatch.setattr(slurm_launch, "run_cli", lambda args, timeout=300.0: (1, ""))
    assert slurm_launch.run_collect(batch) == (False, 1, "")

    # One it removed is no record either.
    def removing(args: list[str], timeout: float = 300.0) -> tuple[int, str]:
        collected.unlink()
        return 1, "x"

    monkeypatch.setattr(slurm_launch, "run_cli", removing)
    assert slurm_launch.run_collect(batch) == (False, 1, "x")
