"""Stopping a lanelet_constraint sweep stops the case it is running (#708).

The sweeper forks each case into a process group of its own
(``os.setpgrp``), so a terminal's Ctrl-C reaches only the sweeper.  It has to
stop the case itself: ``SIGTERM`` to the case's group, so the case unwinds
(``with queue:``, ``CarlaServerManager``, the Autoware stack), ``SIGKILL``
after a grace period, then exit 130 (SIGINT) or 143 (SIGTERM).  A sweeper
that dies without a chance to do that (``SIGKILL``) takes its case with it.

Each test runs the sweeper in a subprocess of its own session, so a signal to
that session models one to a terminal's foreground process group, and every
process the test started -- the sweeper, its case, anything else -- can be
found by that session and killed, whatever way the test ends.

Time bounds (#750).  The machine running these may stall for tens of seconds
(a full ``-n auto`` run beside Codon compiles), so no wall-clock bound below
is part of what a test verifies, except the one lower bound on the grace
period.  The cases sleep :data:`_JOB_SECONDS`, so a case can never end by
itself and pass for one that was stopped; the bounds that only guard
against a hang (:data:`_START_TIMEOUT`, :data:`_EXIT_TIMEOUT`,
:data:`_GONE_TIMEOUT`) are generous, and when one runs out the failure
carries the sweeper's output and the ``/proc`` state of what still runs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
)

#: How long a case sleeps: far longer than any bound here, so a case that is
#: still asleep was not stopped, never one that ended by itself.  (The apps
#: below spell it out; :func:`test_the_apps_sleep_job_seconds` keeps them in
#: step.)  Every test kills what it started, so the sleep never outlives it.
_JOB_SECONDS = 3600

#: Hang guard: from starting the sweeper until its case has written its pid
#: (a Python start-up and, for the Hydra app, Hydra's import and config
#: composition, on a loaded machine).  It fails at once instead if the
#: sweeper exits first.
_START_TIMEOUT = 120.0

#: Grace period the tests with a case that honours ``SIGTERM`` give it.  Not
#: verified: a case that is not sent ``SIGTERM`` is sent ``SIGKILL`` after it,
#: and fails the test because its clean-up did not run.  Long, so a stalled
#: case that was sent ``SIGTERM`` still runs its clean-up in time.
_POLITE_GRACE_SECONDS = 60

#: Hang guard: from the stop signal until the sweeper has exited.  It covers
#: the sweeper's whole teardown -- the case's unwinding, at most
#: :data:`_POLITE_GRACE_SECONDS` before the ``SIGKILL`` -- with room to spare.
_EXIT_TIMEOUT = 120.0

#: Hang guard: how long a case may take to go after its sweeper died of a
#: signal.  The kernel sends it ``SIGTERM`` (``PR_SET_PDEATHSIG``) at once;
#: this is only the time to unwind.  The tests whose sweeper stops its case
#: itself need no such wait: the sweeper reaps its case before it exits.
_GONE_TIMEOUT = 60.0

#: Run first in every app: the stop signals' dispositions of a sweeper started
#: at a terminal.  A process inherits ignored signals, and a pytest started in
#: the background by a non-interactive shell (``pytest &`` in a script, an
#: agent's background command) ignores ``SIGINT`` and ``SIGQUIT``: then so
#: would every sweeper here, and the ``SIGINT`` tests would wait out
#: :data:`_EXIT_TIMEOUT` for a signal that never arrives (#750).
_PREAMBLE = """
import signal as _signal

_signal.signal(_signal.SIGINT, _signal.default_int_handler)
_signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
"""

#: Calls ``_launch_job_isolated`` with a case that writes its pid, then hangs
#: until it is stopped, noting that its clean-up ran.  ``argv[2]`` is the
#: grace period; ``argv[3]`` makes the case ignore ``SIGTERM``.  The case
#: timeout (600 s) is above every bound here, so no test reaches it; it only
#: stops a case of a sweeper nobody killed.
_LAUNCH_APP = """
import os, signal, sys, time
from pathlib import Path
from typing import Any

from autoware_carla_scenario import hydra_compat  # noqa: F401
from autoware_carla_scenario.sweeper import isolation
from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as s

isolation._STOP_GRACE_SECONDS = float(sys.argv[2])
work = Path(sys.argv[1])


class Launcher:
    def launch(self, batches, initial_job_idx):
        if sys.argv[3] == "stubborn":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (work / "job.pid").write_text(str(os.getpid()))
        try:
            time.sleep(3600)  # a case still running: _JOB_SECONDS
        finally:
            (work / "cleaned").write_text("yes")


print(s._launch_job_isolated(Launcher(), ("a=1",), 0, timeout=600), flush=True)
"""

#: A ``@hydra.main`` app swept by LaneletConstraintSweeper, expanded to two
#: cases; the first hangs until it is stopped.
_SWEEP_APP = """
import os, time
from pathlib import Path
from typing import Any

from autoware_carla_scenario import hydra_compat  # noqa: F401

import hydra
from omegaconf import DictConfig

import autoware_carla_scenario.sweeper.lanelet_constraint_sweeper as lcs
from autoware_carla_scenario.sweeper import isolation

# _POLITE_GRACE_SECONDS: room for a stalled case to unwind.
isolation._STOP_GRACE_SECONDS = 60.0
lcs.check_constraint_keys = lambda sweep: None
lcs.resolve_map_paths = lambda cfg: None
lcs.load_map = lambda paths: object()
lcs.scenario_controls = lambda cfg: None
lcs.expand_sweep = lambda sweep, lanelet_map, arguments, **_: [
    ["+case=1", *arguments],
    ["+case=2", *arguments],
]


@hydra.main(version_base=None, config_path=None, config_name=None)
def task(cfg: DictConfig) -> None:
    print(f"case {cfg.case}", flush=True)
    work = Path(cfg.work)
    (work / f"job{cfg.case}.pid").write_text(str(os.getpid()))
    try:
        time.sleep(3600)  # _JOB_SECONDS
    finally:
        # What `with queue:` / `with CarlaServerManager()` do on the way out.
        (work / f"cleaned{cfg.case}").write_text("yes")


if __name__ == "__main__":
    task()
"""

#: Runs ``_launch_job_isolated`` with a case whose ``os.setpgrp`` sends the
#: case ``SIGTERM`` right after it, and prints what came of it as JSON.  In a
#: fresh, single-threaded interpreter, as the sweeper is: forking the
#: multi-threaded pytest-xdist worker could deadlock the child.
_FORK_APP = """
import json, os, signal, sys
from pathlib import Path

from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as s

work = Path(sys.argv[1])


class Launcher:
    def launch(self, batches, initial_job_idx):
        (work / "ran").write_text("yes")
        return []


setpgrp = os.setpgrp


def setpgrp_then_sigterm():
    setpgrp()
    os.kill(os.getpid(), signal.SIGTERM)


# Only the forked child calls it.
s.os.setpgrp = setpgrp_then_sigterm
before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
ok, code = s._launch_job_isolated(Launcher(), ("a=1",), 0, timeout=600)
after = signal.pthread_sigmask(signal.SIG_BLOCK, [])
print(json.dumps({"ok": ok, "code": code, "mask_kept": before == after}), flush=True)
"""


def _stat(pid: int) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` after the command name, or *None*."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        # FileNotFoundError, or ProcessLookupError when the process is reaped
        # between the open and the read.
        return None


def _alive(pid: int) -> bool:
    fields = _stat(pid)
    return fields is not None and fields[0] != "Z"


def _session(sid: int) -> list[int]:
    """The live processes of session *sid*: what a test started, and nothing else."""
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            fields = _stat(int(name))
            if fields is not None and fields[0] != "Z" and int(fields[3]) == sid:
                pids.append(int(name))
    return sorted(pids)


def _proc_status(pid: int) -> str:
    """The lines of ``/proc/<pid>/status`` that say why a process hangs."""
    keys = ("Name", "State", "PPid", "SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt")
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status:
            lines = [line.rstrip() for line in status if line.startswith(keys)]
    except OSError:
        return "gone"
    try:
        with open(f"/proc/{pid}/wchan", encoding="utf-8") as wchan:
            lines.append(f"wchan:\t{wchan.read()}")
    except OSError:
        pass
    return "; ".join(lines)


class _Sweeper:
    """A sweeper subprocess in a session of its own, with its output in a file.

    Its output goes to a file, not a pipe: the case it forks inherits the
    descriptor, and a pipe would tie reading the output to the case exiting.

    The sweeper is reaped only by :meth:`kill`, once its session is empty:
    until then its pid -- the id of its session and of its process group --
    cannot be reused, so signalling that group, or every process of that
    session, reaches only what this test started, never another xdist
    worker's sweep or the user's processes.
    """

    def __init__(self, app: str, tmp_path: Path, *args: str) -> None:
        script = tmp_path / "app.py"
        script.write_text(_PREAMBLE + textwrap.dedent(app), encoding="utf-8")
        self._log = tmp_path / "sweeper.log"
        env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
        with self._log.open("w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(
                [sys.executable, str(script), *args],
                cwd=tmp_path,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                # A session of its own: killpg(SIGINT) models Ctrl-C at a
                # terminal, and _session finds everything the test started.
                start_new_session=True,
            )
        self.pid = self.proc.pid
        #: The case's pid, once it wrote it.
        self.job: int | None = None
        #: The sweeper's exit status once it exited (as Popen's returncode),
        #: read without reaping it (:meth:`poll`).
        self.returncode: int | None = None
        #: What :meth:`kill` could not kill, once it has run.
        self._left: list[int] | None = None

    def poll(self) -> int | None:
        """The sweeper's exit status once it has exited, ``None`` while it runs.

        ``WNOWAIT`` leaves it a zombie: its pid stays reserved (the class
        docstring).
        """
        if self.returncode is None and self._left is None:
            result = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if result is not None:
                self.returncode = (
                    result.si_status
                    if result.si_code == os.CLD_EXITED
                    else -result.si_status
                )
        return self.returncode

    def output(self) -> str:
        try:
            return self._log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def diagnose(self) -> str:
        """The ``/proc`` state of the sweeper, its case and the rest of its session."""
        pids = set(_session(self.pid)) | {self.pid}
        if self.job is not None:
            pids.add(self.job)
        lines = []
        for pid in sorted(pids):
            role = {self.pid: "sweeper", self.job: "job"}.get(pid, "other")
            lines.append(f"  {role} {pid}: {_proc_status(pid)}")
        return "\n".join(lines)

    def fail(self, why: str) -> AssertionError:
        """An AssertionError saying *why*, with the state and output of the sweep.

        The state is taken first, while what hangs still runs; then the whole
        session is killed so the output is complete.
        """
        state = self.diagnose()
        self.kill()
        return AssertionError(
            f"{why}\n/proc state:\n{state}\nsweeper output:\n{self.output()}"
        )

    def wait_for(self, path: Path) -> int:
        """Wait for the case to write its pid to *path*, and return it."""
        deadline = time.monotonic() + _START_TIMEOUT
        while True:
            if path.exists():
                text = path.read_text()
                if text:
                    self.job = int(text)
                    return self.job
            if self.poll() is not None:
                raise self.fail(
                    f"the sweeper exited ({self.returncode}) before {path} was written"
                )
            if time.monotonic() >= deadline:
                raise self.fail(f"{path} was not written in {_START_TIMEOUT:g}s")
            time.sleep(0.05)

    def wait(self, since: str = "it was stopped") -> str:
        """Wait for the sweeper to exit, and return its output."""
        deadline = time.monotonic() + _EXIT_TIMEOUT
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise self.fail(
                    f"the sweeper still runs {_EXIT_TIMEOUT:g}s after {since}"
                )
            time.sleep(0.05)
        return self.output()

    def wait_gone(self, job: int) -> None:
        """Wait for the case to go after its sweeper died (``PR_SET_PDEATHSIG``)."""
        deadline = time.monotonic() + _GONE_TIMEOUT
        while _alive(job):
            if time.monotonic() >= deadline:
                raise self.fail(
                    f"job {job} still runs {_GONE_TIMEOUT:g}s after its sweeper "
                    f"died ({self.returncode})"
                )
            time.sleep(0.05)

    def assert_job_gone(self, job: int, sig: signal.Signals) -> None:
        """The case is gone once the sweeper has exited: it reaps it first."""
        if _alive(job):
            raise self.fail(f"job {job} still runs after the sweep got {sig.name}")

    def kill(self) -> list[int]:
        """``SIGKILL`` everything the test started, then reap the sweeper.

        The sweeper's group first (the sweeper alone: each case has a group
        of its own), so it forks nothing more; then every process of its
        session, until none is left: a case that ignores ``SIGTERM`` would
        otherwise sleep out :data:`_JOB_SECONDS`.  The cases never leave the
        session (``os.setpgrp``, not ``os.setsid``).  Only then is the
        sweeper reaped, so its pid stayed reserved throughout.

        Returns:
            The processes of the session that were still alive after 60 s
            (empty unless something would not die).  A second call signals
            nothing and returns the first call's.
        """
        if self._left is not None:
            return self._left
        _killpg(self.pid)
        deadline = time.monotonic() + 60.0
        left = _session(self.pid)
        while left and time.monotonic() < deadline:
            for pid in left:
                _kill(pid)
            time.sleep(0.01)
            left = _session(self.pid)
        self.poll()
        self._left = left
        try:
            self.proc.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass  # in `left`: still a live process of the session
        if self.returncode is None:
            self.returncode = self.proc.returncode
        return left


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Callable[..., _Sweeper]]:
    """Start sweepers (:class:`_Sweeper`); kill all they started at teardown.

    The teardown runs however the test ended -- passed, failed, or a bound
    ran out -- so no case outlives its test.
    """
    sweepers: list[_Sweeper] = []

    def _start(app: str, *args: str) -> _Sweeper:
        sweeper = _Sweeper(app, tmp_path, *args)
        sweepers.append(sweeper)
        return sweeper

    yield _start
    left = {}
    for sweeper in sweepers:
        still = sweeper.kill()
        if still:
            left[sweeper.pid] = still
    assert not left, f"processes outlived the test: {left}"


def test_the_apps_sleep_job_seconds() -> None:
    """The apps' cases sleep :data:`_JOB_SECONDS`, which they cannot import."""
    for app in (_LAUNCH_APP, _SWEEP_APP):
        assert f"time.sleep({_JOB_SECONDS})" in app
    assert f"_STOP_GRACE_SECONDS = {float(_POLITE_GRACE_SECONDS)}" in _SWEEP_APP


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_to_the_sweeper_stops_the_running_job(
    tmp_path: Path, start: Callable[..., _Sweeper], sig: signal.Signals
) -> None:
    sweeper = start(_LAUNCH_APP, str(tmp_path), str(_POLITE_GRACE_SECONDS), "polite")
    job = sweeper.wait_for(tmp_path / "job.pid")
    os.killpg(sweeper.pid, sig)
    output = sweeper.wait()
    if sig == signal.SIGINT:
        # The KeyboardInterrupt stopped the job (_stop_job), which reaped it.
        sweeper.assert_job_gone(job, sig)
        assert "Stopping the running job" in output, output
    else:
        # This bare app has no SIGTERM handler (sweep() installs one: the
        # Hydra test below): the sweeper dies of the signal, and its job is
        # stopped by the kernel's SIGTERM (PR_SET_PDEATHSIG), in its own time.
        sweeper.wait_gone(job)
    # The job was stopped with SIGTERM, not killed: its clean-up ran.
    assert (tmp_path / "cleaned").exists(), output


def test_a_job_that_ignores_sigterm_is_killed_after_the_grace_period(
    tmp_path: Path, start: Callable[..., _Sweeper]
) -> None:
    """``SIGKILL`` follows ``SIGTERM`` after the grace period, not before.

    The lower bound on the time it took is what shows the grace period was
    waited out; a stall only makes that time longer.  That the grace period
    was the one configured (1 s, not the default 30 s) is read from the
    sweeper's log, not from a wall-clock upper bound.
    """
    sweeper = start(_LAUNCH_APP, str(tmp_path), "1", "stubborn")
    job = sweeper.wait_for(tmp_path / "job.pid")
    started = time.monotonic()
    os.killpg(sweeper.pid, signal.SIGINT)
    output = sweeper.wait()
    elapsed = time.monotonic() - started
    sweeper.assert_job_gone(job, signal.SIGINT)
    assert elapsed >= 0.9, f"the job was killed {elapsed:.2f}s after SIGTERM"
    assert "SIGTERM, SIGKILL after 1s" in output, output
    assert "still runs 1s after SIGTERM: SIGKILL" in output, output
    # Killed, not stopped: its clean-up did not run.
    assert not (tmp_path / "cleaned").exists(), output


def test_a_killed_sweeper_takes_its_job_with_it(
    tmp_path: Path, start: Callable[..., _Sweeper]
) -> None:
    """SIGKILL leaves the sweeper no chance: the job is told by the kernel."""
    sweeper = start(_LAUNCH_APP, str(tmp_path), str(_POLITE_GRACE_SECONDS), "polite")
    job = sweeper.wait_for(tmp_path / "job.pid")
    os.kill(sweeper.pid, signal.SIGKILL)
    output = sweeper.wait()
    sweeper.wait_gone(job)
    assert (tmp_path / "cleaned").exists(), output


@pytest.mark.parametrize(
    ("sig", "returncode"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)]
)
def test_an_interrupted_sweep_stops_its_case_and_exits_128_plus_the_signal(
    tmp_path: Path,
    start: Callable[..., _Sweeper],
    sig: signal.Signals,
    returncode: int,
) -> None:
    sweeper = start(
        _SWEEP_APP,
        "--multirun",
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        # Above every bound here: case 1 never times out into case 2.
        "+sweep.job_timeout_seconds=600",
        f"+work={tmp_path}",
        f"hydra.sweep.dir={tmp_path / 'multirun'}",
    )
    job = sweeper.wait_for(tmp_path / "job1.pid")
    os.killpg(sweeper.pid, sig)
    output = sweeper.wait()
    assert sweeper.returncode == returncode, output
    sweeper.assert_job_gone(job, sig)
    # Stopped by the sweep, not only by the kernel once the sweep was gone.
    assert "Stopping the running job" in output, output
    assert (tmp_path / "cleaned1").exists(), output
    # The sweep stopped: the next case did not start.  Not a matter of time:
    # case 1 sleeps _JOB_SECONDS, so only the sweep going on could start it.
    assert "case 2" not in output, output
    assert not (tmp_path / "job2.pid").exists(), output
    assert "interrupted" in output, output


def test_a_stop_signal_held_off_across_the_fork_reaches_the_job(
    tmp_path: Path, start: Callable[..., _Sweeper]
) -> None:
    """The stop signals are blocked across the fork, then delivered, not lost.

    A SIGTERM that reaches the child before it has its own handler (here: as
    it leaves the sweep's process group) waits until the handler is in place,
    and then stops the job as the sweep's own SIGTERM does: exit 143, not
    death by the signal under whatever handler the child inherited (here
    ``SIG_DFL``).  The parent's signal mask is left as it was.
    """
    sweeper = start(_FORK_APP, str(tmp_path))
    # Not a stop: the case stops itself; the bound only guards a hang.
    output = sweeper.wait("it started")
    assert sweeper.returncode == 0, output
    result = json.loads(output.strip().splitlines()[-1])
    assert (result["ok"], result["code"]) == (False, 128 + signal.SIGTERM), output
    assert not (tmp_path / "ran").exists(), output
    assert result["mask_kept"], output
