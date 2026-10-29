"""Stopping a lanelet_constraint sweep stops the case it is running (#708).

The sweeper forks each case into a process group of its own
(``os.setpgrp``), so a terminal's Ctrl-C reaches only the sweeper.  It has to
stop the case itself: ``SIGTERM`` to the case's group, so the case unwinds
(``with queue:``, ``CarlaServerManager``, the Autoware stack), ``SIGKILL``
after a grace period, then exit 130 (SIGINT) or 143 (SIGTERM).  A sweeper
that dies without a chance to do that (``SIGKILL``) takes its case with it.

Each test runs the sweeper in a subprocess of its own session, so a signal to
that session models one to a terminal's foreground process group.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
)

#: Calls ``_launch_job_isolated`` with a case that writes its pid, then hangs
#: until it is stopped, noting that its clean-up ran.  ``argv[2]`` is the
#: grace period; ``argv[3]`` makes the case ignore ``SIGTERM``.
_LAUNCH_APP = """
import os, signal, sys, time
from pathlib import Path
from typing import Any

from autoware_carla_scenario import hydra_compat  # noqa: F401
from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as s

s._STOP_GRACE_SECONDS = float(sys.argv[2])
work = Path(sys.argv[1])


class Launcher:
    def launch(self, batches, initial_job_idx):
        if sys.argv[3] == "stubborn":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (work / "job.pid").write_text(str(os.getpid()))
        try:
            time.sleep(60)  # a case still running
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
        time.sleep(60)
    finally:
        # What `with queue:` / `with CarlaServerManager()` do on the way out.
        (work / f"cleaned{cfg.case}").write_text("yes")


if __name__ == "__main__":
    task()
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        # FileNotFoundError, or ProcessLookupError when the process is reaped
        # between the open and the read.
        return False


def _wait_for(path: Path, timeout: float = 60.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _start(app: str, tmp_path: Path, *args: str) -> subprocess.Popen[str]:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(app), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # A session of its own: killpg(SIGINT) models Ctrl-C at a terminal.
        start_new_session=True,
    )


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_to_the_sweeper_stops_the_running_job(
    tmp_path: Path, sig: signal.Signals
) -> None:
    sweeper = _start(_LAUNCH_APP, tmp_path, str(tmp_path), "30", "polite")
    job = _wait_for(tmp_path / "job.pid")
    try:
        os.killpg(sweeper.pid, sig)
        sweeper.communicate(timeout=30)
        assert _wait_gone(job), f"job {job} still runs after the sweep got {sig.name}"
        # The job was stopped with SIGTERM, not killed: its clean-up ran.
        assert (tmp_path / "cleaned").exists()
    finally:
        _kill(job)
        _kill(sweeper.pid)


def test_a_job_that_ignores_sigterm_is_killed_after_the_grace_period(
    tmp_path: Path,
) -> None:
    sweeper = _start(_LAUNCH_APP, tmp_path, str(tmp_path), "1", "stubborn")
    job = _wait_for(tmp_path / "job.pid")
    try:
        started = time.monotonic()
        os.killpg(sweeper.pid, signal.SIGINT)
        sweeper.communicate(timeout=30)
        assert _wait_gone(job), f"job {job} outlived the grace period"
        assert time.monotonic() - started >= 0.9
    finally:
        _kill(job)
        _kill(sweeper.pid)


def test_a_killed_sweeper_takes_its_job_with_it(tmp_path: Path) -> None:
    """SIGKILL leaves the sweeper no chance: the job is told by the kernel."""
    sweeper = _start(_LAUNCH_APP, tmp_path, str(tmp_path), "30", "polite")
    job = _wait_for(tmp_path / "job.pid")
    try:
        os.kill(sweeper.pid, signal.SIGKILL)
        sweeper.communicate(timeout=30)
        assert _wait_gone(job), f"job {job} outlived its killed sweeper"
        assert (tmp_path / "cleaned").exists()
    finally:
        _kill(job)


@pytest.mark.parametrize(
    ("sig", "returncode"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)]
)
def test_an_interrupted_sweep_stops_its_case_and_exits_128_plus_the_signal(
    tmp_path: Path, sig: signal.Signals, returncode: int
) -> None:
    sweeper = _start(
        _SWEEP_APP,
        tmp_path,
        "--multirun",
        "hydra/sweeper=lanelet_constraint",
        "+sweep.constraints={}",
        f"+work={tmp_path}",
        f"hydra.sweep.dir={tmp_path / 'multirun'}",
    )
    job = _wait_for(tmp_path / "job1.pid")
    try:
        os.killpg(sweeper.pid, sig)
        output, _ = sweeper.communicate(timeout=60)
        assert sweeper.returncode == returncode, output
        assert _wait_gone(job), f"job {job} still runs after the sweep got {sig.name}"
        assert (tmp_path / "cleaned1").exists(), output
        # The sweep stopped: the next case did not start.
        assert "case 2" not in output, output
        assert "interrupted" in output, output
    finally:
        _kill(job)
        _kill(sweeper.pid)


def test_a_stop_signal_held_off_across_the_fork_reaches_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop signals are blocked across the fork, then delivered, not lost.

    A SIGTERM that reaches the child before it has its own handler (here: as
    it leaves the sweep's process group) waits until the handler is in place,
    and then stops the job as the sweep's own SIGTERM does: exit 143, not
    death by the signal under whatever handler the child inherited.  The
    parent's signal mask is left as it was.
    """
    from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as s

    class Launcher:
        def launch(self, batches: Any, initial_job_idx: int) -> Any:
            (tmp_path / "ran").write_text("yes")
            return []

    setpgrp = os.setpgrp

    def setpgrp_then_sigterm() -> None:
        setpgrp()
        os.kill(os.getpid(), signal.SIGTERM)

    # Only the forked child calls it.
    monkeypatch.setattr(s.os, "setpgrp", setpgrp_then_sigterm)
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    ok, code = s._launch_job_isolated(Launcher(), ("a=1",), 0, timeout=30)
    assert (ok, code) == (False, 128 + signal.SIGTERM)
    assert not (tmp_path / "ran").exists()
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before
