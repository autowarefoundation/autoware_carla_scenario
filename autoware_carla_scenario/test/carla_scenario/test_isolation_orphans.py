"""What a forked case started in its process group does not outlive it (#2539).

Once the case is reaped -- it returned, raised or died of a signal -- the
sweep kills its process group, so a subprocess it left behind does not run
on into the next case.  The cases are forked from a fresh interpreter, not
from the test's.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the sweeper forks its cases (Linux)"
)

_SWEEP = """
import os
import signal
import subprocess
import sys

from autoware_carla_scenario.sweeper.isolation import _launch_isolated

pid_dir = sys.argv[1]


def leaves_a_child(mode):
    def run_case():
        # Not on the sweep's pipes, which would keep the test waiting.
        child = subprocess.Popen(
            ["sleep", "600"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(os.path.join(pid_dir, mode), "w") as pid_file:
            pid_file.write(str(child.pid))
        if mode == "segv":
            os.kill(os.getpid(), signal.SIGSEGV)
        if mode == "raise":
            raise RuntimeError("the case failed")
        return 0

    return run_case


for mode in ("return", "raise", "segv"):
    print(_launch_isolated(leaves_a_child(mode), 60), flush=True)
"""


def _running(pid: int) -> bool:
    """Whether *pid* is a process that is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            state = stat.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _gone_soon(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while _running(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def test_what_a_case_started_does_not_outlive_it(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", _SWEEP, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    pids = {path.name: int(path.read_text()) for path in tmp_path.iterdir()}
    try:
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines() == [
            "(True, 0)",
            "(False, 1)",
            "(False, -11)",
        ]
        assert sorted(pids) == ["raise", "return", "segv"]
        for mode, pid in pids.items():
            assert _gone_soon(pid), f"the child of the {mode} case still runs"
    finally:
        for pid in pids.values():
            if not _running(pid):
                continue
            try:
                os.kill(pid, 9)
            except OSError:
                pass
