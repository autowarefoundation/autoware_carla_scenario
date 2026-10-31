"""Signal what a test started, never a process that reused its pid (#812).

A pid -- and so a process group id, which is its leader's pid -- is free for
reuse once its process is reaped.  Under ``pytest -n auto`` many tests spawn
processes at once and pids wrap (``kernel.pid_max`` is often 32768), so a
teardown that sends ``SIGKILL`` to a pid or a group after that may hit another
test's process, or the session leader of another test's run.

Two rules keep a teardown signal on what the test started:

* A child of the test (a :class:`subprocess.Popen`) is signalled only while
  the test has not reaped it: until then its pid stays reserved, even once it
  has exited (a zombie), so the group it leads cannot be anyone else's.
* Any other process is named by its pid *and* its start time (field 22 of
  ``/proc/<pid>/stat``), read while it still ran; it is signalled only while
  the process under that pid still has that start time, so is still the one
  the test started.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def _stat(pid: int) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` after the command name, or ``None``."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        # FileNotFoundError, or ProcessLookupError when the process is reaped
        # between the open and the read.
        return None


def start_time(pid: int) -> int | None:
    """When *pid* started, in clock ticks since boot; ``None`` once it is gone."""
    fields = _stat(pid)
    # Field 22 of the file; the list starts at field 3 (the state).
    return int(fields[19]) if fields is not None else None


def is_same_process(pid: int, started: int) -> bool:
    """Whether *pid* is still the process that started at *started*."""
    return start_time(pid) == started


def signal_unreaped_group(
    process: subprocess.Popen[Any], sig: int = signal.SIGKILL
) -> None:
    """Signal the group *process* leads, unless the test has already reaped it.

    ``returncode`` is read rather than ``poll()`` called: ``poll()`` would
    reap a leader that has exited and free its pid before the signal, while
    the zombie keeps the group's id reserved and lets what is left of the
    group be killed.
    """
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def signal_process(pid: int, started: int, sig: int = signal.SIGKILL) -> None:
    """Signal *pid* if it is still the process that started at *started*."""
    if not is_same_process(pid, started):
        return
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def signal_group(
    pgid: int, *members: tuple[int, int], sig: int = signal.SIGKILL
) -> None:
    """Signal group *pgid* if one of its *members* still runs in it.

    Each member is a ``(pid, start time)`` the test read while it ran; the
    group's leader is ``(pgid, its start time)``.  A member still in the group
    keeps its id from being reused, so the group is still the test's.
    """
    for pid, started in members:
        fields = _stat(pid)
        if fields is None or int(fields[19]) != started or int(fields[2]) != pgid:
            continue
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        return
