"""Ending a child process together with every process it started.

A child started with ``start_new_session`` leads a process group of its own,
which holds what it went on to start; killing the group ends all of it.  The
SUMO network build (:mod:`..traffic.sumo.network`, ``randomTrips.py``) and the
viewer's SLURM submission (:mod:`..ui.slurm_launch`) end a child timed out so;
the Autoware launcher (:mod:`..autoware_stack.launcher`) and the viewer's runner
(:mod:`..ui.runner`) signal a group with :func:`signal_group`.
"""

from __future__ import annotations

import os
import signal
import subprocess

__all__ = ["kill_group", "signal_group"]


def kill_group(process: subprocess.Popen[str]) -> None:
    """SIGKILL *process*'s group (its own session's, ``start_new_session``).

    A group whose every process has ended already is no error; any other
    failure to signal it (a ``PermissionError``) propagates.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Every process of the group has ended already.


def signal_group(pgid: int, sig: int) -> bool:
    """Send *sig* to the process group *pgid*; ``False`` if none of it is left.

    A group whose every process has ended already is no error, as for
    :func:`kill_group`; any other failure to signal it (a ``PermissionError``)
    propagates.  *sig* 0 only asks whether any of the group is left.
    """
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False  # Every process of the group has ended already.
    return True
