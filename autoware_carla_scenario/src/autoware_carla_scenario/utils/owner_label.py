"""The ``<host>:<pid>`` owner label on the containers a run starts.

A container a run starts carries the host name and pid of the process that
started it, so the next run can tell a leftover (its process gone) from
another live run's container on the same host.  The Autoware stack
(:mod:`..autoware_stack.docker`) and the backends the preparer starts
(:mod:`..prepare.docker`) both label theirs so; each under a label name of
its own.

A host and pid name one process only while the runs on a host share its pid
namespace, which is assumed.
"""

from __future__ import annotations

import os
import socket
from typing import Optional

__all__ = ["owner_alive", "owner_value", "process_alive"]


def owner_value() -> str:
    """The owner label of this process's containers, ``<host>:<pid>``."""
    return f"{socket.gethostname()}:{os.getpid()}"


def process_alive(pid: int) -> bool:
    """Whether process *pid* exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's, but alive
    return True


def owner_alive(owner: str) -> Optional[bool]:
    """Whether the process the owner label *owner* names still runs; None
    when that cannot be told (no label, a malformed one -- a pid that is not
    ASCII digits, is 0 or is too large for a pid --, another host's).

    Pid 0 is no process: ``os.kill(0, 0)`` signals the caller's own process
    group, which would read as alive."""
    host, _, pid = owner.rpartition(":")
    if not host or not (pid.isascii() and pid.isdigit()):
        return None
    if host != socket.gethostname():
        return None
    number = int(pid)
    if number <= 0:
        return None
    try:
        return process_alive(number)
    except OverflowError:
        # Larger than a pid_t holds: no process, and no reason to abort the
        # leftover-container cleanup that asks.
        return None
