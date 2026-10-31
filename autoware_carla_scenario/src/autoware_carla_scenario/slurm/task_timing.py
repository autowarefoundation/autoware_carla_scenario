"""The arithmetic of a SLURM task's start and its memory report (docs/slurm.md).

The pure half of :mod:`.worker`, which holds the batch's start lock, sleeps
and reads ``/proc``: how long a task waits for its start turn given the start
the lock records, the address a bridge listens on in the task's port slot,
the parent and resident memory a ``/proc/<pid>/status`` gives, the resident
memory of a process and of its descendants, and the peaks the task reports
in megabytes.  Nothing here touches a file, a clock or a process, so the
library check compiles it (docs/typecheck.md); :mod:`.worker` uses it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

__all__ = [
    "rss_report",
    "start_delay",
    "status_fields",
    "tree_rss_kb",
    "with_port",
]


def with_port(address: str, port: int) -> str:
    """*address* (``host:port``, or a bare host) with its port set to *port*."""
    host, sep, _ = str(address).rpartition(":")
    return f"{host if sep else address}:{port}"


def start_delay(recorded: str, spacing: float, now: float) -> float:
    """How long a task waits to start *spacing* seconds after the last one.

    *recorded* is the start lock's text: the last start, in seconds since the
    epoch, as ``time.time()`` gave it.  Text that is no number, or no finite
    one, counts as no start (no wait).  The wait is never more than
    *spacing*, however far ahead the clock of the node that wrote it ran
    (#784), nor less than 0.  *spacing* is a finite number, at least 0
    (:func:`.worker.wait_for_start_turn` checks it).
    """
    try:
        last = float(recorded.strip())
    except ValueError:
        return 0.0
    if not math.isfinite(last):
        return 0.0  # "inf"/"nan" is no start time: as if none was
    return min(spacing, max(0.0, last + spacing - now))


def _status_value(line: str) -> int:
    fields = line.split()
    if len(fields) < 2:
        raise ValueError(f"no value in /proc status line {line!r}")
    return int(fields[1])


def status_fields(text: str) -> Tuple[Optional[int], Optional[int]]:
    """The parent pid and the resident memory (KiB) a ``/proc/<pid>/status``
    *text* gives (``PPid:``, ``VmRSS:``); ``None`` for a field it lacks (a
    kernel thread has no ``VmRSS:``).  Where a field repeats, the last wins.

    Raises:
        ValueError: a ``PPid:`` or ``VmRSS:`` line holds no integer.
    """
    ppid: Optional[int] = None
    vmrss: Optional[int] = None
    for line in text.splitlines():
        if line.startswith("PPid:"):
            ppid = _status_value(line)
        elif line.startswith("VmRSS:"):
            vmrss = _status_value(line)
    return ppid, vmrss


def tree_rss_kb(
    parents: Dict[int, int], rss: Dict[int, int], root: int
) -> Tuple[int, int]:
    """(*root*'s resident memory, that of all its descendants), in KiB.

    *parents* maps a pid to its parent's, *rss* a pid to its resident
    memory; a pid missing from *rss* counts as 0.  Each descendant counts
    once, and *root* never as its own descendant, even where the map (read
    from ``/proc`` while pids are reused) has a cycle.
    """
    children: Dict[int, List[int]] = {}
    for pid, parent in parents.items():
        if parent not in children:
            children[parent] = []
        children[parent].append(pid)
    seen: Dict[int, bool] = {root: True}
    total = 0
    stack: List[int] = list(children.get(root, []))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen[pid] = True
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return rss.get(root, 0), total


def rss_report(worker_kb: int, children_kb: int, total_kb: int) -> Dict[str, float]:
    """The task's peak resident memory, in MiB to one decimal: of the worker
    and its descendants together, of the worker, and of the descendants."""
    return {
        "peak_rss_mb": round(total_kb / 1024, 1),
        "peak_worker_rss_mb": round(worker_kb / 1024, 1),
        "peak_children_rss_mb": round(children_kb / 1024, 1),
    }
