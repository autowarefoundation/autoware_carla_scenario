"""Admit a Codon build only when the machine has the memory for it.

A Codon build's peak memory grows with what it compiles: about 1.8 GB for a
scenario, about 9 GB for the whole library check (:data:`BASE_BYTES`,
:data:`BYTES_PER_CHAR`).  Builds started side by side -- the xdist workers of
the test suite, the scenario checker's thread pool -- do not know about each
other, and on a 16 GB machine two library checks and a couple of scenarios
together exhaust it: the out-of-memory killer then takes a build (reported as
"Codon was killed (SIGKILL)") or, on a CI runner, the runner itself, which
cancels the job with "The runner has received a shutdown signal".

:func:`admitted` makes every build wait for its share instead.  The builds of
one machine share a directory of holds (:func:`gate_dir`), one file per
running build recording what it was admitted with, and decide under an
exclusive lock on that directory's lock file, so processes and threads alike
see one another.  A build is admitted when what the running builds hold plus
what it needs fits the machine's memory (:func:`budget_bytes`), or when no
other build is running -- one too large for the machine still runs, alone.  A
hold whose process has died is forgotten.

The gate is Linux's (``fcntl``, ``/proc/meminfo``), as Codon is; elsewhere,
or with ``AUTOWARE_CARLA_SCENARIO_CODON_GATE=0``, every build is admitted at
once.
"""

from __future__ import annotations

import itertools
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_BYTES",
    "BYTES_PER_CHAR",
    "GATE_ENV",
    "admitted",
    "budget_bytes",
    "codon_bytes",
    "gate_dir",
]

#: Turns the gate off (``0``), or names the directory of holds.
GATE_ENV = "AUTOWARE_CARLA_SCENARIO_CODON_GATE"

_GIB = 1 << 30

#: Codon's peak memory for a build of next to nothing (a scenario, a module of
#: the library alone), as measured: 1.8 GB.
BASE_BYTES = int(1.9 * _GIB)

#: What each character of compiled source adds to it: the whole library check,
#: 2.8 million characters, peaks at 9.3 GB.
BYTES_PER_CHAR = 2.8 * _GIB / 1_000_000

#: Kept free for everything else on the machine: the test processes, the OS.
_RESERVE_BYTES = int(1.5 * _GIB)

#: How long a build waits between looks at what is running (s).
_POLL_S = 0.5

_COUNTER = itertools.count()


def codon_bytes(source_chars: int) -> int:
    """The memory a Codon build of *source_chars* characters needs at its peak."""
    return int(BASE_BYTES + BYTES_PER_CHAR * max(0, source_chars))


def budget_bytes() -> int:
    """The memory the builds of this machine may hold together: its memory
    (or its cgroup's limit, if lower) less a reserve for everything else."""
    total = _meminfo_total()
    limit = _cgroup_limit()
    if limit is not None and (total is None or limit < total):
        total = limit
    if total is None:
        return 0  # unknown: admit every build (see admitted)
    return max(0, total - _RESERVE_BYTES)


def gate_dir() -> Path | None:
    """The directory of holds this machine's builds share, or ``None`` when
    the gate is off."""
    configured = os.environ.get(GATE_ENV, "")
    if configured == "0":
        return None
    if configured:
        return Path(configured)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"acs-codon-gate-{uid}"


@contextmanager
def admitted(
    need: int,
    *,
    directory: Path | None = None,
    budget: Callable[[], int] = budget_bytes,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    """Hold *need* bytes of the machine's memory while the block runs: wait,
    first, until the running builds leave room for it (or none is running).

    Args:
        need: What the build needs (:func:`codon_bytes`).
        directory: The directory of holds; :func:`gate_dir` by default.
        budget: What the builds may hold together (:func:`budget_bytes`).
        sleep: How to wait between looks (for the tests).
    """
    try:
        import fcntl  # noqa: PLC0415 - Linux's; the gate is off without it
    except ImportError:
        yield
        return
    where = directory if directory is not None else gate_dir()
    if where is None:
        yield
        return
    where.mkdir(parents=True, exist_ok=True)
    hold = where / f"{os.getpid()}-{threading.get_ident()}-{next(_COUNTER)}.hold"
    waited = False
    while True:
        with open(where / "gate.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            held = _held(where)
            room = budget()
            if not held or room <= 0 or sum(held) + need <= room:
                hold.write_text(str(need))
                break
        if not waited:
            logger.info(
                "Codon build waits for memory: %.1f GB needed, %.1f GB of "
                "%.1f GB held by %d running",
                need / _GIB,
                sum(held) / _GIB,
                room / _GIB,
                len(held),
            )
            waited = True
        sleep(_POLL_S)
    try:
        yield
    finally:
        hold.unlink(missing_ok=True)


def _held(where: Path) -> list[int]:
    """What the running builds hold; the holds of dead processes are removed."""
    held: list[int] = []
    for path in where.glob("*.hold"):
        pid = int(path.name.split("-", 1)[0])
        if not _alive(pid):
            path.unlink(missing_ok=True)
            continue
        try:
            held.append(int(path.read_text() or "0"))
        except (OSError, ValueError):
            continue  # being written or removed right now
    return held


def _alive(pid: int) -> bool:
    """Whether the process *pid* is running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _meminfo_total() -> int | None:
    """``MemTotal`` of ``/proc/meminfo``, in bytes."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_limit() -> int | None:
    """The memory limit of this process's cgroup (v2, then v1), if it has one."""
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            text = path.read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < (1 << 60):
            return int(text)
    return None
