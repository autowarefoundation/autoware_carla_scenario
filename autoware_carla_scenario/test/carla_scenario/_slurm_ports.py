"""Ports for the SLURM launcher's tests: a range of their own per test (#699).

A test that runs a batch through the fake SLURM gets its tasks' slots from
``launcher.port_base``.  The fake SLURM numbers every test's first job 1000, so
with one fixed base every test's tasks took the same first slots.  The slot
locks keep two *running* tasks apart, but not a test that checks a port after
its task let the slot go: another xdist worker's (or another suite's) task
could have taken it, and the check saw that task's CARLA.

:func:`port_range` hands each test a range of :data:`RANGE_WIDTH` ports that
no other test on the machine holds at the same time -- an exclusive ``flock``
in the node-wide directory the slot locks use -- and :func:`fake_carlas`
counts only the fake CARLAs a given test started.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from autoware_carla_scenario.slurm.ports import (
    PORT_SLOT_WIDTH,
    carla_processes_on_port,
    default_lock_dir,
)

#: Ports per test: ten slots.
RANGE_WIDTH = 100
#: The ``launcher.port_slots`` a range holds.
RANGE_SLOTS = RANGE_WIDTH // PORT_SLOT_WIDTH
#: Away from the defaults and from a real batch's 20000-29999, and below
#: Linux's ephemeral range (32768-60999).
_FIRST = 30000
_RANGES = (32700 - _FIRST) // RANGE_WIDTH


@contextmanager
def port_range(timeout: float = 600.0) -> Iterator[int]:
    """The first port of a range no other test holds until the block ends."""
    lock_dir = default_lock_dir()
    start = os.getpid() % _RANGES  # xdist workers try different ranges first
    deadline = time.monotonic() + timeout
    while True:
        for offset in range(_RANGES):
            base = _FIRST + ((start + offset) % _RANGES) * RANGE_WIDTH
            path = lock_dir / f"autoware_carla_scenario-test-ports-{base}.lock"
            try:
                fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o600)
            except OSError:
                continue  # another user's lock file (the /tmp fallback)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            try:
                yield base
            finally:
                os.close(fd)  # and with it the lock
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"every test port range is held ({lock_dir})")
        time.sleep(0.5)


def fake_carlas(record_dir: Path, port: int) -> list[int]:
    """The fake CARLAs for *port* that a test whose ``FAKE_RECORD_DIR`` is
    *record_dir* started: those its tasks inherited the variable from.

    A CARLA that has exited but is not yet reaped has no environment left,
    so it does not count.
    """
    marker = f"FAKE_RECORD_DIR={record_dir}".encode()
    pids = []
    for pid in carla_processes_on_port(port):
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue  # gone, or not ours to read
        if marker in environ and b"--fake-carla" in args:
            pids.append(pid)
    return pids
