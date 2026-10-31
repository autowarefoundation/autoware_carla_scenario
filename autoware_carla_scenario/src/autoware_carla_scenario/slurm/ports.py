"""The ports a SLURM task's CARLA and servers listen on, one slot per task.

Several tasks share a node, each with a CARLA of its own, and a user may run
one on the defaults (2000, TrafficManager 8100) besides.  A task takes a slot
of :data:`PORT_SLOT_WIDTH` consecutive ports from ``launcher.port_base``:

==========  ==========================================
``base+0``  CARLA RPC (``server.port``)
``base+1``  CARLA streaming (``-carla-streaming-port``)
``base+2``  CARLA secondary (``-carla-secondary-port``)
``base+3``  TrafficManager (``traffic_manager.port``)
``base+4``  an Autoware bridge the run hosts (``autoware.address``)
``base+5``  a driver bridge the run hosts (``driver.bridge.address``)
==========  ==========================================

The slot to try first comes from the job: ``SLURM_ARRAY_JOB_ID`` plus
``SLURM_ARRAY_TASK_ID``, so the tasks of one array start on different slots.
A slot is taken only when no other task holds it -- an exclusive ``flock`` on
``<lock dir>/autoware_carla_scenario-port-<first port>.lock``, released when the
task's process ends, however it ends -- and none of its ports is bound;
otherwise the next one is tried.  The lock directory must be one every job on
the node sees: not ``/tmp`` where a cluster gives each job a private one, so
``/dev/shm/acs-slurm-ports-<uid>`` by default (:func:`default_lock_dir`), or
``launcher.port_lock_dir``.  The default 20000-29999 stays below Linux's ephemeral
range (32768-60999), so an outgoing connection never takes a slot's port.

The arithmetic of the slots is in :mod:`.port_slots`, which the library check
compiles (docs/typecheck.md); this module takes the slots and re-exports it.
"""

from __future__ import annotations

import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .port_slots import (
    PORT_SLOT_WIDTH,
    SlotPorts,
    first_slot,
    lock_name,
    slot_order,
    watchdog_argv,
)

__all__ = [
    "KILL_WAIT_S",
    "PORT_SLOT_WIDTH",
    "CarlaWatchdog",
    "PortSlot",
    "acquire_slot",
    "carla_processes_on_port",
    "default_lock_dir",
    "first_slot",
    "kill_carla_on_port",
]

#: Where the slot locks are when no node-wide directory can be made.
LOCK_DIR = Path("/tmp")


def default_lock_dir(shm: Path = Path("/dev/shm")) -> Path:
    """A node-wide directory for the slot locks: ``<shm>/acs-slurm-ports-<uid>``
    when it can be made and written to, else :data:`LOCK_DIR`.

    ``/dev/shm`` is the node's own even where SLURM gives each job a private
    ``/tmp`` (``job_container/tmpfs``), in which two tasks would not see each
    other's locks.
    """
    path = shm / f"acs-slurm-ports-{os.getuid()}"
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except OSError:
        return LOCK_DIR
    return path if os.access(path, os.W_OK | os.X_OK) else LOCK_DIR


@dataclass
class PortSlot(SlotPorts):
    """A slot of ports this process holds until :meth:`release`."""

    _lock: IO[bytes] | None = None

    def release(self, *, keep_lock: bool = False) -> None:
        """Let the slot go.  With *keep_lock*, only this process's descriptor
        is closed: a process that inherited it (:class:`CarlaWatchdog`) keeps
        the slot locked, which ``LOCK_UN`` would end for every holder."""
        if self._lock is not None:
            try:
                if not keep_lock:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None


def _port_free(port: int) -> bool:
    """Whether nothing listens on *port* (TCP, any address)."""
    for family, address in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue  # no IPv6 on this host
        try:
            # A port left in TIME_WAIT by an earlier CARLA is not taken: CARLA
            # binds with SO_REUSEADDR too.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((address, port))
        except OSError:
            return False
        finally:
            sock.close()
    return True


def acquire_slot(
    base: int,
    slots: int,
    *,
    start: int,
    lock_dir: Path | None = None,
) -> PortSlot:
    """Take the first free slot from *start* on (wrapping round).

    Raises:
        RuntimeError: Every slot is held or has a port in use.
    """
    lock_dir = default_lock_dir() if lock_dir is None else lock_dir
    lock_dir.mkdir(parents=True, exist_ok=True)
    for index, slot_base in slot_order(base, slots, start):
        path = lock_dir / lock_name(slot_base)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
        except OSError:
            continue  # another user's lock file we cannot open: not ours
        stream = os.fdopen(fd, "rb")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            continue
        slot = PortSlot(index=index, base=slot_base, _lock=stream)
        if all(_port_free(port) for port in slot.ports()):
            return slot
        slot.release()
    raise RuntimeError(
        f"No free port slot among {slots} from {base}: every one is held by "
        "another task or has a port in use (launcher.port_base, launcher.port_slots)"
    )


def carla_processes_on_port(port: int) -> list[int]:
    """The pids of the CARLA processes started for RPC port *port*.

    A process that has exited counts no more, reaped or not: a zombie's
    ``cmdline`` is empty, so it never carries the port's marker.
    """
    marker = f"-carla-rpc-port={port}".encode()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if marker in args:
            pids.append(int(entry.name))
    return pids


#: How long :func:`kill_carla_on_port` waits for a ``SIGKILL`` to take effect.
KILL_WAIT_S = 2.0


def _wait_until_gone(port: int, timeout_s: float) -> list[int]:
    """Poll :func:`carla_processes_on_port` until it is empty or *timeout_s*
    has passed; what is left then."""
    deadline = time.monotonic() + timeout_s
    pids = carla_processes_on_port(port)
    while pids and time.monotonic() < deadline:
        time.sleep(0.05)
        pids = carla_processes_on_port(port)
    return pids


def kill_carla_on_port(
    port: int, *, grace_s: float = 5.0, kill_wait_s: float = KILL_WAIT_S
) -> list[int]:
    """Stop whatever CARLA still runs for *port* (TERM, then KILL); the pids
    still running afterwards, empty when it is gone.

    A process ignoring ``SIGTERM`` for *grace_s* gets ``SIGKILL``, which then
    has *kill_wait_s* to take effect.  Only processes of this user can be
    signalled; the port is the task's own slot, so nothing else is started
    with it.
    """
    pids = carla_processes_on_port(port)
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, kill_wait_s)):
        if not pids:
            break
        sent = False
        for pid in pids:
            try:
                os.kill(pid, sig)
                sent = True
            except (ProcessLookupError, PermissionError):
                pass
        if not sent:
            # Nothing to wait for: what exited meanwhile is gone, and another
            # user's process outlives any wait.
            return carla_processes_on_port(port)
        pids = _wait_until_gone(port, wait_s)
    return pids


class CarlaWatchdog:
    """Stops the task's CARLA if the worker dies without cleaning up.

    CARLA runs in a session of its own (``CarlaServerManager.start``), so the
    ``SIGKILL`` that follows a cancel's ``SIGTERM`` after the scheduler's grace
    period (SLURM's ``KillWait``) reaches the worker but not CARLA -- unless
    the site tracks a job's processes by cgroup.  A worker that is too slow to
    stop CARLA within that period (a node under load) would leave it running,
    holding the GPU and the slot's ports.

    The watchdog is a process of its own (a session of its own too, so the
    job's signals miss it) that holds the read end of a pipe the worker holds
    the write end of.  End of file without a ``done`` first means the worker
    died: it then runs :func:`kill_carla_on_port`.  It also holds the slot's
    lock, so no other task takes the slot before CARLA is gone.

    With *port_file* (the task's ``.port`` file, which the worker would have
    removed), the watchdog removes it once CARLA is gone -- never before: the
    file is the job script's pointer to a CARLA still running, so a watchdog
    killed midway must not have dropped it.  A CARLA that survives keeps it.
    """

    def __init__(self, slot: PortSlot, port_file: Path | None = None) -> None:
        pass_fds = [] if slot._lock is None else [slot._lock.fileno()]
        self._process: subprocess.Popen[bytes] | None = subprocess.Popen(
            watchdog_argv(
                sys.executable,
                __name__,
                slot.rpc,
                str(port_file) if port_file is not None else None,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=pass_fds,
        )

    def finish(self) -> None:
        """The worker cleaned up itself: the watchdog has nothing to do."""
        process, self._process = self._process, None
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.write(b"done")
            process.stdin.close()
        except OSError:
            pass  # it is gone: nothing to tell
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass  # exits by itself; not worth holding the worker up


def _watch(port: int, port_file: str | None = None) -> int:
    if sys.stdin.buffer.read().startswith(b"done"):
        return 0
    left = kill_carla_on_port(port)
    if port_file is not None and not left:
        try:
            Path(port_file).unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--watch":
        sys.exit(_watch(int(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else None))
    sys.exit(2)
