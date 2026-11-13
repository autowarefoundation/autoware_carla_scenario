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

A task's ROS domains are held the same way (:class:`RosDomains`): each domain
id the task's config names is mapped to the first id from it on that no other
task on the node holds (``<lock dir>/autoware_carla_scenario-ros-domain-<id>.lock``),
so the Autoware stacks and ROS 2 egos of tasks sharing a node never share a
DDS domain.  Not part of the port slot: a slot's index runs to
``launcher.port_slots`` (1000 by default), past the 233 domains ROS 2 has.
The worker reads a domain id an environment holds with :func:`_env_domain_id`
and checks the range of one before it takes it with :func:`_take_domain`.

The arithmetic of the slots is in :mod:`.port_slots`, which the library check
compiles (docs/typecheck.md); this module takes the slots and re-exports it.
"""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from ..constants import MAX_ROS_DOMAIN_ID
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
    "RosDomains",
    "acquire_slot",
    "carla_processes_on_port",
    "default_lock_dir",
    "first_slot",
    "kill_carla_on_port",
]

logger = logging.getLogger(__name__)

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
        # 0o700: Codon has no octal literal.
        path.mkdir(mode=448, exist_ok=True)
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
        stream = _try_lock(lock_dir / lock_name(slot_base))
        if stream is None:
            continue  # held by another task, or another user's lock file
        slot = PortSlot(index=index, base=slot_base, _lock=stream)
        if all(_port_free(port) for port in slot.ports()):
            return slot
        slot.release()
    raise RuntimeError(
        f"No free port slot among {slots} from {base}: every one is held by "
        "another task or has a port in use (launcher.port_base, launcher.port_slots)"
    )


def _try_lock(path: Path) -> IO[bytes] | None:
    """*path* opened and locked (exclusive, not waiting); None when another
    process holds it or it cannot be opened (another user's lock file)."""
    try:
        # 0o644: Codon has no octal literal.
        fd = os.open(path, os.O_RDONLY | os.O_CREAT, 420)
    except OSError:
        return None
    stream = os.fdopen(fd, "rb")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        return None
    return stream


class RosDomains:
    """The ROS domains a task holds, none of them held by another task on the node.

    :meth:`take` maps a domain id the config names (``ros2.domain_id``, an
    ``autoware.launcher.ros_domain_ids`` entry) to the first id from it on
    (up to :data:`~autoware_carla_scenario.constants.MAX_ROS_DOMAIN_ID`, then
    from 0) whose lock no other process holds, and keeps it: the same id
    always maps to the same domain within a task, two different ones to two
    different domains.  The first task on a node gets the ids it asked for.
    The locks go when :meth:`release` is called or the process ends.
    """

    _lock_dir: Path | None
    _highest: int
    _mapped: dict[int, int]
    _held: dict[int, IO[bytes]]

    def __init__(
        self, lock_dir: Path | None = None, *, highest: int = MAX_ROS_DOMAIN_ID
    ) -> None:
        self._lock_dir = lock_dir
        self._highest = highest
        self._mapped = {}
        self._held = {}

    @property
    def mapping(self) -> dict[int, int]:
        """Each configured id taken so far, and the domain it maps to."""
        return dict(self._mapped)

    def take(self, wanted: int) -> int:
        """The domain this task uses for the configured id *wanted*.

        Raises:
            ValueError: *wanted* is not a domain id (0 to the highest).
            RuntimeError: Every domain is held by another task on the node.
        """
        if isinstance(wanted, bool) or not 0 <= wanted <= self._highest:
            raise ValueError(
                f"ROS domain id must be a whole number from 0 to {self._highest}, "
                f"got {wanted!r}"
            )
        if wanted in self._mapped:
            return self._mapped[wanted]
        if self._lock_dir is None:
            self._lock_dir = default_lock_dir()
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        for domain in [*range(wanted, self._highest + 1), *range(wanted)]:
            if domain in self._held:
                continue  # this task's, for another configured id
            stream = _try_lock(
                self._lock_dir / f"autoware_carla_scenario-ros-domain-{domain}.lock"
            )
            if stream is None:
                continue
            self._held[domain] = stream
            self._mapped[wanted] = domain
            if domain != wanted:
                logger.info(
                    "ROS domain %d is another task's on this node: using %d",
                    wanted,
                    domain,
                )
            return domain
        raise RuntimeError(
            f"No free ROS domain on this node: all {self._highest + 1} are held by "
            "this task or other tasks (lower array_parallelism, or use fewer "
            "autoware.launcher.ros_domain_ids)"
        )

    def release(self) -> None:
        """Let every domain go."""
        for stream in self._held.values():
            stream.close()
        self._held.clear()
        self._mapped.clear()


def _env_domain_id(value: str, key: str) -> int:
    """``ROS_DOMAIN_ID`` as an environment holds it (a ``str``), as an int;
    refused naming *key* unless it is a whole number."""
    text = value.strip()
    if not (text.isascii() and text.isdigit()):
        raise ValueError(
            f"{key} must be a ROS domain id, a whole number from 0 to "
            f"{MAX_ROS_DOMAIN_ID}, got {value!r}"
        )
    return int(text)


def _take_domain(domains: RosDomains, wanted: int, key: str) -> int:
    """The task's domain for *wanted*, the value of *key*; an id out of
    range refused naming the key."""
    if not 0 <= wanted <= MAX_ROS_DOMAIN_ID:
        raise ValueError(
            f"{key} must be a ROS domain id, a whole number from 0 to "
            f"{MAX_ROS_DOMAIN_ID}, got {wanted}"
        )
    return domains.take(wanted)


def carla_processes_on_port(port: int) -> list[int]:
    """The pids of the CARLA processes started for RPC port *port*.

    A process that has exited counts no more, reaped or not: a zombie's
    ``cmdline`` is empty, so it never carries the port's marker.
    """
    marker = f"-carla-rpc-port={port}".encode()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not (entry.name.isascii() and entry.name.isdigit()):
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

    _process: subprocess.Popen[bytes] | None

    def __init__(self, slot: PortSlot, port_file: Path | None = None) -> None:
        pass_fds = [] if slot._lock is None else [slot._lock.fileno()]
        # Typed through a local: Codon 0.19 infers Popen's type parameter
        # from a Popen target, not from an Optional one.
        process: subprocess.Popen[bytes] = subprocess.Popen(
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
        self._process = process

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
