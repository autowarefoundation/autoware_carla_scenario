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
``/tmp/autoware_carla_scenario-port-<first port>.lock``, released when the task's
process ends, however it ends -- and none of its ports is bound; otherwise the
next one is tried.  The default 20000-29999 stays below Linux's ephemeral
range (32768-60999), so an outgoing connection never takes a slot's port.
"""

from __future__ import annotations

import fcntl
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

__all__ = [
    "PORT_SLOT_WIDTH",
    "PortSlot",
    "acquire_slot",
    "carla_processes_on_port",
    "first_slot",
    "kill_carla_on_port",
]

#: Ports per slot; the first six are used, the rest are spare.
PORT_SLOT_WIDTH = 10

#: Where the slot locks are (node-local, shared by every user's tasks).
LOCK_DIR = Path("/tmp")


@dataclass
class PortSlot:
    """A slot of ports this process holds until :meth:`release`."""

    index: int
    base: int
    _lock: IO[bytes] | None = None

    @property
    def rpc(self) -> int:
        return self.base

    @property
    def streaming(self) -> int:
        return self.base + 1

    @property
    def secondary(self) -> int:
        return self.base + 2

    @property
    def traffic_manager(self) -> int:
        return self.base + 3

    @property
    def autoware_bridge(self) -> int:
        return self.base + 4

    @property
    def driver_bridge(self) -> int:
        return self.base + 5

    def ports(self) -> tuple[int, ...]:
        return tuple(self.base + k for k in range(6))

    def to_dict(self) -> dict[str, int]:
        return {
            "slot": self.index,
            "rpc": self.rpc,
            "streaming": self.streaming,
            "secondary": self.secondary,
            "traffic_manager": self.traffic_manager,
            "autoware_bridge": self.autoware_bridge,
            "driver_bridge": self.driver_bridge,
        }

    def release(self) -> None:
        if self._lock is not None:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close()
                self._lock = None


def first_slot(environ: dict[str, str] | os._Environ[str], slots: int) -> int:
    """The slot a task tries first: its array job id plus its task id."""
    job = environ.get("SLURM_ARRAY_JOB_ID") or environ.get("SLURM_JOB_ID") or ""
    task = environ.get("SLURM_ARRAY_TASK_ID") or "0"
    try:
        start = int(job) + int(task)
    except ValueError:
        start = os.getpid()
    return start % slots


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
    lock_dir: Path = LOCK_DIR,
) -> PortSlot:
    """Take the first free slot from *start* on (wrapping round).

    Raises:
        RuntimeError: Every slot is held or has a port in use.
    """
    for offset in range(slots):
        index = (start + offset) % slots
        slot_base = base + index * PORT_SLOT_WIDTH
        path = lock_dir / f"autoware_carla_scenario-port-{slot_base}.lock"
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
    """The pids of the CARLA processes started for RPC port *port*."""
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


def kill_carla_on_port(port: int, *, grace_s: float = 5.0) -> list[int]:
    """Stop whatever CARLA still runs for *port* (TERM, then KILL); its pids.

    Only processes of this user can be signalled; the port is the task's own
    slot, so nothing else is started with it.
    """
    pids = carla_processes_on_port(port)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = []
        for pid in pids:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        if not alive or sig == signal.SIGKILL:
            break
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and carla_processes_on_port(port):
            time.sleep(0.2)
        pids = carla_processes_on_port(port)
    return pids
