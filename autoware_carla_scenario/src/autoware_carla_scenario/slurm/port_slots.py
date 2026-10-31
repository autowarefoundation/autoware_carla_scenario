"""The arithmetic of a SLURM task's port slots (docs/slurm.md).

The pure half of :mod:`.ports`, which takes a slot (its lock, the test that
its ports are free) and stops a CARLA left on it: how wide a slot is, which
port of a slot serves what, the slot a task tries first, the order it tries
the others in, the name of a slot's lock file, and the command line of the
watchdog that guards a slot.  Nothing here touches a lock, a socket or a
process, so the library check compiles it (docs/typecheck.md);
:mod:`.ports` re-exports what it uses of it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "PORT_SLOT_WIDTH",
    "SlotPorts",
    "first_slot",
    "lock_name",
    "slot_order",
    "watchdog_argv",
]

#: Ports per slot; the first six are used, the rest are spare.
PORT_SLOT_WIDTH = 10


@dataclass
class SlotPorts:
    """The ports of slot *index*, which starts at port *base*.

    :class:`.ports.PortSlot` adds the lock that holds the slot.
    """

    index: int
    base: int

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


def first_slot(environ: Mapping[str, str], slots: int) -> int:
    """The slot a task tries first: its array job id plus its task id."""
    # get(key, "") rather than get(key): one type for `or` (an unset and an
    # empty variable are both falsy, as before).
    job = environ.get("SLURM_ARRAY_JOB_ID", "") or environ.get("SLURM_JOB_ID", "")
    task = environ.get("SLURM_ARRAY_TASK_ID", "") or "0"
    try:
        start = int(job) + int(task)
    except ValueError:
        start = os.getpid()
    return start % slots


def slot_order(base: int, slots: int, start: int) -> list[tuple[int, int]]:
    """The slots to try, from *start* on (wrapping round), as ``(index, first
    port)`` pairs."""
    order: list[tuple[int, int]] = []
    for offset in range(slots):
        index = (start + offset) % slots
        order.append((index, base + index * PORT_SLOT_WIDTH))
    return order


def lock_name(slot_base: int) -> str:
    """The name of the lock file of the slot that starts at port *slot_base*."""
    return f"autoware_carla_scenario-port-{slot_base}.lock"


def watchdog_argv(
    executable: str, module: str, rpc_port: int, port_file: Optional[str] = None
) -> list[str]:
    """The command line of the watchdog (:class:`.ports.CarlaWatchdog`) of the
    CARLA on *rpc_port*: *module* run with ``-m`` by *executable*, and the
    task's ``.port`` file when there is one."""
    argv = [executable, "-m", module, "--watch", str(rpc_port)]
    if port_file is not None:
        argv.append(port_file)
    return argv
