"""The arithmetic of the SLURM launcher's port slots (``slurm.port_slots``, #772),
and that ``slurm.ports`` keeps its public API over it."""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path
from unittest import mock

import pytest

from autoware_carla_scenario.slurm import port_slots, ports
from autoware_carla_scenario.slurm.port_slots import (
    PORT_SLOT_WIDTH,
    SlotPorts,
    first_slot,
    lock_name,
    slot_order,
    watchdog_argv,
)


def test_a_slot_names_its_ports() -> None:
    slot = SlotPorts(index=2, base=20020)
    assert (slot.rpc, slot.streaming, slot.secondary) == (20020, 20021, 20022)
    assert (slot.traffic_manager, slot.autoware_bridge, slot.driver_bridge) == (
        20023,
        20024,
        20025,
    )
    assert slot.ports() == (20020, 20021, 20022, 20023, 20024, 20025)
    assert slot.to_dict() == {
        "slot": 2,
        "rpc": 20020,
        "streaming": 20021,
        "secondary": 20022,
        "traffic_manager": 20023,
        "autoware_bridge": 20024,
        "driver_bridge": 20025,
    }
    assert len(slot.ports()) <= PORT_SLOT_WIDTH


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "3"}, 123),
        ({"SLURM_ARRAY_JOB_ID": "120", "SLURM_ARRAY_TASK_ID": "4"}, 124),
        ({"SLURM_JOB_ID": "1999"}, 999),
        # An empty variable counts as unset, as an unset one does.
        ({"SLURM_ARRAY_JOB_ID": "", "SLURM_JOB_ID": "9", "SLURM_ARRAY_TASK_ID": ""}, 9),
        # No job id, or one that is not a number: the process id.
        ({}, 12345 % 1000),
        ({"SLURM_JOB_ID": "x"}, 12345 % 1000),
    ],
)
def test_first_slot(environ: dict[str, str], expected: int) -> None:
    with mock.patch("os.getpid", return_value=12345):
        assert first_slot(environ, 1000) == expected


def test_slots_are_tried_from_start_wrapping_round() -> None:
    assert slot_order(20000, 4, 2) == [(2, 20020), (3, 20030), (0, 20000), (1, 20010)]
    assert slot_order(20000, 3, 7) == [(1, 20010), (2, 20020), (0, 20000)]
    assert slot_order(20000, 0, 0) == []


def test_lock_name() -> None:
    assert lock_name(20030) == "autoware_carla_scenario-port-20030.lock"


def test_watchdog_argv() -> None:
    assert watchdog_argv("/py", "pkg.mod", 20020) == [
        "/py",
        "-m",
        "pkg.mod",
        "--watch",
        "20020",
    ]
    assert watchdog_argv("/py", "pkg.mod", 20020, "/b/t.port")[-1] == "/b/t.port"


def test_ports_reexports_the_moved_names() -> None:
    assert ports.PORT_SLOT_WIDTH is port_slots.PORT_SLOT_WIDTH
    assert ports.first_slot is port_slots.first_slot
    assert set(ports.__all__) >= {"PORT_SLOT_WIDTH", "PortSlot", "first_slot"}


def test_port_slot_keeps_its_dataclass_api() -> None:
    slot = ports.PortSlot(index=2, base=20020)
    assert isinstance(slot, ports.PortSlot) and isinstance(slot, SlotPorts)
    assert ports.PortSlot(2, 20020) == slot
    assert repr(slot) == "PortSlot(index=2, base=20020, _lock=None)"
    assert [f.name for f in dataclasses.fields(slot)] == ["index", "base", "_lock"]
    assert dataclasses.asdict(slot) == {"index": 2, "base": 20020, "_lock": None}
    assert pickle.loads(pickle.dumps(slot)) == slot
    assert slot.to_dict() == SlotPorts(index=2, base=20020).to_dict()
    slot.release()  # no lock: nothing to do


def test_acquire_slot_takes_the_slots_in_order(tmp_path: Path) -> None:
    held = tmp_path / lock_name(30000 + 1 * PORT_SLOT_WIDTH)
    with mock.patch.object(ports, "_port_free", return_value=True):
        first = ports.acquire_slot(30000, 3, start=1, lock_dir=tmp_path)
        try:
            assert (first.index, first.base) == (1, 30010)
            assert held.exists()
            second = ports.acquire_slot(30000, 3, start=1, lock_dir=tmp_path)
            assert (second.index, second.base) == (2, 30020)
            second.release()
        finally:
            first.release()


def test_the_watchdog_is_started_with_the_built_command_line() -> None:
    slot = ports.PortSlot(index=0, base=20000)
    with mock.patch.object(ports.subprocess, "Popen") as popen:
        ports.CarlaWatchdog(slot)
        ports.CarlaWatchdog(slot, Path("/b/t.port"))
    calls = [call.args[0] for call in popen.call_args_list]
    prefix = [sys.executable, "-m", "autoware_carla_scenario.slurm.ports", "--watch"]
    assert calls == [prefix + ["20000"], prefix + ["20000", "/b/t.port"]]
