"""The pure arithmetic of a SLURM task (``slurm.task_timing``, #786), against
what ``slurm.worker`` computed before it was split out, and that the worker
still computes the same through it."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pytest

from autoware_carla_scenario.slurm import task_timing, worker
from autoware_carla_scenario.slurm.task_timing import (
    rss_report,
    start_delay,
    status_fields,
    tree_rss_kb,
    with_port,
)

# -- What slurm.worker computed before #786, verbatim but for the I/O --------


def _master_with_port(address: str, port: int) -> str:
    host, sep, _ = str(address).rpartition(":")
    return f"{host if sep else address}:{port}"


def _master_wait(recorded: str, spacing: float, now: float) -> float:
    try:
        last: Optional[float] = float(recorded.strip())
    except ValueError:
        last = None
    if last is not None and not math.isfinite(last):
        last = None
    return 0.0 if last is None else min(spacing, max(0.0, last + spacing - now))


def _master_status(
    pid: int, status: str, parents: dict[int, int], rss: dict[int, int]
) -> None:
    for line in status.splitlines():
        if line.startswith("PPid:"):
            parents[pid] = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            rss[pid] = int(line.split()[1])


def _master_tree(
    parents: dict[int, int], rss: dict[int, int], root: int
) -> tuple[int, int]:
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    total = 0
    stack = list(children.get(root, []))
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return rss.get(root, 0), total


def _master_report(worker_kb: int, children_kb: int, total_kb: int) -> dict:
    return {
        "peak_rss_mb": round(total_kb / 1024, 1),
        "peak_worker_rss_mb": round(worker_kb / 1024, 1),
        "peak_children_rss_mb": round(children_kb / 1024, 1),
    }


# -- with_port ---------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    ["localhost:5555", "10.0.0.1:1", "host", "", ":7", "a:b:c", "[::1]:9000", "x:"],
)
def test_with_port_matches_master(address: str) -> None:
    assert with_port(address, 20024) == _master_with_port(address, 20024)


def test_with_port_replaces_or_adds_the_port() -> None:
    assert with_port("bridge.local:5555", 20024) == "bridge.local:20024"
    assert with_port("bridge.local", 20024) == "bridge.local:20024"
    assert worker._with_port is with_port


# -- start_delay -------------------------------------------------------------


@pytest.mark.parametrize(
    "recorded",
    [
        "1990.0",  # long ago
        "1998.0",  # 2 s ago
        "2000.0",  # just now
        "2003.0",  # a node whose clock runs 3 s ahead
        "2600.0",  # ... 600 s ahead: no more than the spacing
        "1.0e3",
        "  1999.5 \n",
        "1_999",
        "inf",
        "-inf",
        "nan",
        "Infinity",
        "1e999",  # overflows to inf
        "garbage",
        "1999.0 garbage",
        "",
        "\n",
    ],
)
@pytest.mark.parametrize("spacing", [0.0, 5.0, 30.0])
def test_start_delay_matches_master(recorded: str, spacing: float) -> None:
    assert start_delay(recorded, spacing, 2000.0) == _master_wait(
        recorded, spacing, 2000.0
    )


@pytest.mark.parametrize(
    ("recorded", "now", "expected"),
    [
        ("1990.0", 2000.0, 0.0),
        ("1998.0", 2000.0, 3.0),
        ("2000.0", 2000.0, 5.0),
        # Clock skew: the lock's writer runs ahead of this node.
        ("2600.0", 2000.0, 5.0),
        # ... or behind it by more than the spacing.
        ("1000.0", 2000.0, 0.0),
        ("nan", 2000.0, 0.0),
        ("-inf", 2000.0, 0.0),
        ("not a time", 2000.0, 0.0),
    ],
)
def test_start_delay_is_within_zero_and_the_spacing(
    recorded: str, now: float, expected: float
) -> None:
    assert start_delay(recorded, 5.0, now) == expected


def test_the_worker_waits_what_start_delay_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.slept: list[float] = []

        def time(self) -> float:
            return 2000.0

        def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)

    clock = _Clock()
    monkeypatch.setattr(worker, "time", clock)
    lock = tmp_path / "start.lock"
    lock.write_text("2601.5\n", encoding="utf-8")  # skewed far ahead
    assert worker.wait_for_start_turn(lock, 4.0) == 4.0
    assert clock.slept == [4.0]
    assert float(lock.read_text(encoding="utf-8")) == 2000.0


# -- status_fields -----------------------------------------------------------

_STATUS = """Name:\tbash
Umask:\t0022
State:\tS (sleeping)
Tgid:\t4242
Ngid:\t0
Pid:\t4242
PPid:\t4100
TracerPid:\t0
VmPeak:\t   12000 kB
VmRSS:\t    5120 kB
RssAnon:\t    1024 kB
"""


@pytest.mark.parametrize(
    "status",
    [
        _STATUS,
        # A kernel thread: no VmRSS.
        "Name:\tkthreadd\nPid:\t2\nPPid:\t0\n",
        # No PPid either.
        "Name:\tx\n",
        "",
        # A field that repeats: the last wins.
        "PPid:\t1\nVmRSS:\t10 kB\nPPid:\t7\nVmRSS:\t20 kB\n",
        # Lines that only look alike.
        "TracerPid:\t9\nVmRSSx\n PPid:\t3\nVmRSS:\t+12 kB\n",
    ],
)
def test_status_fields_match_master(status: str) -> None:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    _master_status(1, status, parents, rss)
    assert status_fields(status) == (parents.get(1), rss.get(1))


@pytest.mark.parametrize(
    "status",
    [
        "PPid:\n",  # no value
        "PPid:4100\n",  # no separator
        "VmRSS:\tlots kB\n",
        "PPid:\t1\nVmRSS:\n",
    ],
)
def test_an_unparsable_status_line_fails_as_it_did(status: str) -> None:
    # The sampler counts a failed sample as 0 (`except Exception`); the
    # parser fails on the same lines master's did, as a ValueError.
    with pytest.raises((ValueError, IndexError)):
        _master_status(1, status, {}, {})
    with pytest.raises(ValueError):
        status_fields(status)


def test_status_fields_reads_ppid_and_vmrss() -> None:
    assert status_fields(_STATUS) == (4100, 5120)
    assert status_fields("Name:\tkthreadd\nPPid:\t0\n") == (0, None)


# -- tree_rss_kb -------------------------------------------------------------

_TREES = [
    # root 100: worker -> carla (-> 2 threads' processes), and a sibling tree.
    ({101: 100, 102: 101, 103: 101, 200: 1, 201: 200}, None),
    # Orphans: a parent that is not in the map, a child with no RSS.
    ({101: 100, 102: 999, 103: 101, 104: 103}, {101: 7, 104: 0}),
    # Root with no RSS and no children.
    ({5: 1, 6: 5}, None),
    ({}, None),
]


@pytest.mark.parametrize(("parents", "rss"), _TREES)
@pytest.mark.parametrize("root", [100, 101, 1, 5, 42])
def test_tree_rss_kb_matches_master(
    parents: dict[int, int], rss: Optional[dict[int, int]], root: int
) -> None:
    sizes = rss if rss is not None else {pid: pid * 10 for pid in (*parents, 100)}
    assert tree_rss_kb(parents, sizes, root) == _master_tree(parents, sizes, root)


def test_tree_rss_kb_sums_the_descendants() -> None:
    parents = {101: 100, 102: 101, 103: 101, 200: 1}
    rss = {100: 1000, 101: 2000, 102: 30, 103: 40, 200: 99999}
    assert tree_rss_kb(parents, rss, 100) == (1000, 2070)
    assert tree_rss_kb(parents, rss, 101) == (2000, 70)
    assert tree_rss_kb(parents, rss, 7) == (0, 0)


def test_tree_rss_kb_counts_each_pid_once_in_a_cycle() -> None:
    # /proc read while pids are reused can describe a cycle, on which master's
    # walk never ended; each pid counts once, and the root not at all. (Each
    # pid has one parent, so a cycle the walk reaches runs through the root.)
    rss = {1: 10, 2: 20, 3: 30, 4: 40}
    assert tree_rss_kb({2: 1, 3: 2, 1: 3}, rss, 1) == (10, 50)
    assert tree_rss_kb({1: 1, 2: 1}, rss, 1) == (10, 20)  # a self-parent root
    # A cycle away from the root, and a self-parent, are not reached.
    assert tree_rss_kb({2: 1, 3: 4, 4: 3, 5: 5}, rss, 1) == (10, 20)
    assert _master_tree({2: 1, 3: 4, 4: 3, 5: 5}, rss, 1) == (10, 20)


def test_the_worker_reads_proc_through_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc = tmp_path / "proc"
    for pid, status in {
        "100": "PPid:\t1\nVmRSS:\t1000 kB\n",
        "101": "PPid:\t100\nVmRSS:\t2000 kB\n",
        "102": "PPid:\t101\n",  # a kernel thread
        "103": "PPid:\t101\nVmRSS:\t3 kB\n",
        "self": "PPid:\t0\nVmRSS:\t1 kB\n",  # not a pid
        "104": None,  # gone before its status was read
    }.items():
        (proc / pid).mkdir(parents=True)
        if status is not None:
            (proc / pid / "status").write_text(status)

    real_path = worker.Path
    monkeypatch.setattr(
        worker, "Path", lambda p: real_path(proc) if p == "/proc" else real_path(p)
    )
    assert worker._process_tree_rss_kb(100) == (1000, 2003)


# -- rss_report --------------------------------------------------------------


@pytest.mark.parametrize(
    ("worker_kb", "children_kb", "total_kb"),
    [
        (0, 0, 0),
        (1, 2, 3),
        (1024, 2048, 3072),
        (51, 52, 103),
        (123456, 7890123, 8013579),
    ],
)
def test_rss_report_matches_master(
    worker_kb: int, children_kb: int, total_kb: int
) -> None:
    assert rss_report(worker_kb, children_kb, total_kb) == _master_report(
        worker_kb, children_kb, total_kb
    )


def test_the_sampler_reports_through_rss_report() -> None:
    sampler = worker._RssSampler()
    sampler.peak_worker_kb, sampler.peak_children_kb = 2048, 10240
    sampler.peak_total_kb = 12288
    assert sampler.to_dict() == {
        "peak_rss_mb": 12.0,
        "peak_worker_rss_mb": 2.0,
        "peak_children_rss_mb": 10.0,
    }


def test_task_timing_exports() -> None:
    assert sorted(task_timing.__all__) == task_timing.__all__
    for name in task_timing.__all__:
        assert callable(getattr(task_timing, name))
