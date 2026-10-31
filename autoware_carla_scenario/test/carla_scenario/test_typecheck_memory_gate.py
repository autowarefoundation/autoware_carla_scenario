"""Codon builds wait for the memory they need (typecheck/memory_gate.py).

The builds of one machine share a directory of holds; a build is admitted when
what the running builds hold and what it needs fit the budget, or when none is
running.  Driven here with a budget given and a directory per test, and with
the second "build" in another thread or process.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import memory_gate
from autoware_carla_scenario.typecheck.memory_gate import (
    GATE_ENV,
    admitted,
    budget_bytes,
    codon_bytes,
    gate_dir,
)

GB = 1 << 30


def _budget(n: int):  # type: ignore[no-untyped-def]
    return lambda: n * GB


def test_the_need_grows_with_the_source() -> None:
    # As measured: next to nothing, ~1.8 GB; the whole library, ~9.3 GB.
    assert 1.8 * GB <= codon_bytes(3_000) <= 2.0 * GB
    assert 9.0 * GB <= codon_bytes(2_800_000) <= 10.0 * GB


def test_builds_that_fit_run_side_by_side(tmp_path: Path) -> None:
    with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
        with admitted(9 * GB, directory=tmp_path, budget=_budget(14)):
            assert len(list(tmp_path.glob("*.hold"))) == 2
    assert not list(tmp_path.glob("*.hold"))


def test_a_build_that_does_not_fit_waits_for_one_to_end(tmp_path: Path) -> None:
    order: list[str] = []
    first_in = threading.Event()

    def first() -> None:
        with admitted(9 * GB, directory=tmp_path, budget=_budget(14)):
            first_in.set()
            time.sleep(0.6)
            order.append("first ends")

    def second() -> None:
        first_in.wait()
        with admitted(9 * GB, directory=tmp_path, budget=_budget(14)):
            order.append("second runs")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert order == ["first ends", "second runs"]


def test_a_build_too_large_for_the_machine_runs_alone(tmp_path: Path) -> None:
    with admitted(30 * GB, directory=tmp_path, budget=_budget(14)):
        pass


def _hold_in_child(directory: str, ready: object, release: object) -> None:
    with admitted(9 * GB, directory=Path(directory), budget=_budget(14)):
        ready.set()  # type: ignore[attr-defined]
        release.wait(10)  # type: ignore[attr-defined]


def test_a_build_in_another_process_is_waited_for(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    child = context.Process(target=_hold_in_child, args=(str(tmp_path), ready, release))
    child.start()
    try:
        assert ready.wait(30)
        looks: list[float] = []

        def sleep(seconds: float) -> None:
            looks.append(seconds)
            release.set()
            time.sleep(0.2)

        with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=sleep):
            pass
        assert looks  # it waited for the other process's build
    finally:
        release.set()
        child.join(10)


def test_the_hold_of_a_dead_process_is_forgotten(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=os.getpid)
    child.start()
    child.join(10)
    assert child.pid is not None
    (tmp_path / f"{child.pid}-1-1.hold").write_text(str(12 * GB))

    def never(seconds: float) -> None:
        raise AssertionError("waited for a dead process")

    with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=never):
        pass
    assert not list(tmp_path.glob(f"{child.pid}-*.hold"))


def test_the_gate_can_be_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GATE_ENV, "0")
    assert gate_dir() is None
    with admitted(30 * GB, budget=_budget(1)):
        with admitted(30 * GB, budget=_budget(1)):
            pass
    monkeypatch.setenv(GATE_ENV, str(tmp_path))
    assert gate_dir() == tmp_path


def test_the_budget_is_the_machine_s_memory_less_a_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_gate, "_meminfo_total", lambda: 16 * GB)
    monkeypatch.setattr(memory_gate, "_cgroup_limit", lambda: None)
    assert 14 * GB <= budget_bytes() < 16 * GB
    monkeypatch.setattr(memory_gate, "_cgroup_limit", lambda: 8 * GB)
    assert budget_bytes() < 8 * GB
