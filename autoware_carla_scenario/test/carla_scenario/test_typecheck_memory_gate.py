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


@pytest.fixture
def fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the per-user fallbacks into *tmp_path*; the runtime one returned."""
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("HOME", str(home))
    return runtime / "acs-codon-gate"


def test_a_directory_others_can_write_to_is_not_used(
    tmp_path: Path, fallbacks: Path
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with admitted(2 * GB, directory=shared, budget=_budget(14)):
        assert not list(shared.iterdir())
        assert len(list(fallbacks.glob("*.hold"))) == 1
    assert (fallbacks.stat().st_mode & 0o777) == 0o700


def test_a_symlinked_directory_is_not_used(tmp_path: Path, fallbacks: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with admitted(2 * GB, directory=link, budget=_budget(14)):
        assert not list(real.iterdir())
        assert len(list(fallbacks.glob("*.hold"))) == 1


def test_a_configured_directory_is_checked_too(
    tmp_path: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv(GATE_ENV, str(shared))
    with admitted(2 * GB, budget=_budget(14)):
        assert not list(shared.iterdir())
        assert len(list(fallbacks.glob("*.hold"))) == 1


def test_a_directory_of_another_user_is_not_used(
    tmp_path: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As another user, every directory here -- the fallbacks too -- is
    # someone else's: the build runs ungated rather than blocked or failed.
    gate = tmp_path / "gate"
    gate.mkdir(mode=0o700)
    monkeypatch.setattr(os, "getuid", lambda: os.geteuid() + 1)

    def never(seconds: float) -> None:
        raise AssertionError("waited")

    (gate / f"{os.getpid()}-1-1.hold").write_text(str(12 * GB))
    with admitted(9 * GB, directory=gate, budget=_budget(14), sleep=never):
        assert len(list(gate.glob("*.hold"))) == 1  # only the planted one
    assert not list(fallbacks.parent.glob("acs-codon-gate/*.hold"))


def test_a_file_that_is_not_a_hold_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "junk.hold").write_text(str(12 * GB))
    (tmp_path / "-1-1.hold").write_text(str(12 * GB))

    def never(seconds: float) -> None:
        raise AssertionError("waited for a stray file")

    with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=never):
        assert len(list(tmp_path.glob("*.hold"))) == 3
    assert (tmp_path / "junk.hold").exists()


def test_a_symlinked_lock_is_not_followed(tmp_path: Path) -> None:
    # The build runs ungated; the symlink's target is not touched.
    gate = tmp_path / "gate"
    gate.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_text("precious")
    (gate / "gate.lock").symlink_to(target)
    ran = False
    with admitted(2 * GB, directory=gate, budget=_budget(14)):
        ran = True
        assert not list(gate.glob("*.hold"))
    assert ran
    assert target.read_text() == "precious"
    assert (gate / "gate.lock").is_symlink()


def test_a_hold_records_its_process_s_start_time(tmp_path: Path) -> None:
    start = memory_gate._start_time(os.getpid())
    assert start is not None
    with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
        (hold,) = tmp_path.glob("*.hold")
        assert hold.read_text() == f"{2 * GB} {start}"
        assert (hold.stat().st_mode & 0o777) == 0o600


def test_the_hold_of_a_reused_pid_is_forgotten(tmp_path: Path) -> None:
    start = memory_gate._start_time(os.getpid())
    assert start is not None
    stale = tmp_path / f"{os.getpid()}-1-1.hold"
    stale.write_text(f"{12 * GB} {start + 1}")  # this pid, another process

    def never(seconds: float) -> None:
        raise AssertionError("waited for a process that is gone")

    with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=never):
        pass
    assert not stale.exists()


class _Waited(Exception):
    pass


def test_the_hold_of_a_live_process_is_kept(tmp_path: Path) -> None:
    start = memory_gate._start_time(os.getpid())
    live = tmp_path / f"{os.getpid()}-1-1.hold"
    live.write_text(f"{12 * GB} {start}")

    def waits(seconds: float) -> None:
        raise _Waited

    with pytest.raises(_Waited):
        with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=waits):
            pass
    assert live.exists()


def test_without_proc_a_hold_falls_back_to_the_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_gate, "_start_time", lambda pid: None)
    live = tmp_path / f"{os.getpid()}-1-1.hold"
    live.write_text(f"{12 * GB} 1")

    def waits(seconds: float) -> None:
        raise _Waited

    with pytest.raises(_Waited):
        with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=waits):
            pass
    assert live.exists()


def test_the_start_time_is_read_after_the_last_parenthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = " ".join(str(n) for n in range(3, 30))  # field n holds n
    text = f"42 (a) (b c) {fields}\n"
    real = Path.read_text

    def read_text(self: Path, *args: object, **kwargs: object) -> str:
        if str(self) == "/proc/42/stat":
            return text
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    assert memory_gate._start_time(42) == 22
