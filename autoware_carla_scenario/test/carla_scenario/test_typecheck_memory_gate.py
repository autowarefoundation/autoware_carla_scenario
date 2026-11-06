"""Codon builds wait for the memory they need (typecheck/memory_gate.py).

The builds of one machine share a directory of holds; a build is admitted when
what the running builds hold and what it needs fit the budget, or when none is
running.  Driven here with a budget given and a directory per test, and with
the second "build" in another thread or process.
"""

from __future__ import annotations

import errno
import fcntl
import logging
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


@pytest.fixture(autouse=True)
def _no_earlier_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts as a process whose builds never timed out on a lock."""
    monkeypatch.setattr(memory_gate, "_LOCK_TIMED_OUT", threading.Event())


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


@pytest.fixture
def default_gate(
    tmp_path: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Put the gate's default directory (none configured) into *tmp_path*."""
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(memory_gate.tempfile, "gettempdir", lambda: str(temp))
    monkeypatch.delenv(GATE_ENV, raising=False)
    gate = gate_dir()
    assert gate is not None and gate.parent == temp
    return gate


@pytest.mark.parametrize("mode", [0o755, 0o775, 0o777])
def test_the_default_directory_others_can_write_to_is_made_private(
    default_gate: Path, fallbacks: Path, mode: int
) -> None:
    # As an older gate left it under a umask of 002 (0o775), or worse.
    default_gate.mkdir()
    (default_gate / "gate.lock").write_text("")
    default_gate.chmod(mode)
    with admitted(2 * GB, budget=_budget(14)):
        assert len(list(default_gate.glob("*.hold"))) == 1
    assert (default_gate.stat().st_mode & 0o777) == 0o700
    assert not fallbacks.exists()


def test_the_default_directory_with_other_files_is_not_touched(
    default_gate: Path, fallbacks: Path
) -> None:
    default_gate.mkdir()
    notes = default_gate / "notes.txt"
    notes.write_text("someone's")
    default_gate.chmod(0o777)
    with admitted(2 * GB, budget=_budget(14)):
        assert len(list(fallbacks.glob("*.hold"))) == 1
    assert (default_gate.stat().st_mode & 0o7777) == 0o777
    assert sorted(default_gate.iterdir()) == [notes]
    assert notes.read_text() == "someone's"


def test_a_directory_others_can_write_to_is_not_used(
    tmp_path: Path, fallbacks: Path
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with admitted(2 * GB, directory=shared, budget=_budget(14)):
        assert not list(shared.iterdir())
        assert len(list(fallbacks.glob("*.hold"))) == 1
    assert (shared.stat().st_mode & 0o7777) == 0o777  # never repaired
    assert (fallbacks.stat().st_mode & 0o777) == 0o700


def test_a_symlinked_directory_is_not_used(tmp_path: Path, fallbacks: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with admitted(2 * GB, directory=link, budget=_budget(14)):
        assert not list(real.iterdir())
        assert len(list(fallbacks.glob("*.hold"))) == 1


def test_a_configured_directory_is_checked_but_never_changed(
    tmp_path: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A team's setgid directory, a colleague's file in it.
    team = tmp_path / "team"
    team.mkdir()
    team.chmod(0o2775)
    report = team / "report.txt"
    report.write_text("a colleague's")
    _foreign(monkeypatch, report)
    monkeypatch.setenv(GATE_ENV, str(team))
    with admitted(2 * GB, budget=_budget(14)):
        assert len(list(fallbacks.glob("*.hold"))) == 1
    assert (team.stat().st_mode & 0o7777) == 0o2775
    assert sorted(team.iterdir()) == [report]
    assert report.read_text() == "a colleague's"


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


def test_a_fifo_named_like_a_hold_is_ignored(tmp_path: Path) -> None:
    fifo = tmp_path / f"{os.getpid()}-9-9.hold"
    os.mkfifo(fifo)

    def never(seconds: float) -> None:
        raise AssertionError("waited for a FIFO")

    with admitted(9 * GB, directory=tmp_path, budget=_budget(1), sleep=never):
        pass
    assert fifo.exists()


def _ungated(tmp_path: Path) -> None:
    """A build that is admitted at once, its body run, and no hold left."""
    ran = False
    with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
        ran = True
    assert ran
    assert not list(tmp_path.glob("*.hold"))


def test_a_lock_that_fails_runs_the_build_ungated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flock(fd: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", flock)
    _ungated(tmp_path)


def test_a_hold_that_cannot_be_written_is_not_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.write
    record = str(2 * GB).encode()

    def write(fd: int, data: bytes) -> int:
        if bytes(data).startswith(record):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(fd, data)

    monkeypatch.setattr(os, "write", write)
    _ungated(tmp_path)


def test_a_listing_that_fails_runs_the_build_ungated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def listdir(path: object = ".") -> list[str]:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(os, "listdir", listdir)
    _ungated(tmp_path)


def test_a_budget_that_fails_runs_the_build_ungated(tmp_path: Path) -> None:
    def budget() -> int:
        raise RuntimeError("no meminfo")

    ran = False
    with admitted(2 * GB, directory=tmp_path, budget=budget):
        ran = True
    assert ran
    assert not list(tmp_path.glob("*.hold"))


def test_what_the_build_raises_propagates_and_its_hold_is_removed(
    tmp_path: Path,
) -> None:
    with pytest.raises(KeyError, match="from the build"):
        with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
            raise KeyError("from the build")
    assert not list(tmp_path.glob("*.hold"))


def test_a_hold_that_cannot_be_removed_masks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.unlink

    def unlink(path: object, *args: object, **kwargs: object) -> None:
        if str(path).endswith(".hold"):
            raise PermissionError(errno.EACCES, "denied")
        real(path, *args, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="from the build"):
        with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
            monkeypatch.setattr(os, "unlink", unlink)
            raise KeyError("from the build")
    with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
        pass  # no exception without one from the build either


def _foreign(monkeypatch: pytest.MonkeyPatch, *paths: Path) -> None:
    """Make *paths* look owned by another user, to the gate's fstat and stat
    (by inode and change time, so it holds as root and not, and not for a
    later file that reuses the inode): ownership cannot be given away
    without privileges."""
    inodes = {(info.st_ino, info.st_ctime_ns) for info in map(Path.lstat, paths)}
    other = os.getuid() + 1
    real_fstat, real_stat = os.fstat, os.stat

    def disguise(result: os.stat_result) -> os.stat_result:
        if (result.st_ino, result.st_ctime_ns) not in inodes:
            return result
        fields = list(result)
        fields[4] = other  # st_uid
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", lambda fd: disguise(real_fstat(fd)))
    monkeypatch.setattr(
        os, "stat", lambda *args, **kwargs: disguise(real_stat(*args, **kwargs))
    )


def _never(seconds: float) -> None:
    raise AssertionError("waited")


def test_another_user_s_hold_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = tmp_path / "gate"
    gate.mkdir(mode=0o700)
    start = memory_gate._start_time(os.getpid())
    planted = gate / f"{os.getpid()}-1-1.hold"  # a live process, its start
    planted.write_text(f"{1000 * GB} {start}")
    _foreign(monkeypatch, planted)
    with admitted(9 * GB, directory=gate, budget=_budget(14), sleep=_never):
        assert not planted.exists()


def test_a_repaired_directory_is_cleared_of_other_users_entries(
    default_gate: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_gate.mkdir()
    planted = default_gate / "1-0-0.hold"
    planted.write_text(str(1000 * GB))
    lock = default_gate / "gate.lock"
    lock.write_text("")
    mine = default_gate / f"{os.getpid()}-7-7.hold"
    mine.write_text("0")
    default_gate.chmod(0o777)
    _foreign(monkeypatch, planted, lock)
    with admitted(9 * GB, budget=_budget(14), sleep=_never):
        assert len(list(default_gate.glob("*.hold"))) == 2  # ours
    assert not planted.exists()
    assert mine.exists()
    assert (default_gate.stat().st_mode & 0o777) == 0o700
    assert not fallbacks.exists()


def test_a_repaired_directory_with_a_lock_it_cannot_remove_is_not_used(
    default_gate: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_gate.mkdir()
    lock = default_gate / "gate.lock"
    lock.mkdir()  # another user's directory, not empty: unlink cannot remove it
    (lock / "inside").write_text("")
    default_gate.chmod(0o777)
    _foreign(monkeypatch, lock)
    with admitted(2 * GB, budget=_budget(14), sleep=_never):
        assert len(list(fallbacks.glob("*.hold"))) == 1
    assert (lock / "inside").exists()


def test_another_user_s_lock_runs_the_build_ungated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = tmp_path / "gate"
    gate.mkdir(mode=0o700)
    lock = gate / "gate.lock"
    lock.write_text("")
    (gate / f"{os.getpid()}-1-1.hold").write_text(str(1000 * GB))
    _foreign(monkeypatch, lock)
    ran = False
    with admitted(9 * GB, directory=gate, budget=_budget(14), sleep=_never):
        ran = True
    assert ran
    assert lock.exists()
    assert len(list(gate.glob("*.hold"))) == 1  # only the one planted


@pytest.mark.parametrize("pid", ["99999999999999999999", str(2**31), "0"])
def test_a_hold_with_no_process_s_pid_is_ignored(tmp_path: Path, pid: str) -> None:
    (tmp_path / f"{pid}-1-1.hold").write_text(str(1000 * GB))
    with admitted(9 * GB, directory=tmp_path, budget=_budget(14), sleep=_never):
        pass
    assert memory_gate._alive(10**20) is False


def test_an_oserror_of_the_build_propagates_unchanged(tmp_path: Path) -> None:
    error = OSError(errno.EIO, "from the build")
    runs = 0
    with pytest.raises(OSError) as raised:
        with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
            runs += 1
            assert len(list(tmp_path.glob("*.hold"))) == 1
            raise error
    assert raised.value is error
    assert raised.value.__context__ is None
    assert runs == 1
    assert not list(tmp_path.glob("*.hold"))


def test_a_configured_directory_others_can_only_read_is_used_as_it_is(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    gate.chmod(0o755)
    with admitted(2 * GB, directory=gate, budget=_budget(14)):
        assert len(list(gate.glob("*.hold"))) == 1
    assert (gate.stat().st_mode & 0o7777) == 0o755


def test_an_older_gate_s_lock_is_made_private(tmp_path: Path) -> None:
    lock = tmp_path / "gate.lock"
    lock.write_text("")
    lock.chmod(0o644)
    with admitted(2 * GB, directory=tmp_path, budget=_budget(14)):
        pass
    assert (lock.stat().st_mode & 0o777) == 0o600


def _hold_lock(path: Path) -> int:
    """Lock *path* through a descriptor of its own (flock locks of two open
    descriptions conflict, even within one process)."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_a_lock_held_elsewhere_runs_the_build_ungated_in_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(memory_gate, "LOCK_TIMEOUT_S", 0.3)
    fd = _hold_lock(tmp_path / "gate.lock")
    try:
        began = time.monotonic()
        ran = False
        with caplog.at_level(logging.WARNING, logger=memory_gate.__name__):
            with admitted(2 * GB, directory=tmp_path, budget=_budget(14), sleep=_never):
                ran = True
        assert ran
        assert 0.3 <= time.monotonic() - began < 10
        assert not list(tmp_path.glob("*.hold"))
        assert "held elsewhere" in caplog.text
    finally:
        os.close(fd)


def test_a_lock_held_briefly_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_gate, "LOCK_TIMEOUT_S", 30.0)
    fd = _hold_lock(tmp_path / "gate.lock")
    release = threading.Timer(0.3, os.close, args=(fd,))
    release.start()
    try:
        with admitted(2 * GB, directory=tmp_path, budget=_budget(14), sleep=_never):
            assert len(list(tmp_path.glob("*.hold"))) == 1  # gated
    finally:
        release.join()


def test_no_temporary_directory_fails_nothing(
    fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gettempdir() -> str:
        raise FileNotFoundError(errno.ENOENT, "No usable temporary directory")

    monkeypatch.setattr(memory_gate.tempfile, "gettempdir", gettempdir)
    monkeypatch.delenv(GATE_ENV, raising=False)
    assert gate_dir() == fallbacks
    with admitted(2 * GB, budget=_budget(14)):
        assert len(list(fallbacks.glob("*.hold"))) == 1


def test_a_failed_repair_keeps_falling_back(
    default_gate: Path, fallbacks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first run makes the directory 0o700 and then cannot remove another
    # user's non-empty gate.lock; the second must not trust it for its mode.
    default_gate.mkdir()
    lock = default_gate / "gate.lock"
    lock.mkdir()
    (lock / "inside").write_text("")
    default_gate.chmod(0o777)
    _foreign(monkeypatch, lock)
    for _ in range(2):
        with admitted(2 * GB, budget=_budget(14), sleep=_never):
            assert len(list(fallbacks.glob("*.hold"))) == 1
        assert (default_gate.stat().st_mode & 0o777) == 0o700
    assert (lock / "inside").exists()
    assert not list(default_gate.glob("*.hold"))


def test_an_older_gate_s_lock_held_through_an_old_descriptor_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_gate, "LOCK_TIMEOUT_S", 0.5)
    monkeypatch.setattr(memory_gate, "_REPLACE_WAIT_S", 0.05)
    lock = tmp_path / "gate.lock"
    lock.write_text("")
    lock.chmod(0o644)
    old = lock.stat().st_ino
    fd = _hold_lock(lock)  # someone opened it while it was 0o644, and kept it
    try:
        began = time.monotonic()
        with admitted(2 * GB, directory=tmp_path, budget=_budget(14), sleep=_never):
            assert len(list(tmp_path.glob("*.hold"))) == 1  # gated
        assert time.monotonic() - began < 0.5
        assert lock.stat().st_ino != old
        assert (lock.stat().st_mode & 0o777) == 0o600
        assert lock.stat().st_nlink == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["gate.lock"]
        with admitted(2 * GB, directory=tmp_path, budget=_budget(14), sleep=_never):
            assert len(list(tmp_path.glob("*.hold"))) == 1  # and again
    finally:
        os.close(fd)


def test_a_lock_that_timed_out_is_not_waited_for_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(memory_gate, "LOCK_TIMEOUT_S", 0.3)
    fd = _hold_lock(tmp_path / "gate.lock")  # 0o600: not replaced
    try:
        began = time.monotonic()
        _ungated(tmp_path)
        assert time.monotonic() - began >= 0.3
        with caplog.at_level(logging.WARNING, logger=memory_gate.__name__):
            for _ in range(3):
                began = time.monotonic()
                _ungated(tmp_path)
                assert time.monotonic() - began < 0.15
        assert caplog.text.count("an earlier build timed out") <= 1
    finally:
        os.close(fd)
    # Released: the next build is gated again.
    with admitted(2 * GB, directory=tmp_path, budget=_budget(14), sleep=_never):
        assert len(list(tmp_path.glob("*.hold"))) == 1


@pytest.mark.parametrize("mode", [0o700, 0o755])
def test_a_hard_linked_lock_is_not_touched_and_the_gate_falls_back(
    tmp_path: Path, default_gate: Path, fallbacks: Path, mode: int
) -> None:
    precious = tmp_path / "precious"
    precious.write_text("mine, elsewhere")
    precious.chmod(0o644)
    default_gate.mkdir()
    os.link(precious, default_gate / "gate.lock")
    default_gate.chmod(mode)
    with admitted(2 * GB, budget=_budget(14), sleep=_never):
        assert len(list(fallbacks.glob("*.hold"))) == 1
    assert (precious.stat().st_mode & 0o777) == 0o644
    assert precious.stat().st_nlink == 2
    assert precious.read_text() == "mine, elsewhere"


def test_a_hard_linked_lock_in_a_configured_directory_runs_ungated(
    tmp_path: Path,
) -> None:
    precious = tmp_path / "precious"
    precious.write_text("mine, elsewhere")
    precious.chmod(0o644)
    gate = tmp_path / "gate"
    gate.mkdir(mode=0o700)
    os.link(precious, gate / "gate.lock")
    _ungated(gate)
    assert (precious.stat().st_mode & 0o777) == 0o644
    assert precious.stat().st_nlink == 2


def _linked_hold(directory: Path, elsewhere: Path) -> Path:
    """A hold of this live process, hard-linked from *elsewhere*."""
    start = memory_gate._start_time(os.getpid())
    elsewhere.write_text(f"{1000 * GB} {start}")
    hold = directory / f"{os.getpid()}-1-1.hold"
    os.link(elsewhere, hold)
    return hold


def test_a_hard_linked_hold_is_not_counted(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir(mode=0o700)
    hold = _linked_hold(gate, tmp_path / "elsewhere")
    with admitted(9 * GB, directory=gate, budget=_budget(14), sleep=_never):
        assert len(list(gate.glob("*.hold"))) == 2
    assert hold.exists()  # a configured directory's: left alone


def test_a_hard_linked_hold_is_removed_from_the_gate_s_own_directory(
    tmp_path: Path, default_gate: Path, fallbacks: Path
) -> None:
    default_gate.mkdir(mode=0o700)
    elsewhere = tmp_path / "elsewhere"
    hold = _linked_hold(default_gate, elsewhere)
    with admitted(9 * GB, budget=_budget(14), sleep=_never):
        assert len(list(default_gate.glob("*.hold"))) == 1  # ours
    assert not hold.exists()
    assert elsewhere.exists()
    assert not fallbacks.exists()
