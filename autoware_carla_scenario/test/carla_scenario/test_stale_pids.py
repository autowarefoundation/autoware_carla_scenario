"""The teardown helpers never signal a pid the test no longer owns (#812)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from ._stale_pids import (
    _stat,
    is_same_process,
    signal_group,
    signal_process,
    signal_unreaped_group,
    start_time,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A process leading a session of its own, killed and reaped after the test."""
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    yield process
    if process.returncode is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Every kill and killpg the helpers send, recorded instead of sent."""
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: calls.append(("killpg", pgid, sig))
    )
    return calls


def test_an_unreaped_group_is_signalled(sleeper: subprocess.Popen[bytes]) -> None:
    signal_unreaped_group(sleeper)
    assert sleeper.wait(timeout=10) == -signal.SIGKILL


def test_an_exited_but_unreaped_leader_still_has_its_group_signalled(
    sent: list[tuple[str, int, int]],
) -> None:
    """Its zombie keeps the group's id reserved, so the group is still ours."""
    process = subprocess.Popen(["true"], start_new_session=True)
    os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    signal_unreaped_group(process)
    process.wait()
    assert sent == [("killpg", process.pid, signal.SIGKILL)]


def test_a_reaped_leader_is_not_signalled(sent: list[tuple[str, int, int]]) -> None:
    """Its pid is free: the group of that id may be another test's by now."""
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait()
    signal_unreaped_group(process)
    assert sent == []


def test_a_process_is_named_by_its_pid_and_start_time(
    sleeper: subprocess.Popen[bytes],
) -> None:
    started = start_time(sleeper.pid)
    assert started is not None
    assert is_same_process(sleeper.pid, started)
    assert not is_same_process(sleeper.pid, started + 1)


def test_a_command_name_with_parentheses_and_spaces_is_parsed(
    tmp_path: Path,
) -> None:
    """The fields are counted after the last ')', not the first space."""
    sleep = shutil.which("sleep")
    assert sleep is not None
    named = tmp_path / ") 1 2 (x) y"
    shutil.copy(sleep, named)
    process = subprocess.Popen([str(named), "60"], start_new_session=True)
    try:
        fields = _stat(process.pid)
        assert fields is not None
        # ppid (field 4) and pgrp (field 5) land where they belong ...
        assert (int(fields[1]), int(fields[2])) == (os.getpid(), process.pid)
        # ... and so does the start time (field 22): not before this test's.
        started, own = start_time(process.pid), start_time(os.getpid())
        assert started is not None and own is not None
        assert started >= own
    finally:
        process.kill()
        process.wait()


def test_a_pid_now_held_by_another_process_is_not_signalled(
    sleeper: subprocess.Popen[bytes],
) -> None:
    """A start time other than the one recorded: the pid was reused."""
    started = start_time(sleeper.pid)
    assert started is not None
    signal_process(sleeper.pid, started - 1)
    signal_group(sleeper.pid, (sleeper.pid, started - 1))
    assert sleeper.poll() is None

    signal_group(sleeper.pid, (sleeper.pid, started))
    assert sleeper.wait(timeout=10) == -signal.SIGKILL


def test_a_recorded_process_is_signalled(sleeper: subprocess.Popen[bytes]) -> None:
    started = start_time(sleeper.pid)
    assert started is not None
    signal_process(sleeper.pid, started, signal.SIGTERM)
    assert sleeper.wait(timeout=10) == -signal.SIGTERM


def test_a_gone_process_is_not_signalled(sent: list[tuple[str, int, int]]) -> None:
    process = subprocess.Popen(["true"], start_new_session=True)
    started = start_time(process.pid)
    assert started is not None
    process.wait()
    signal_process(process.pid, started)
    signal_group(process.pid, (process.pid, started))
    assert sent == []
