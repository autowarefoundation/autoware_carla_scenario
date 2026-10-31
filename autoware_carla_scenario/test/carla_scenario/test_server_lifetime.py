"""A CARLA the run launched does not outlive the run (#707).

The CARLA here is a fake ``CarlaUnreal.sh`` that, like the real one, runs the
"UE5 binary" (a ``sleep``) as a child of its own rather than ``exec``-ing it,
and writes both pids to the test's own directory.  ``_ping`` is replaced by a
check for the "UE5" pid file, so no port is ever listened on and two tests run
in parallel never see each other's server.  Every process a test starts is
killed when it ends, whatever happened.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoware_carla_scenario.examples import run as run_module
from autoware_carla_scenario.server import CarlaServerManager
from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as sweeper

from ._stale_pids import signal_process

#: How long a dead run's CARLA may take to go: the watchdog notices within a
#: second, and the fake's sleep ends on the SIGTERM it then sends.
_GONE_WITHIN = 20.0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _gone(pid: int, timeout: float = _GONE_WITHIN) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_pid(path: Path) -> tuple[int, int] | None:
    """The pid and start time a fake wrote to *path*, or ``None`` until it has."""
    try:
        pid, started = (int(field) for field in path.read_text().split())
    except (FileNotFoundError, ValueError):
        return None
    return pid, started


def _wait_for(path: Path, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        written = _read_pid(path)
        if written is not None:
            return written[0]
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written within {timeout} s")


#: The first argument of ``sh -c``, as a shell word.
_ARG0 = '"$0"'


def _write_pid(target: str) -> str:
    """Shell that writes "$$ <start time>" (field 22 of stat) to *target*.

    With the pid, the start time names the process even once it is gone, so
    the teardown never kills a process that reused the pid (#812).  *target*
    is a shell word; the file is written aside and renamed into place.
    """
    return (
        's=$(cat /proc/$$/stat); s=${s##*") "}; set -- $s; '
        f'echo "$$ ${{20}}" > {target}.tmp && mv {target}.tmp {target}'
    )


class _FakeCarla:
    """A fake ``CarlaUnreal.sh`` in *directory*, and the pids it writes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.launcher_pid = directory / "launcher.pid"
        self.ue5_pid = directory / "ue5.pid"
        self.exe = directory / "CarlaUnreal.sh"
        self.exe.write_text(
            "#!/bin/sh\n"
            f"{_write_pid(shlex.quote(str(self.launcher_pid)))}\n"
            # A child, not exec: the real launcher runs UE5 the same way.
            f"/bin/sh -c '{_write_pid(_ARG0)} && exec sleep 600' "
            f"'{self.ue5_pid}'\n"
        )
        self.exe.chmod(0o755)

    def pids(self) -> list[int]:
        return [_wait_for(self.launcher_pid), _wait_for(self.ue5_pid)]

    def kill(self) -> None:
        """Kill the launcher and UE5, if they are still the processes they were.

        Most tests have seen them go by now, and their pids may be another
        test's (#812).
        """
        for path in (self.launcher_pid, self.ue5_pid):
            written = _read_pid(path)
            if written is not None:
                signal_process(*written)


@pytest.fixture
def fake_carla(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeCarla]:
    carla = _FakeCarla(tmp_path)
    monkeypatch.setenv(CarlaServerManager.ENV_VAR, str(carla.exe))
    yield carla
    carla.kill()


#: A run: launches CARLA, says so, and waits to be killed.  argv: the UE5
#: pid file (the fake's "port"), the port, and a file to mark stop() in.
_RUN = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from autoware_carla_scenario.server import CarlaServerManager

    ue5_pid, port, stopped = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    CarlaServerManager._ping = lambda self: ue5_pid.exists()
    stop = CarlaServerManager.stop

    def recording_stop(self):
        if self._process is not None:
            stopped.write_text("stopped")
        stop(self)

    CarlaServerManager.stop = recording_stop
    {install}
    with CarlaServerManager(port=port, reuse_if_running={reuse}):
        print("ready", flush=True)
        time.sleep(600)
    """
)


def _start_run(
    fake: _FakeCarla, port: int, *, reuse: bool = False, handle_sigterm: bool = False
) -> subprocess.Popen[str]:
    install = (
        "from autoware_carla_scenario.examples.run import _interrupt_on_sigterm\n"
        # Held: a guard collected at once would put the default back.
        "guard = _interrupt_on_sigterm()\n"
        "guard.__enter__()"
        if handle_sigterm
        else ""
    )
    code = _RUN.format(install=install, reuse=reuse)
    run = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(fake.ue5_pid),
            str(port),
            str(fake.directory / "stopped"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert run.stdout is not None
    if run.stdout.readline().strip() != "ready":
        run.kill()
        run.wait()
        raise AssertionError("the run did not start its CARLA")
    return run


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_a_killed_run_takes_the_carla_it_launched_down(
    fake_carla: _FakeCarla, free_port: int, sig: signal.Signals
) -> None:
    run = _start_run(fake_carla, free_port)
    try:
        pids = fake_carla.pids()
        run.send_signal(sig)
        run.wait(10)
        for pid in pids:
            assert _gone(pid), f"CARLA (pid {pid}) outlived {sig.name} to its run"
    finally:
        run.kill()
        run.wait()


def test_sigterm_to_a_run_stops_carla_through_its_cleanup(
    fake_carla: _FakeCarla, free_port: int
) -> None:
    """The runner's SIGTERM raises KeyboardInterrupt: ``with`` blocks unwind."""
    run = _start_run(fake_carla, free_port, handle_sigterm=True)
    try:
        pids = fake_carla.pids()
        run.terminate()
        run.wait(30)
        assert (fake_carla.directory / "stopped").exists(), "stop() did not run"
        for pid in pids:
            assert _gone(pid, timeout=1.0), f"CARLA (pid {pid}) is still running"
    finally:
        run.kill()
        run.wait()


def test_a_carla_the_user_started_is_never_killed(
    fake_carla: _FakeCarla, free_port: int
) -> None:
    """A run that reused a CARLA it did not launch leaves it running, killed or not."""
    user = subprocess.Popen(
        [str(fake_carla.exe)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        pids = fake_carla.pids()
        for sig in (signal.SIGKILL, signal.SIGTERM):
            run = _start_run(fake_carla, free_port, reuse=True)
            try:
                run.send_signal(sig)
                run.wait(10)
            finally:
                run.kill()
                run.wait()
            time.sleep(1.5)  # longer than the watchdog takes to notice
            assert all(_alive(pid) for pid in pids), f"{sig.name}: user's CARLA died"
        assert not (fake_carla.directory / "stopped").exists()
    finally:
        _kill(user.pid)
        user.wait()


class _HungJob:
    """A sweep job that launches CARLA and hangs (a case past its timeout)."""

    def __init__(self, port: int) -> None:
        self.port = port

    def launch(self, batches: object, initial_job_idx: int) -> None:
        with CarlaServerManager(port=self.port, reuse_if_running=False):
            time.sleep(600)


def test_a_sweep_job_killed_for_its_timeout_takes_its_carla_down(
    fake_carla: _FakeCarla, free_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        CarlaServerManager, "_ping", lambda self: fake_carla.ue5_pid.exists()
    )
    ok, code = sweeper._launch_job_isolated(_HungJob(free_port), ("a=1",), 0, timeout=3)
    assert (ok, code) == (False, None)  # timed out
    for pid in fake_carla.pids():
        assert _gone(pid), f"CARLA (pid {pid}) outlived the killed job"


def test_a_start_that_fails_stops_what_it_launched(
    fake_carla: _FakeCarla, free_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CARLA that never answers is stopped by start(), not left to atexit."""
    monkeypatch.setattr(CarlaServerManager, "_ping", lambda self: False)
    manager = CarlaServerManager(port=free_port, timeout=2.0, reuse_if_running=False)
    with pytest.raises(RuntimeError, match="did not become reachable"):
        manager.start()
    for pid in fake_carla.pids():
        assert _gone(pid, timeout=1.0), f"CARLA (pid {pid}) left running"
    assert manager._process is None


def test_a_carla_started_from_a_thread_outlives_the_thread(
    fake_carla: _FakeCarla, free_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watchdog follows the process, not the thread that launched CARLA.

    ``PR_SET_PDEATHSIG`` would have stopped it when the thread ended.
    """
    monkeypatch.setattr(
        CarlaServerManager, "_ping", lambda self: fake_carla.ue5_pid.exists()
    )
    manager = CarlaServerManager(port=free_port, reuse_if_running=False)
    try:
        thread = threading.Thread(target=manager.start)
        thread.start()
        thread.join(30)
        pids = fake_carla.pids()
        time.sleep(1.5)  # longer than the watchdog takes to notice
        assert all(_alive(pid) for pid in pids), "CARLA died with its thread"
    finally:
        manager.stop()
    for pid in pids:
        assert _gone(pid, timeout=1.0), f"CARLA (pid {pid}) outlived stop()"


def test_the_sigterm_handler_is_installed_only_over_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with run_module._interrupt_on_sigterm():
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
            # Once: a second SIGTERM must not cut the clean-up short.
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        # Nor the atexit clean-up that follows the block.
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with run_module._interrupt_on_sigterm():
            assert signal.getsignal(signal.SIGTERM) is run_module._raise_interrupt
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

        def own(signum: int, frame: object) -> None:
            pass

        signal.signal(signal.SIGTERM, own)
        with run_module._interrupt_on_sigterm():
            assert signal.getsignal(signal.SIGTERM) is own
        assert signal.getsignal(signal.SIGTERM) is own
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_a_run_stopped_by_sigterm_exits_143(monkeypatch: pytest.MonkeyPatch) -> None:
    def terminated() -> None:
        raise run_module._Terminated(signal.SIGTERM)

    monkeypatch.setattr(run_module, "_main", terminated)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit) as exit_info:
            run_module.main()
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert exit_info.value.code == 143


def test_ctrl_c_still_ends_a_run_as_a_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "_main", interrupted)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_module.main()
    finally:
        signal.signal(signal.SIGTERM, previous)
