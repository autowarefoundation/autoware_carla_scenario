"""CARLA UE5 server lifecycle management."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar, List, Optional

import typesafe_carla.carla as carla

from dotenv import load_dotenv

from .carla_install import installed_executable

load_dotenv(override=False)

logger = logging.getLogger(__name__)


def fog_density_scale_args(scale: float) -> List[str]:
    """``CarlaUnreal.sh`` arguments setting ``carla.Weather.FogDensityScale``.

    How thick a ``fog_density`` is drawn; pass them as *extra_args*.
    """
    return [f"-ExecCmds=carla.Weather.FogDensityScale {scale}"]


#: Runs CARLA's launcher as its child and stops it when the process that
#: started it dies (#707).  CARLA runs in a session of its own, so neither the
#: SIGTERM that ends a run nor the SIGKILL a sweeper sends a job past its
#: timeout reaches it, and ``atexit`` runs on neither.  This watchdog leads
#: CARLA's process group: every second it checks its parent (*argv[1]*) is
#: still there, and once it is gone it sends the group SIGTERM, gives CARLA
#: *argv[2]* seconds and then sends SIGKILL.  ``CarlaUnreal.sh`` runs UE5 as a
#: child rather than ``exec``-ing it, which is why the whole group is signalled.
#:
#: The parent is polled rather than ``prctl(PR_SET_PDEATHSIG)`` asked for (the
#: Autoware launcher's ``_DIE_WITH_PARENT``): that one fires when the *thread*
#: that started the child ends, which would stop a server started from a
#: worker thread that has since finished.  The watchdog's own SIGTERM, which
#: :meth:`CarlaServerManager.stop` sends the group, is caught and ignored --
#: not ``SIG_IGN``, which CARLA would inherit -- so the watchdog lives until
#: CARLA exits and then exits with its code.
_WATCHDOG = (
    "import os, signal, subprocess, sys, time\n"
    "parent, grace = int(sys.argv[1]), float(sys.argv[2])\n"
    "signal.signal(signal.SIGTERM, lambda signum, frame: None)\n"
    "def rest_of_group():\n"
    "    me, group = os.getpid(), os.getpgid(0)\n"
    "    try:\n"
    "        names = os.listdir('/proc')\n"
    "    except OSError:\n"
    "        return True\n"
    "    for name in names:\n"
    "        if not name.isdigit() or int(name) == me:\n"
    "            continue\n"
    "        try:\n"
    "            with open('/proc/%s/stat' % name) as stat:\n"
    "                fields = stat.read().rsplit(')', 1)[1].split()\n"
    "        except (OSError, IndexError):\n"
    "            continue\n"
    "        if int(fields[2]) == group and fields[0] != 'Z':\n"
    "            return True\n"
    "    return False\n"
    "def stop_group():\n"
    "    os.killpg(0, signal.SIGTERM)\n"
    "    deadline = time.monotonic() + grace\n"
    "    while time.monotonic() < deadline and rest_of_group():\n"
    "        try:\n"
    "            child.wait(timeout=0.2)\n"
    "        except subprocess.TimeoutExpired:\n"
    "            pass\n"
    "    os.killpg(0, signal.SIGKILL)\n"
    "if os.getppid() != parent:\n"
    "    sys.exit(143)\n"
    "child = subprocess.Popen(sys.argv[3:])\n"
    "while True:\n"
    "    try:\n"
    "        code = child.wait(timeout=1.0)\n"
    "        break\n"
    "    except subprocess.TimeoutExpired:\n"
    "        pass\n"
    "    if os.getppid() != parent:\n"
    "        stop_group()\n"
    "sys.exit(code if code >= 0 else 128 - code)\n"
)

#: Seconds the watchdog gives CARLA between SIGTERM and SIGKILL.
_WATCHDOG_GRACE_S = 10.0


def _dying_with_this_process(argv: List[str]) -> List[str]:
    """*argv*, run under :data:`_WATCHDOG`: stopped when this process dies.

    ``-I``: the watchdog imports nothing from the run's directory or
    ``PYTHONPATH`` -- a scenario package's ``signal.py`` must not stand in for
    the standard library's.
    """
    return [
        sys.executable,
        "-I",
        "-c",
        _WATCHDOG,
        str(os.getpid()),
        str(_WATCHDOG_GRACE_S),
    ] + argv


class CarlaServerManager:
    """Start and stop a CARLA UE5 server process.

    The path to the server's launcher (``CarlaUnreal.sh``) is read from the
    ``CARLA_EXECUTABLE`` environment variable, or, when that is not set, is the
    one ``scenario-setup`` installed (``~/.autoware_carla_scenario/bin/carla``).

    If *reuse_if_running* is ``True`` (the default) and a CARLA server is
    already reachable on *host*:*port* at the time :meth:`start` is called,
    no new process is launched – the existing server is reused.  This is the
    recommended mode for local development where CARLA may already be running.
    A reused server is never stopped: not by :meth:`stop`, and not when this
    process ends.

    A server this manager launched does not outlive the process that launched
    it, however that process ends: :meth:`stop` (or ``with``, or ``atexit``)
    stops it when the process exits normally, and a watchdog that leads its
    process group stops it when the process is killed (``SIGTERM``, or the
    ``SIGKILL`` a sweeper sends a job past its timeout), within a second or so
    and then a grace period.

    Example – managed lifecycle::

        with CarlaServerManager() as server:
            client = carla.Client(server.host, server.port)
            ...

    Example – reuse an already-running server::

        manager = CarlaServerManager(reuse_if_running=True)
        manager.start()   # no-op if CARLA is already up
        ...
        manager.stop()    # no-op if the server was not launched by us
    """

    ENV_VAR: ClassVar[str] = "CARLA_EXECUTABLE"

    host: str
    port: int
    timeout: float
    extra_args: List[str]
    reuse_if_running: bool
    _process: Optional[subprocess.Popen[bytes]]
    _reused: bool

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 120.0,
        extra_args: Optional[List[str]] = None,
        reuse_if_running: bool = True,
    ) -> None:
        """Initialize the server manager.

        Args:
            host: Hostname to connect / bind the CARLA server to.
            port: TCP port for the CARLA RPC server.
            timeout: Seconds to wait for the server to become reachable
                after launching it.  Defaults to 120 s (UE5 can be slow
                on first boot).
            extra_args: Additional CLI arguments passed to ``CarlaUnreal.sh``.
            reuse_if_running: When ``True`` (default), :meth:`start` skips
                launching a new process if a CARLA server is already
                reachable on *host*:*port*.  :meth:`stop` is also a no-op
                in that case so the externally-managed server is left alive.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.extra_args = []
        if extra_args:
            self.extra_args = extra_args
        self.reuse_if_running = reuse_if_running
        self._process = None
        self._reused = False  # True when we connected to an existing server

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch (or reuse) the CARLA server.

        Behaviour:
        1. If *reuse_if_running* is ``True`` and the server is already
           reachable, record that we are reusing it and return immediately.
        2. Otherwise, find the launcher (:meth:`executable`), launch the
           process, and poll until the server accepts connections.

        Raises:
            RuntimeError: If there is no launcher (when a new process must be
                launched), the executable does not exist, or the server does
                not become reachable within *timeout* seconds.
        """
        if self.reuse_if_running and self._ping():
            self._reused = True
            if self.extra_args:
                logger.warning(
                    "Reusing the CARLA server at %s:%d: it was not started with %s; "
                    "start it with them for them to apply",
                    self.host,
                    self.port,
                    " ".join(self.extra_args),
                )
            return

        exe_path = self.executable()
        if exe_path is None:
            raise RuntimeError(
                f"No CARLA server to use at {self.host}:{self.port}: "
                f"'{self.ENV_VAR}' is not set and no CARLA is installed. Start the "
                "server (on an x86_64 host if this machine is not one, then point "
                "server.host at it), run `scenario-setup` to download one, or set "
                f"{self.ENV_VAR} to the path of CarlaUnreal.sh."
            )
        if not exe_path.exists():
            raise RuntimeError(f"CARLA executable not found: {exe_path}")

        cmd = [str(exe_path)] + self.extra_args
        # The server listens where the client will look for it, unless told otherwise.
        if not any(arg.startswith("-carla-rpc-port=") for arg in self.extra_args):
            cmd.append(f"-carla-rpc-port={self.port}")
        # Without a display UE crashes opening its window; it renders off-screen then.
        headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if headless and "-RenderOffScreen" not in self.extra_args:
            cmd.append("-RenderOffScreen")
        # start_new_session=True puts the watchdog, CarlaUnreal.sh and all its
        # children (the actual UE5 binary) into a new process group so that
        # stop() can kill the entire group with os.killpg(), and the
        # watchdog can do the same if this process dies without stopping it.
        self._process = subprocess.Popen(  # noqa: S603 - the user's own launcher
            _dying_with_this_process(cmd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._reused = False
        # Guarantee cleanup even if __exit__ / stop() is never called explicitly
        # (e.g. pytest interrupted by Ctrl-C or an unhandled exception).
        atexit.register(self.stop)
        try:
            self._wait_until_ready()
        except BaseException:
            # A server that never became reachable (or a Ctrl-C while
            # waiting) is not left for atexit: the caller gets no manager to
            # stop, since a failed __enter__ skips __exit__ (#711).
            self.stop()
            raise

    @classmethod
    def executable(cls) -> Optional[Path]:
        """The launcher :meth:`start` runs: ``CARLA_EXECUTABLE``, else the installed one."""
        configured = os.environ.get(cls.ENV_VAR)
        return Path(configured) if configured else installed_executable()

    def stop(self) -> None:
        """Terminate the CARLA server process group.

        If the server was *reused* (not launched by this manager), this method
        is a no-op so the externally-managed process is left running.

        CarlaUnreal.sh launches the real UE5 binary as a child process.
        Sending SIGTERM only to the shell would leave the UE5 binary running
        as an orphan.  Instead, we send SIGTERM/SIGKILL to the *entire
        process group* created by ``start_new_session=True`` in :meth:`start`.

        After the direct child (CarlaUnreal.sh) exits, the UE5 binary and its
        worker processes may still be shutting down.  We therefore poll the
        process group with signal 0 and send SIGKILL if it has not fully
        disappeared within the grace period.
        """
        if self._reused or self._process is None:
            return
        try:
            pgid = os.getpgid(self._process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._process.wait()

            # Wait for all processes in the group (UE5 binary, worker threads,
            # etc.) to finish.  CarlaUnreal.sh may exit before the UE5 binary
            # completes its cleanup, so we poll the group explicitly.
            if not self._wait_for_pgid_exit(pgid, timeout=5.0):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except ProcessLookupError:
            # Process already exited; nothing to do.
            pass
        self._process = None
        atexit.unregister(self.stop)

    def is_alive(self) -> bool:
        """Return True if the server is reachable (regardless of who started it)."""
        if not self._reused and (
            self._process is None or self._process.poll() is not None
        ):
            return False
        return self._ping()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CarlaServerManager":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_pgid_exit(self, pgid: int, timeout: float) -> bool:
        """Poll until the process group is fully gone or *timeout* expires.

        Uses signal 0 (existence check, no real signal delivered) on the
        group.  Returns ``True`` if the group has disappeared, ``False`` if
        it still exists when the timeout elapses.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)  # raises ProcessLookupError if gone
                time.sleep(0.1)
            except (ProcessLookupError, PermissionError):
                return True
        return False

    def _ping(self) -> bool:
        """Try to connect to the CARLA RPC port; return True on success."""
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(2.0)
            client.get_server_version()
            return True
        except Exception:
            return False

    def _wait_until_ready(self) -> None:
        """Poll until the server accepts connections or the timeout expires."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self._ping():
                return
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"CARLA server exited with code {self._process.returncode} "
                    "before it became reachable"
                )
            time.sleep(1.0)
        raise RuntimeError(
            f"CARLA server did not become reachable within {self.timeout}s "
            f"at {self.host}:{self.port}"
        )
