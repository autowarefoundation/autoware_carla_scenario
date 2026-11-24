"""CARLA UE5 server lifecycle management."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar, List, Optional

import typesafe_carla.carla as carla

from dotenv import find_dotenv, load_dotenv

from .carla_install_paths import installed_executable
from .carla_release import client_server_mismatch

# The run's own ``.env``: the nearest one from the working directory up, not
# the one nearest this file (find_dotenv()'s default), which is no file in a
# global, Docker or ``uv tool`` install and the framework checkout's in an
# editable one.  The environment wins over it.
load_dotenv(find_dotenv(usecwd=True), override=False)

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
#: *argv[2]* seconds and then sends SIGKILL.  While it waits it looks for
#: the rest of the group every 0.2 s -- sleeping once the launcher has
#: exited, as ``child.wait()`` then returns at once (#2198).
#: ``CarlaUnreal.sh`` runs UE5 as a child rather than ``exec``-ing it, which is
#: why the whole group is signalled.
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
    "        if not (name.isascii() and name.isdigit()) or int(name) == me:\n"
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
    "        else:\n"
    "            time.sleep(0.2)\n"
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


def _unbracketed_host(host: str) -> str:
    """*host* without the brackets of an IPv6 literal (``"[::1]"`` is
    ``"::1"``), as the socket API and ``carla.Client`` take it; any other
    host as it is."""
    if len(host) >= 2 and host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _host_port(host: str, port: int) -> str:
    """*host*:*port* for a message, an IPv6 address in brackets
    (``[::1]:2000``)."""
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


#: The ``CarlaUnreal.sh`` argument that sets the port CARLA listens on.
_RPC_PORT_ARG = "-carla-rpc-port="


def _check_rpc_port_args(extra_args: List[str], port: int) -> None:
    """Refuse an argument of *extra_args* that moves the port CARLA listens on
    away from *port*.

    Unreal reads ``-carla-rpc-port=`` case-insensitively, so it is matched
    whatever its case, and with a second leading ``-``
    (``--carla-rpc-port=``) too.

    Raises:
        ValueError: If one of them names another port, or no port: CARLA
            would listen there while the manager waited at *port*.
    """
    for arg in extra_args:
        spelled = arg.lower()
        dashes = 1 if spelled.startswith("-" + _RPC_PORT_ARG) else 0
        if not spelled[dashes:].startswith(_RPC_PORT_ARG):
            continue
        value = arg[dashes + len(_RPC_PORT_ARG) :]
        digits = value != "" and all(c in "0123456789" for c in value)
        if not (digits and int(value) == port):
            raise ValueError(
                f"extra_args has {arg!r}, but the server's port is {port}: CARLA "
                f"would listen on {value!r} while the manager waits on {port}. "
                "Set the port (server.port) instead, or make them the same."
            )


def _is_loopback_literal(name: str) -> bool:
    """Whether *name* (lower case, unbracketed) is ``localhost`` or a loopback
    address: IPv4 ``127.0.0.0/8``, IPv6 ``::1`` or IPv4-mapped
    ``::ffff:127.x.y.z``, spelled out without ``ipaddress``.
    Not serve_bind's ``is_loopback``: this one also takes ``*.localhost``
    names, and takes neither a scoped address (``::1%lo``) nor an
    IPv4-mapped one written in hex (``::ffff:7f00:1``).
    """
    if name == "localhost" or name.endswith(".localhost"):
        return True
    if name in ("::1", "0:0:0:0:0:0:0:1"):
        return True
    if name.startswith("::ffff:"):
        name = name[len("::ffff:") :]
    parts = name.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part or len(part) > 3 or any(c not in "0123456789" for c in part):
            return False
        if int(part) > 255:
            return False
    return int(parts[0]) == 127


def _is_local_host(host: str) -> bool:
    """Whether *host* names this machine, so that a CARLA server
    :class:`CarlaServerManager` launches here is the one reached at it.

    The rule (docs/installation.md, "CARLA Simulator"):

    * ``localhost`` (and ``*.localhost``), an IPv4 loopback address
      (``127.0.0.0/8``), ``::1`` (bracketed or not) or an IPv4-mapped
      loopback address, without asking the system; an empty host, which the
      socket API reads as this machine, too;
    * this machine's own name (``socket.gethostname()``);
    * any other name or address that resolves to an address of one of this
      machine's interfaces (``0.0.0.0``, the LAN address, a name for it in
      ``/etc/hosts``): one a socket can be bound to here.

    Anything else -- an address of another machine, a name that does not
    resolve -- is remote: a server launched here would never answer there.
    """
    name = host.strip().strip("[]").rstrip(".").lower()
    if name == "" or _is_loopback_literal(name):
        return True
    if name == socket.gethostname().strip().rstrip(".").lower():
        return True
    # bind() resolves the name for its family and is refused (EADDRNOTAVAIL,
    # or a resolver error) unless the address is one of this machine's.
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind((name, 0))
            return True
        except (OSError, ValueError):
            # ValueError: a name the IDNA codec refuses (a label too long).
            continue
    return False


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

    A server is only ever launched on this machine, so only for a *host* that
    names it (``localhost``, a loopback address, this machine's name or one of
    its addresses).  For any other host :meth:`start` uses the server already
    listening there, and raises at once when there is none: it never launches
    one here and waits for it to appear at the remote address.

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
    #: How long :meth:`start` keeps pinging a local port that accepts a
    #: connection but did not answer as CARLA within one ping, before it
    #: gives up (with *reuse_if_running*): a busy CARLA is slow to answer.
    REUSE_PING_S: ClassVar[float] = 30.0

    host: str
    port: int
    timeout: float
    extra_args: List[str]
    reuse_if_running: bool
    _process: Optional[subprocess.Popen[bytes]]
    _reused: bool
    _launcher: Optional[Path]

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
            host: Hostname to connect / bind the CARLA server to.  An IPv6
                address may be bracketed (``"[::1]"``); :attr:`host` holds it
                without the brackets, as a client connects to it.
            port: TCP port for the CARLA RPC server.
            timeout: Seconds to wait for the server to become reachable
                after launching it.  Defaults to 120 s (UE5 can be slow
                on first boot).
            extra_args: Additional CLI arguments passed to ``CarlaUnreal.sh``.
                A ``-carla-rpc-port=`` among them (in any case, or as
                ``--carla-rpc-port=``) must name *port*: one that names
                another makes :meth:`start` raise ``ValueError`` rather than
                wait for a server on the wrong port.
            reuse_if_running: When ``True`` (default), :meth:`start` skips
                launching a new process if a CARLA server is already
                reachable on *host*:*port*.  :meth:`stop` is also a no-op
                in that case so the externally-managed server is left alive.
        """
        self.host = _unbracketed_host(host)
        self.port = port
        self.timeout = timeout
        self.extra_args = []
        if extra_args:
            self.extra_args = extra_args
        self.reuse_if_running = reuse_if_running
        self._process = None
        self._reused = False  # True when we connected to an existing server
        self._launcher = None  # The launcher start() last ran.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch (or reuse) the CARLA server.

        Behaviour:
        1. If *reuse_if_running* is ``True`` and the server is already
           reachable, record that we are reusing it and return immediately.
        2. If *host* is not this machine, raise: a server launched here would
           never answer at it.
        3. If something else already listens on *host*:*port*, raise rather
           than launch a server that could not bind it: with
           *reuse_if_running* ``False``, any process accepting a connection
           there; with it ``True``, one that keeps refusing to answer as CARLA
           for :attr:`REUSE_PING_S` (a busy CARLA can miss one ping's 2 s
           deadline, so it is pinged again until then, and reused once it
           answers; a port that stops accepting connections meanwhile, as
           one a server shutting down held, is launched on).
        4. Otherwise, find the launcher (:meth:`executable`), launch the
           process, and poll until the server accepts connections.  Its own
           process is checked before each ping: once it has exited,
           :meth:`start` fails with its exit code, even if a server then
           answers on the port (that one is not the server it launched).

        A remote server is therefore always reused, so a queue never restarts
        one (:meth:`ScenarioQueue._server_down` asks only for a server it
        launched): a run it drops fails as on any reused server.

        Raises:
            ValueError: If a ``-carla-rpc-port=`` in *extra_args* names a
                port other than *port* (checked only when a server is to be
                launched).
            RuntimeError: If *host* is remote and no server answers there,
                the port is taken by a process that is not a CARLA server
                to reuse, there is no launcher (when a new process must be
                launched), the launched process exits before it answers,
                the executable does not exist, or the server does not become
                reachable within *timeout* seconds.
        """
        remote = not _is_local_host(self.host)
        # A remote server is the one there or none: never launched here, so
        # reuse_if_running=False has nothing to launch instead of it.
        if (self.reuse_if_running or remote) and self._ping():
            self._reuse()
            return
        if remote:
            raise RuntimeError(
                f"No CARLA server reachable at {_host_port(self.host, self.port)}; the runner "
                "only launches CARLA for a local host (localhost, a loopback "
                "address or this machine's own name or address). Start the server "
                "there, or check server.host and server.port."
            )
        # A server launched onto a taken port dies on it, while the process
        # holding it may answer the pings meant for the new one (#2233).
        if self._tcp_connects():
            if not self.reuse_if_running:
                raise RuntimeError(
                    f"Port {self.port} on {self.host} is in use by another process; "
                    "a CARLA server launched there could not listen on it. Stop "
                    "that process, choose another server.port, or, if it is a CARLA "
                    "server to use, set reuse_if_running=True."
                )
            if self._ping_for(self.REUSE_PING_S):
                self._reuse()
                return
            # A port freed while it was pinged (a server still shutting
            # down, as a queue's restart stops one) is launched on.
            if self._tcp_connects():
                raise RuntimeError(
                    f"Port {self.port} on {self.host} is in use by a process that "
                    f"did not answer as a CARLA server within {self.REUSE_PING_S:g}s; "
                    "a CARLA server launched there could not listen on it. Stop "
                    "that process, wait until the CARLA server there answers, or "
                    "choose another server.port."
                )

        # Refused before anything is launched: a server told to listen
        # elsewhere would never answer where it is waited for.
        _check_rpc_port_args(self.extra_args, self.port)
        exe_path = self.executable()
        if exe_path is None:
            raise RuntimeError(
                f"No CARLA server to use at {_host_port(self.host, self.port)}: "
                f"'{self.ENV_VAR}' is not set and no CARLA is installed. Start the "
                "server (on an x86_64 host if this machine is not one, then point "
                "server.host at it), run `scenario-setup` to download one, or set "
                f"{self.ENV_VAR} to the path of CarlaUnreal.sh."
            )
        if not exe_path.exists():
            raise RuntimeError(f"CARLA executable not found: {exe_path}")
        self._launcher = exe_path

        cmd = [str(exe_path)] + self.extra_args
        # The server listens where the client will look for it.  Spelled
        # exactly so, even when extra_args names the same port otherwise
        # (-CARLA-RPC-PORT=, a leading zero): it is the marker the SLURM
        # cleanup finds this server's processes by (slurm.ports, slurm.scripts).
        rpc_port_arg = f"{_RPC_PORT_ARG}{self.port}"
        if rpc_port_arg not in self.extra_args:
            cmd.append(rpc_port_arg)
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
            self._check_client_matches()
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
        # start_new_session makes the watchdog its group's leader, so the
        # group's id is its pid.  os.getpgid() cannot stand in for this: once
        # poll() (is_alive(), _wait_until_ready()) has reaped the watchdog it
        # raises, though CarlaUnreal.sh's children (UE5) may still run in the
        # group (#1766).  Linux does not reuse a pid while a group still has
        # it as its id, so signalling it reaches only what start() launched.
        pgid = self._process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The whole group already exited.
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait()

        # Wait for all processes in the group (UE5 binary, worker threads,
        # etc.) to finish.  CarlaUnreal.sh may exit before the UE5 binary
        # completes its cleanup, so we poll the group explicitly.
        if not self._wait_for_pgid_exit(pgid, timeout=5.0):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._process = None
        atexit.unregister(self.stop)

    @property
    def reused(self) -> bool:
        """Whether :meth:`start` reused a server already running, rather than
        launching one: a reused server is not this manager's to stop."""
        return self._reused

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

    def _check_running(self) -> None:
        """Raise if the process :meth:`start` launched has exited."""
        process = self._process
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"CARLA server exited with code {process.returncode} "
                f"before it became reachable at {_host_port(self.host, self.port)}"
                f"{self._log_hint()}"
            )

    def _reuse(self) -> None:
        """Record that the server already running is reused."""
        self._check_client_matches()
        self._reused = True
        if self.extra_args:
            logger.warning(
                "Reusing the CARLA server at %s: it was not started with %s; "
                "start it with them for them to apply",
                _host_port(self.host, self.port),
                " ".join(self.extra_args),
            )

    def _check_client_matches(self) -> None:
        """Raise when the server speaks another CARLA branch's RPC than the client.

        A client of the wrong branch (``TYPESAFE_CARLA_CARLA_REF``) fails on
        the run's first world settings with a msgpack error naming neither;
        this says which client to select (carla_release). A server that
        does not answer the question is left to fail where it is used: the
        check is about the branch, not about reaching the server.
        """
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(10.0)
            versions = (client.get_client_version(), client.get_server_version())
        except Exception as exc:  # typesafe_carla raises RuntimeError and others
            logger.debug("CARLA version check skipped: %s", exc)
            return
        problem = client_server_mismatch(*versions)
        if problem is not None:
            raise RuntimeError(problem)

    def _log_hint(self) -> str:
        """Where the launched CARLA's own log is, as an error points to it
        (its output is not kept: the launcher's goes to ``/dev/null``)."""
        launcher = self._launcher
        if launcher is None:
            return ""
        log = launcher.parent / "CarlaUnreal" / "Saved" / "Logs"
        return f"; its log, if it wrote one, is under {log}"

    def _tcp_connects(self) -> bool:
        """Whether a plain TCP connection to *host*:*port* succeeds: some
        process listens there, CARLA or not."""
        try:
            socket.create_connection((self.host, self.port), timeout=2.0).close()
        except OSError:
            return False
        return True

    def _ping_for(self, seconds: float) -> bool:
        """Ping until the server answers (``True``) or *seconds* have passed,
        or the port stops accepting connections (``False``)."""
        deadline = time.monotonic() + seconds
        while True:
            if self._ping():
                return True
            if time.monotonic() >= deadline or not self._tcp_connects():
                return False
            time.sleep(1.0)

    def _ping(self) -> bool:
        """Try to connect to the CARLA RPC port; return True on success."""
        # A closed port is answered here, not by the CARLA client: on arm64
        # its call on a refused connection can abort the process rather than
        # raise (maps.opendrive._from_server checks the same way).
        if not self._tcp_connects():
            return False
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(2.0)
            client.get_server_version()
            return True
        except Exception:
            return False

    def _wait_until_ready(self) -> None:
        """Poll until the server accepts connections or the timeout expires.

        The launched process is checked before each ping, and again after
        the one that answers: a server that answers once it has exited is
        another process's, not the one :meth:`start` launched (#2233).
        """
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self._check_running()
            if self._ping():
                self._check_running()
                return
            time.sleep(1.0)
        raise RuntimeError(
            f"CARLA server did not become reachable within {self.timeout}s "
            f"at {_host_port(self.host, self.port)}"
        )
