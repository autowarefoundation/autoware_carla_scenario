"""How the lanelet sweeper runs each case: in a forked child of its own.

:class:`~autoware_carla_scenario.sweeper.lanelet_constraint_sweeper.LaneletConstraintSweeper`
runs every case with :func:`_launch_isolated`, which forks a child into a
process group of its own, so a case that crashes (a ``SIGSEGV`` in CARLA's
client) or hangs takes only itself down.  This module is the process
handling of that: the fork, the stop signals held off across it, the
timeout, stopping a case (``SIGTERM``, then ``SIGKILL`` after
:data:`_STOP_GRACE_SECONDS`) and waiting, after one that could not stop the
CARLA server it launched, until that server is gone
(:func:`_wait_until_carla_gone`, #717).  What a case runs is the caller's
(``run_case``); this module needs nothing of Hydra, so the library check
compiles it (docs/typecheck.md).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

from ..server import _WATCHDOG, _WATCHDOG_GRACE_S

# The sweeper's logger: what it logs of its cases reads as before this was
# split out of it.
logger = logging.getLogger("autoware_carla_scenario.sweeper.lanelet_constraint_sweeper")

#: How long a case is given to unwind -- leave its ``with queue:``, stop the
#: CARLA server and the Autoware stack it started -- after the sweep, being
#: stopped, sends it ``SIGTERM``.  It is sent ``SIGKILL`` after that.
_STOP_GRACE_SECONDS = 30.0

if TYPE_CHECKING:
    # Codon cannot read a global of `server` (whose top level runs
    # load_dotenv()) at this module's top level ("name '_WATCHDOG_GRACE_S' is
    # not defined"), though a function of it can: the check takes the value.
    _CARLA_GONE_TIMEOUT_S = 15.0
else:
    #: How long the sweep waits, after a case that left a CARLA server it
    #: launched running (it timed out or died of a signal), for that server to
    #: exit (:func:`_wait_until_carla_gone`).  The server's watchdog notices
    #: the case is gone within about a second, sends CARLA ``SIGTERM`` and
    #: ``SIGKILL`` :data:`~autoware_carla_scenario.server._WATCHDOG_GRACE_S`
    #: later; the margin covers the noticing and a loaded machine.
    _CARLA_GONE_TIMEOUT_S = _WATCHDOG_GRACE_S + 5.0

#: Set in the environment of each case the sweeper forks: the sweeper retries
#: the case itself (``server.cooldown_max_retries``), so the case's own queue
#: runs it with no retries (``examples/run.py``).  Without it a case that
#: raises would be retried by both, and run ``(1 + r)**2`` times, not ``1 + r``.
SWEEPER_OWNS_RETRIES_ENV = "AUTOWARE_CARLA_SCENARIO_SWEEPER_OWNS_RETRIES"

#: The signals that stop a sweep or its case.  Held off across the fork until
#: the parent is ready to stop the case and the case is ready to be stopped:
#: one arriving in between would otherwise leave the case running unwatched,
#: or interrupt the child before it could leave the sweep's frames.
_STOP_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM})


class _Interrupted(KeyboardInterrupt):
    """A ``KeyboardInterrupt`` raised for a signal other than ``SIGINT``.

    It carries the signal, so the sweep exits ``128 + signal`` (143 for
    ``SIGTERM``) and a case stopped with it unwinds as it does on Ctrl-C.
    """

    signum: int

    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signal.Signals(signum).name}")
        self.signum = signum


def _raise_interrupted(signum: int, _frame: Any) -> None:
    """``SIGTERM`` handler: unwind as Ctrl-C does, once.

    A second ``SIGTERM`` (a supervisor repeating itself) must not cut the
    clean-up the first started short; ``SIGKILL`` remains for that.
    """
    signal.signal(signum, signal.SIG_IGN)
    raise _Interrupted(signum)


def _interrupt_signal(exc: BaseException) -> int:
    """The signal a ``KeyboardInterrupt`` stands for (``SIGINT`` unless told)."""
    return int(getattr(exc, "signum", signal.SIGINT))


def _proc_stat(pid: str) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` after the command name, or *None*."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def _proc_pids() -> list[str]:
    try:
        return [name for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []


def _carla_groups_of(case_pid: int) -> set[int]:
    """The process groups of the CARLA servers *case_pid* launched, still running.

    :class:`~autoware_carla_scenario.server.CarlaServerManager` runs every
    server it launches under :data:`~autoware_carla_scenario.server._WATCHDOG`,
    which leads the server's process group and is given the launching
    process's pid: ``python -I -c <watchdog> <pid> ...``.  A server the case
    only reused (one the user started) has no such watchdog and is not
    counted, whatever host or port it is on.  Linux (``/proc``); elsewhere
    nothing is found.
    """
    head = [b"-I", b"-c", _WATCHDOG.encode(), str(case_pid).encode()]
    groups: set[int] = set()
    for pid in _proc_pids():
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as cmdline:
                argv = cmdline.read().split(b"\0")
        except OSError:
            continue
        if argv[1:5] != head:
            continue
        fields = _proc_stat(pid)
        if fields is not None and fields[0] != "Z":
            groups.add(int(fields[2]))
    return groups


def _live_groups(groups: set[int]) -> set[int]:
    """Those of the process groups *groups* with a process that is not a zombie.

    A zombie is not counted: one nobody reaps (a container with no init)
    would otherwise keep its group alive for ever.
    """
    live: set[int] = set()
    for pid in _proc_pids():
        fields = _proc_stat(pid)
        if fields is not None and fields[0] != "Z" and int(fields[2]) in groups:
            live.add(int(fields[2]))
    return live


def _wait_until_carla_gone(
    case_pid: int, timeout: float = _CARLA_GONE_TIMEOUT_S
) -> bool:
    """Wait, at most *timeout* s, until the CARLA servers *case_pid* launched exit.

    Called once a case has been reaped.  A case that stopped its server
    (it ended, or raised, and its ``with`` blocks ran) left nothing, and this
    returns at once.  One that timed out or died of a signal left the server
    to its watchdog, and the next case must neither reuse it nor race it for
    the port (#717).  A server that still runs at *timeout* is sent
    ``SIGKILL`` with its group.

    Returns:
        Whether the servers exited in time.
    """
    groups = _carla_groups_of(case_pid)
    if not groups:
        return True
    logger.info(
        "Waiting up to %gs for the CARLA server the stopped case (pid %d) "
        "launched to exit.",
        timeout,
        case_pid,
    )
    deadline = time.monotonic() + timeout
    while True:
        groups = _live_groups(groups)
        if not groups:
            return True
        if time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    logger.warning(
        "The CARLA server the stopped case (pid %d) launched still runs %gs "
        "after the case ended: SIGKILL.",
        case_pid,
        timeout,
    )
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return False


def _request_pdeathsig() -> None:
    """Ask the kernel for ``SIGTERM`` when the parent dies (``PR_SET_PDEATHSIG``).

    Linux's ``prctl``, called through ctypes, which Codon has nothing for:
    :func:`_die_with_parent` calls this in the ``else`` branch of
    ``if TYPE_CHECKING:``.  Elsewhere (no ``prctl``) it does nothing.
    """
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL(None).prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
    except (AttributeError, OSError):
        pass


def _die_with_parent(parent_pid: int) -> None:
    """Ask the kernel for ``SIGTERM`` when the sweep dies (Linux).

    Covers a sweep that gets no chance to stop its case (``SIGKILL``, a crash),
    as ``autoware_stack/launcher.py`` does for the Autoware stack.  Exits at
    once if the sweep is already gone.
    """
    if TYPE_CHECKING:
        pass
    else:
        _request_pdeathsig()
    if os.getppid() != parent_pid:
        os._exit(128 + int(signal.SIGTERM))


def _run_in_child(
    run_case: Callable[[], int],
    parent_pid: int,
    signal_mask: set[int | signal.Signals],
) -> None:
    """Run one case in the forked child and ``os._exit`` with its status.

    *run_case* runs the case and returns its exit status.  Never returns:
    whatever the case does, the child must not unwind into the sweep's own
    frames, which it shares with the parent after the fork.  It starts with
    :data:`_STOP_SIGNALS` blocked and restores *signal_mask* once its
    handlers are in place.
    """
    code = 1
    try:
        # A process group of its own, so the parent can signal the whole tree.
        os.setpgrp()
        # Reset the inherited SIGALRM handler so it doesn't fire in the child.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(0)
        # Stopped by the sweep with SIGTERM: unwind as on Ctrl-C, so the
        # case's `with` blocks stop what it started.
        signal.signal(signal.SIGTERM, _raise_interrupted)
        _die_with_parent(parent_pid)
        # The sweep retries the case itself (server.cooldown_max_retries): the
        # case's own queue must not retry it again, or a case that raises
        # would run (1 + r)**2 times instead of 1 + r.
        os.environ[SWEEPER_OWNS_RETRIES_ENV] = "1"
        # A stop signal that came in since the fork is delivered here.
        signal.pthread_sigmask(signal.SIG_SETMASK, signal_mask)
        code = run_case()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except KeyboardInterrupt as exc:
        code = 128 + _interrupt_signal(exc)
        # An exception is always true (`exc or "SIGINT"` was always `exc`).
        logger.warning("Job stopped (%s)", exc)
    except BaseException:  # noqa: BLE001 - the child must reach os._exit
        logger.exception("Job raised an exception in child process")
        code = 1
    finally:
        # No stop signal may raise past here and skip os._exit.
        signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001, S110
                pass
        os._exit(code)


def _signal_job(child_pid: int, sig: signal.Signals) -> None:
    """Send *sig* to the job's process group (its leader's pid)."""
    try:
        os.killpg(child_pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop_job(child_pid: int, grace: float) -> None:
    """Stop a running job: ``SIGTERM`` to its group, ``SIGKILL`` after *grace*.

    Reaps the job before returning.  ``SIGTERM`` first gives the case the
    chance to clean up (its ``with queue:``, ``CarlaServerManager`` and the
    Autoware stack); an interrupt while waiting (a second Ctrl-C) skips the
    rest of the grace period.
    """
    logger.warning(
        "Stopping the running job (pid %d): SIGTERM, SIGKILL after %gs.",
        child_pid,
        grace,
    )
    _signal_job(child_pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    try:
        while True:
            try:
                done, _ = os.waitpid(child_pid, os.WNOHANG)
            except ChildProcessError:
                break
            if done:
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "Job (pid %d) still runs %gs after SIGTERM: SIGKILL.",
                    child_pid,
                    grace,
                )
                _signal_job(child_pid, signal.SIGKILL)
                os.waitpid(child_pid, 0)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        _signal_job(child_pid, signal.SIGKILL)
        try:
            os.waitpid(child_pid, 0)
        except ChildProcessError:
            pass
    # Whatever of its group outlived the job.
    _signal_job(child_pid, signal.SIGKILL)


def _launch_isolated(
    run_case: Callable[[], int], timeout: int
) -> tuple[bool, int | None]:
    """Fork a child process to run a single case, isolating SIGSEGV crashes.

    The case runs in a process group of its own, which a terminal's Ctrl-C
    does not reach.  So when this returns other than normally -- a
    ``KeyboardInterrupt`` (Ctrl-C, or ``SIGTERM`` while the sweep runs), or
    any other exception -- the case is stopped (:func:`_stop_job`) before the
    exception goes on.  A sweep that dies without the chance (``SIGKILL``)
    takes the case with it through ``PR_SET_PDEATHSIG``.

    Args:
        run_case: Runs the case, in the child, and returns its exit status.
        timeout: Hard timeout in seconds (0 = no timeout).

    Once the case is reaped, waits for any CARLA server it launched and left
    running -- a case that timed out or died of a signal could not stop it --
    to exit (:func:`_wait_until_carla_gone`), so whatever runs next neither
    reuses that server nor races it for its port (#717).

    Returns:
        ``(succeeded, exit_code)`` where *exit_code* is 0 on success, the
        positive status the child exited with otherwise (what *run_case*
        returned, or 1 when it raised), a negative int (``-signal_number``)
        on signal death, or ``None`` on timeout.
    """
    parent_pid = os.getpid()
    # Held off until the parent is inside the try that stops the case, and the
    # child has its process group, handlers and PR_SET_PDEATHSIG.
    signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    try:
        child_pid = os.fork()
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, signal_mask)
        raise

    if child_pid == 0:
        try:
            _run_in_child(run_case, parent_pid, signal_mask)
        finally:
            os._exit(1)  # never reached unless the child's own exit was skipped

    # ---- parent process ----
    timed_out = False

    def _kill_child(signum: int, frame: Any) -> None:  # noqa: ARG001
        nonlocal timed_out
        timed_out = True
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except OSError:
            pass

    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        # The child does it too; whichever runs first, the group exists
        # before the parent may need to signal it.
        try:
            os.setpgid(child_pid, child_pid)
        except OSError:
            pass
        if timeout > 0:
            signal.signal(signal.SIGALRM, _kill_child)
            signal.alarm(timeout)
        # A stop signal that came in since the fork is delivered here, where
        # the except below stops the case.
        signal.pthread_sigmask(signal.SIG_SETMASK, signal_mask)

        _, status = os.waitpid(child_pid, 0)
    except BaseException:
        # Interrupted (or anything else): never leave the job running
        # unsupervised.  Its timeout no longer guards it once this returns.
        signal.alarm(0)
        _stop_job(child_pid, _STOP_GRACE_SECONDS)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, signal_mask)

    _wait_until_carla_gone(child_pid)

    # One return of one `int | None`: Codon types a tuple by its values, and
    # neither `(False, None)` nor `(False, -sig)` is the declared one.
    succeeded = False
    exit_code: int | None = None  # timed out
    if timed_out:
        pass
    elif os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        exit_code = -sig
    else:
        code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
        succeeded = code == 0
        exit_code = code
    return succeeded, exit_code
