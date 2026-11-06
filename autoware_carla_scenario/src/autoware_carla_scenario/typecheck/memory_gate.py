"""Admit a Codon build only when the machine has the memory for it.

A Codon build's peak memory grows with what it compiles: about 1.8 GB for a
scenario, about 9 GB for the whole library check (:data:`BASE_BYTES`,
:data:`BYTES_PER_CHAR`).  Builds started side by side -- the xdist workers of
the test suite, the scenario checker's thread pool -- do not know about each
other, and on a 16 GB machine two library checks and a couple of scenarios
together exhaust it: the out-of-memory killer then takes a build (reported as
"Codon was killed (SIGKILL)") or, on a CI runner, the runner itself, which
cancels the job with "The runner has received a shutdown signal".

:func:`admitted` makes every build wait for its share instead.  The builds of
one machine share a directory of holds (:func:`gate_dir`), one file per
running build recording what it was admitted with, and decide under an
exclusive lock on that directory's lock file, so processes and threads alike
see one another.  A build is admitted when what the running builds hold plus
what it needs fits the machine's memory (:func:`budget_bytes`), or when no
other build is running -- one too large for the machine still runs, alone.  A
hold whose process has died is forgotten, and so is one whose pid now names
another process (its start time, from ``/proc/<pid>/stat``, differs).

The directory is private: it is used only if it is a real directory (not a
symlink) owned by this user that no one else can write to (one of this user's
that others can is made 0o700 first), it is used through a descriptor opened
once it is checked, and the lock and hold files are opened without following
symlinks; a lock or hold owned by another user is never trusted (a hold is
removed, a lock runs the build ungated), and a repaired directory is first
cleared of every entry another user owns.  When it is not, the gate warns and falls back to a per-user
directory (under ``$XDG_RUNTIME_DIR``, then ``~/.cache``) and, failing that,
admits the build ungated -- as it does when its own files or the budget fail:
the gate never blocks or fails a check of its own accord.

The gate is Linux's (``fcntl``, ``/proc/meminfo``), as Codon is.  Without
``fcntl``, or with ``AUTOWARE_CARLA_SCENARIO_CODON_GATE=0``, every build is
admitted at once, with no files; where ``fcntl`` exists but the machine's
memory cannot be read (macOS: no ``/proc/meminfo``), the budget is 0 and
every build is admitted at once, though its hold is still recorded.
"""

from __future__ import annotations

import errno
import itertools
import logging
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_BYTES",
    "BYTES_PER_CHAR",
    "GATE_ENV",
    "admitted",
    "budget_bytes",
    "codon_bytes",
    "gate_dir",
]

#: Turns the gate off (``0``), or names the directory of holds.
GATE_ENV = "AUTOWARE_CARLA_SCENARIO_CODON_GATE"

_GIB = 1 << 30

#: Codon's peak memory for a build of next to nothing (a scenario, a module of
#: the library alone), as measured: 1.8 GB.
BASE_BYTES = int(1.9 * _GIB)

#: What each character of compiled source adds to it: the whole library check,
#: 2.8 million characters, peaks at 9.3 GB.
BYTES_PER_CHAR = 2.8 * _GIB / 1_000_000

#: Kept free for everything else on the machine: the test processes, the OS.
_RESERVE_BYTES = int(1.5 * _GIB)

#: How long a build waits between looks at what is running (s).
_POLL_S = 0.5

_COUNTER = itertools.count()


def codon_bytes(source_chars: int) -> int:
    """The memory a Codon build of *source_chars* characters needs at its peak."""
    return int(BASE_BYTES + BYTES_PER_CHAR * max(0, source_chars))


def budget_bytes() -> int:
    """The memory the builds of this machine may hold together: its memory
    (or its cgroup's limit, if lower) less a reserve for everything else."""
    total = _meminfo_total()
    limit = _cgroup_limit()
    if limit is not None and (total is None or limit < total):
        total = limit
    if total is None:
        return 0  # unknown: admit every build (see admitted)
    return max(0, total - _RESERVE_BYTES)


def gate_dir() -> Path | None:
    """The directory of holds this machine's builds share, or ``None`` when
    the gate is off.

    A configured directory (:data:`GATE_ENV`, or :func:`admitted`'s
    *directory*) is held to the same rule as the default: :func:`admitted`
    uses it only if it is a private directory of this user's, and falls back
    otherwise."""
    configured = os.environ.get(GATE_ENV, "")
    if configured == "0":
        return None
    if configured:
        return Path(configured)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"acs-codon-gate-{uid}"


@contextmanager
def admitted(
    need: int,
    *,
    directory: Path | None = None,
    budget: Callable[[], int] = budget_bytes,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    """Hold *need* bytes of the machine's memory while the block runs: wait,
    first, until the running builds leave room for it (or none is running).

    Args:
        need: What the build needs (:func:`codon_bytes`).
        directory: The directory of holds; :func:`gate_dir` by default.
            Used only if it is a real directory owned by this user that its
            group and others cannot write to; else a per-user fallback is,
            or the build runs ungated (with a warning).
        budget: What the builds may hold together (:func:`budget_bytes`).
        sleep: How to wait between looks (for the tests).
    """
    try:
        import fcntl  # noqa: PLC0415 - Linux's; the gate is off without it
    except ImportError:
        yield
        return
    configured = directory if directory is not None else gate_dir()
    if configured is None:
        yield
        return
    gate = _private_dir(configured)
    if gate is None:
        yield  # no private directory to gate in: run ungated (warned)
        return
    where, dir_fd = gate
    try:
        name = f"{os.getpid()}-{threading.get_ident()}-{next(_COUNTER)}.hold"
        try:
            recorded = _wait_for_room(fcntl, dir_fd, name, need, budget, sleep)
        except OSError as error:
            _warn(f"Codon memory gate: {where}: {error}; the build runs ungated")
            recorded = False
        if not recorded:
            yield  # the gate failed (warned): run ungated
            return
        try:
            yield
        finally:
            # Never masks what the block raised: a failure here is only logged.
            try:
                os.unlink(name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
            except OSError as error:
                _warn(f"Codon memory gate: cannot remove {where / name} ({error})")
    finally:
        with suppress(OSError):
            os.close(dir_fd)


def _wait_for_room(
    fcntl: ModuleType,
    dir_fd: int,
    name: str,
    need: int,
    budget: Callable[[], int],
    sleep: Callable[[float], None],
) -> bool:
    """Wait until *need* fits, then record it in the hold *name* of the
    directory *dir_fd*: ``True`` once recorded, ``False`` if *budget* failed
    (warned).  Raises :class:`OSError` when the gate's files fail."""
    start = _start_time(os.getpid())
    record = str(need) if start is None else f"{need} {start}"
    waited = False
    while True:
        lock = os.open(
            "gate.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=dir_fd,
        )
        try:
            owner = os.fstat(lock).st_uid
            if owner != os.getuid():
                raise PermissionError(
                    errno.EPERM, f"gate.lock is owned by uid {owner}, not {os.getuid()}"
                )
            fcntl.flock(lock, fcntl.LOCK_EX)
            held = _held(dir_fd)
            try:
                room = budget()
            except Exception as error:  # noqa: BLE001 - the gate never fails a build
                _warn(
                    f"Codon memory gate: the memory budget failed ({error!r}); "
                    "the build runs ungated"
                )
                return False
            if not held or room <= 0 or sum(held) + need <= room:
                _write_hold(dir_fd, name, record)
                return True
        finally:
            os.close(lock)
        if not waited:
            logger.info(
                "Codon build waits for memory: %.1f GB needed, %.1f GB of "
                "%.1f GB held by %d running",
                need / _GIB,
                sum(held) / _GIB,
                room / _GIB,
                len(held),
            )
            waited = True
        sleep(_POLL_S)


def _fallback_dirs() -> list[Path]:
    """The per-user directories the gate falls back to, in order."""
    dirs: list[Path] = []
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if runtime:
        dirs.append(Path(runtime) / "acs-codon-gate")
    try:
        dirs.append(Path.home() / ".cache" / "autoware_carla_scenario" / "codon-gate")
    except (RuntimeError, KeyError):
        pass  # no home directory
    return dirs


def _private_dir(configured: Path) -> tuple[Path, int] | None:
    """*configured*, if it is (or can be made) a private directory of this
    user's; else the first per-user fallback that is; else ``None``.  With
    the directory comes a descriptor of it, which every later use of the
    directory goes through, so it cannot be swapped once checked."""
    opened = _open_private(configured)
    if isinstance(opened, int):
        return configured, opened
    for candidate in _fallback_dirs():
        if candidate == configured:
            continue
        fallback = _open_private(candidate)
        if isinstance(fallback, int):
            _warn(
                f"Codon memory gate: not using {configured} ({opened}); "
                f"using {candidate} instead"
            )
            return candidate, fallback
    _warn(
        f"Codon memory gate: not using {configured} ({opened}) and no "
        "private per-user directory is available; builds run ungated"
    )
    return None


def _open_private(where: Path) -> int | str:
    """A descriptor of *where*, or why it cannot be the gate's directory: it
    is created (mode 0o700) if missing, and must be a real directory (not a
    symlink) owned by this user; one its group or others may write to (made
    so by an older gate under a lax umask) is made 0o700 first."""
    try:
        where.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(where, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        return str(error)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            problem = f"owned by uid {info.st_uid}, not {os.getuid()}"
        elif info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            os.fchmod(fd, 0o700)
            info = os.fstat(fd)
            if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                problem = f"writable by others (mode {stat.S_IMODE(info.st_mode):o})"
            else:
                problem = ""
                _remove_foreign(fd)  # others could write here until now
        else:
            problem = ""
    except OSError as error:
        problem = f"cannot be checked or made private ({error})"
    if not problem:
        return fd
    with suppress(OSError):
        os.close(fd)
    return problem


def _remove_foreign(dir_fd: int) -> None:
    """Remove every entry of the directory *dir_fd* that another user owns."""
    uid = os.getuid()
    for name in os.listdir(dir_fd):
        with suppress(OSError):
            if os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_uid != uid:
                os.unlink(name, dir_fd=dir_fd)


def _write_hold(dir_fd: int, name: str, record: str) -> None:
    """Create the hold *name* (never through a symlink, never over a file)
    with *record*; one that cannot be written is removed again."""
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=dir_fd,
    )
    written = False
    try:
        os.write(fd, record.encode())
        written = True
    finally:
        os.close(fd)
        if not written:
            with suppress(OSError):
                os.unlink(name, dir_fd=dir_fd)


_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()


def _warn(message: str) -> None:
    """Log *message* as a warning, once per process."""
    with _WARNED_LOCK:
        if message in _WARNED:
            return
        _WARNED.add(message)
    logger.warning(message)


def _held(dir_fd: int) -> list[int]:
    """What the running builds hold, in the directory *dir_fd*; the holds of
    dead processes are removed (and those of a pid since reused by another
    process); a file whose name does not start with a pid, or that is not a
    regular file, is ignored."""
    held: list[int] = []
    for name in os.listdir(dir_fd):
        if not name.endswith(".hold"):
            continue
        try:
            pid = int(name.split("-", 1)[0])
        except ValueError:
            continue  # not a hold of ours
        if not 0 < pid <= _PID_MAX:
            continue  # no process's
        try:
            text = _read_hold(dir_fd, name)
            if text is None:
                continue  # not a regular file
            if text is _FOREIGN:
                with suppress(OSError):
                    os.unlink(name, dir_fd=dir_fd)
                continue  # another user's: never trusted
            fields = text.split()
            need = int(fields[0]) if fields else 0
            start = int(fields[1]) if len(fields) > 1 else None
        except (OSError, ValueError):
            continue  # being written or removed right now
        if not _alive(pid, start):
            with suppress(OSError):
                os.unlink(name, dir_fd=dir_fd)
            continue
        held.append(need)
    return held


#: The largest pid Linux hands out (``pid_max`` is at most 2**22; this is
#: what ``os.kill`` accepts).
_PID_MAX = 2**31 - 1

#: What :func:`_read_hold` returns for a hold another user owns.
_FOREIGN = "\0foreign"


def _read_hold(dir_fd: int, name: str) -> str | None:
    """The text of the hold *name*, ``None`` if it is not a regular file, or
    :data:`_FOREIGN` if another user owns it; never read through a symlink,
    and never waited on (a FIFO)."""
    fd = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd
    )
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        if info.st_uid != os.getuid():
            return _FOREIGN
        return os.read(fd, 256).decode()
    finally:
        os.close(fd)


def _start_time(pid: int) -> int | None:
    """When the process *pid* started (field 22 of ``/proc/<pid>/stat``, in
    clock ticks since boot), or ``None`` if that cannot be read."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        # The command name (field 2) is in parentheses and may itself hold
        # spaces and parentheses: the fields after it follow the last ")".
        after = text[text.rindex(")") + 1 :].split()
        return int(after[22 - 3])
    except (OSError, ValueError, IndexError):
        return None


def _alive(pid: int, start: int | None = None) -> bool:
    """Whether the process *pid* is running -- and, given the *start* time
    its hold recorded, is still the process that recorded it."""
    if start is not None:
        now = _start_time(pid)
        if now is not None:
            return now == start
        # /proc unreadable (gone, or not mounted): fall back to the pid alone
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def _meminfo_total() -> int | None:
    """``MemTotal`` of ``/proc/meminfo``, in bytes."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_limit() -> int | None:
    """The memory limit of this process's cgroup (v2, then v1), if it has one."""
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            text = path.read_text().strip()
        except OSError:
            continue
        if text.isascii() and text.isdigit() and int(text) < (1 << 60):
            return int(text)
    return None
