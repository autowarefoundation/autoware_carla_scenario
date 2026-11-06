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
symlink) owned by this user that no one else can write to, it is used through
a descriptor opened once it is checked, and the lock and hold files are opened
without following symlinks.  A lock or hold is trusted only if it is a regular
file of this user's with one link (no hard link elsewhere): a hold that is not
is never counted (and removed, if another user owns it or the directory is
the gate's own), a lock that is not runs the build ungated.  A lock others
may open (an older gate's 0o644) is replaced by a fresh 0o600 one, so a
descriptor someone kept on the old one no longer contends for it; so is, in
the gate's own directories, one held for over :data:`LOCK_TIMEOUT_S` (through
a descriptor opened before it was made private).  In a configured directory,
a lock held that long -- by anyone -- runs the build ungated instead, and the
process's later builds then try for it only briefly until one takes it.

Only the gate's own directories (the default and the fallbacks) are ever
repaired -- made 0o700 and cleared of the gate's files other users own --
and only when they hold nothing but the gate's files; every use of one clears
it again (of fresh locks left by builds that died, too) and checks that
nothing in it is another user's and that its lock is a regular file of this
user's, and falls back otherwise; a hard-linked lock there is swapped for a
fresh one.  In any directory, a lock found replaced while it is checked (by
another build's swap) is let go, and the one in place tried instead.  In a
configured directory the gate changes
nothing but its own files (the lock and the holds).  When the directory is
not private, the gate warns and falls back to a per-user directory (under
``$XDG_RUNTIME_DIR``, then ``~/.cache``) and, failing that, admits the build
ungated -- as it does when its own files or the budget fail: the gate never
blocks or fails a check of its own accord.

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
    "LOCK_TIMEOUT_S",
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

#: How long a build tries for the gate's lock before it runs ungated (s).
#: The gate's own builds hold it only to look at the holds and add one, a few
#: milliseconds; one held longer is someone else's (another user may open an
#: older gate's world-readable lock and keep it), which must not stall builds.
LOCK_TIMEOUT_S = 120.0

#: How long a build waits between tries for the lock (s).
_LOCK_POLL_S = 0.05

#: How long a build tries for an older gate's lock before it replaces it (s):
#: the gate's own builds release it within milliseconds, so one not released
#: by then is held through a descriptor someone kept.
_REPLACE_WAIT_S = 1.0

#: How often a build waiting for the lock checks it was not replaced (s).
_NAMES_CHECK_S = 0.5

#: Set while the last try of this process for the lock timed out: later builds
#: then try for it only for :data:`_REPLACE_WAIT_S` (long enough for the
#: gate's own builds) before they run ungated; cleared once one takes it.
_LOCK_TIMED_OUT = threading.Event()

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
    otherwise -- without ever changing it, or anything in it but the gate's
    own lock and holds."""
    configured = os.environ.get(GATE_ENV, "")
    if configured == "0":
        return None
    if configured:
        return Path(configured)
    return _default_dir()


def _default_dir() -> Path | None:
    """The gate's own directory of holds, when none is configured: under the
    temporary directory, or, with none usable, the first per-user fallback
    (``None``, the gate off, without one either)."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    try:
        return Path(tempfile.gettempdir()) / f"acs-codon-gate-{uid}"
    except OSError as error:  # FileNotFoundError: no usable temporary directory
        fallbacks = _fallback_dirs()
        _warn(
            f"Codon memory gate: no temporary directory ({error}); "
            + (f"using {fallbacks[0]}" if fallbacks else "builds run ungated")
        )
        return fallbacks[0] if fallbacks else None


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
    where, dir_fd, own = gate
    try:
        name = f"{os.getpid()}-{threading.get_ident()}-{next(_COUNTER)}.hold"
        try:
            recorded = _wait_for_room(fcntl, dir_fd, own, name, need, budget, sleep)
        except (OSError, ValueError) as error:  # ValueError: a file's contents
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
    own: bool,
    name: str,
    need: int,
    budget: Callable[[], int],
    sleep: Callable[[float], None],
) -> bool:
    """Wait until *need* fits, then record it in the hold *name* of the
    directory *dir_fd* (*own*: one of the gate's own directories): ``True``
    once recorded, ``False`` if *budget* failed (warned).  Raises
    :class:`OSError` when the gate's files fail."""
    start = _start_time(os.getpid())
    record = str(need) if start is None else f"{need} {start}"
    waited = False
    while True:
        lock = _take_lock(fcntl, dir_fd, own)
        try:
            held = _held(dir_fd, own)
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


def _take_lock(fcntl: ModuleType, dir_fd: int, own: bool) -> int:
    """A descriptor of the directory *dir_fd*'s ``gate.lock``, exclusively
    locked, that still names it: created if missing, and replaced by a fresh
    one (:func:`_replace_lock`) if others may open it or -- in one of the
    gate's own directories (*own*) -- if it has another link or stays taken
    for :data:`LOCK_TIMEOUT_S`.  A file found replaced since it was opened
    (another build's swap: no links left, or no longer the one named) is let
    go and the one in place tried instead.  Raises :class:`OSError` when it
    is not a regular file of this user's with one link (in one of the gate's
    own directories: with any links), and :class:`TimeoutError` when it is
    not taken in time: :data:`LOCK_TIMEOUT_S` in all, or, once a build of
    this process has timed out on it (:data:`_LOCK_TIMED_OUT`), only
    :data:`_REPLACE_WAIT_S`, long enough for the gate's own builds."""
    flagged = _LOCK_TIMED_OUT.is_set()
    deadline = time.monotonic() + (_REPLACE_WAIT_S if flagged else LOCK_TIMEOUT_S)
    while True:
        fd = os.open(
            "gate.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
            dir_fd=dir_fd,
        )
        try:
            info = os.fstat(fd)
            fresh: int | None = None
            if info.st_nlink == 0 or not _names_lock(dir_fd, info):
                pass  # replaced since it was opened: try the one there now
            elif problem := _unsound(info, links=own):
                raise PermissionError(errno.EPERM, f"gate.lock {problem}")
            elif info.st_nlink != 1:
                # A hard link (to some other file of this user's): never
                # locked or changed, but swapped out by name.
                fresh = _replace_lock(fcntl, dir_fd, fd, info, wait=None)
            elif stat.S_IMODE(info.st_mode) & 0o077:
                # An older gate's 0o644: someone may hold a descriptor of it.
                fresh = _replace_lock(fcntl, dir_fd, fd, info, wait=_REPLACE_WAIT_S)
            elif _try_lock(fcntl, fd, deadline, dir_fd, info):
                if _names_lock(dir_fd, info):
                    _LOCK_TIMED_OUT.clear()
                    return fd
            elif not _names_lock(dir_fd, info):
                pass  # replaced while it was waited for: try the new one
            elif own and not flagged:
                # Held all this time: through a descriptor someone opened
                # before the lock was private (an earlier gate made it 0o600
                # in place).  Ours to replace, once.
                fresh = _replace_lock(fcntl, dir_fd, fd, info, wait=0.0)
                flagged = True  # never twice in one build
                if fresh is None:
                    deadline = time.monotonic() + _REPLACE_WAIT_S
            else:
                raise _timed_out(flagged)
            if fresh is not None:
                with suppress(OSError):
                    os.close(fd)  # releases the old lock, if it was taken
                _LOCK_TIMED_OUT.clear()
                return fresh
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            raise
        os.close(fd)  # replaced since it was opened: try the one there now
        if time.monotonic() >= deadline:
            raise _timed_out(flagged)


def _timed_out(flagged: bool) -> TimeoutError:
    """The error of a build that did not get the lock in time, remembered
    (:data:`_LOCK_TIMED_OUT`) for the process's later builds."""
    _LOCK_TIMED_OUT.set()
    if flagged:
        return TimeoutError(
            errno.ETIMEDOUT,
            "gate.lock still held elsewhere (an earlier build timed out)",
        )
    return TimeoutError(
        errno.ETIMEDOUT, f"gate.lock held elsewhere for over {LOCK_TIMEOUT_S:g} s"
    )


def _unsound(info: os.stat_result, *, links: bool = False) -> str:
    """Why the file *info* describes cannot be the gate's lock, or ``""``:
    it must be a regular file of this user's with no other link (a hard link
    could make it some other file of this user's) -- unless *links*, where
    one with more is to be replaced (:func:`_take_lock`)."""
    if not stat.S_ISREG(info.st_mode):
        return "is not a regular file"
    if info.st_uid != os.getuid():
        return f"is owned by uid {info.st_uid}, not {os.getuid()}"
    if info.st_nlink != 1 and not links:
        return f"has {info.st_nlink} links"
    return ""


def _names_lock(dir_fd: int, info: os.stat_result) -> bool:
    """Whether ``gate.lock`` in the directory *dir_fd* is still the file
    *info* describes (another build may have replaced it meanwhile)."""
    try:
        now = os.stat("gate.lock", dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (now.st_dev, now.st_ino) == (info.st_dev, info.st_ino)


def _replace_lock(
    fcntl: ModuleType,
    dir_fd: int,
    old: int,
    info: os.stat_result,
    *,
    wait: float | None,
) -> int | None:
    """Put a fresh 0o600 ``gate.lock``, locked, in place of *old* (whose
    file *info* describes) in the directory *dir_fd*, and return it; ``None``
    if another build has replaced *old* already.

    A descriptor someone opened on the old file then locks a file the gate no
    longer uses.  The old lock is tried for first, for *wait* (``None``: not
    at all, for a hard-linked file no build locks), so that a
    build of this gate inside it finishes before the swap, and one waiting
    for it finds the file replaced (:func:`_names_lock`) and tries the new
    one.  One held longer -- through a descriptor someone kept, which the
    swap is for -- is replaced while held.  The builds that waited out that
    holder together may then each put a fresh lock in place before seeing
    another's, and so decide at once, each for that one look at the holds:
    at worst those builds are admitted together whatever they need, once per
    machine (and so may a build of an older gate inside the old lock).
    A fresh lock that another build's has replaced by the time it is in place
    is given up (``None``), so a build only ever returns the lock in place.

    The fresh file is named ``gate.lock.<pid>-...`` until it is in place and
    records its process's start time, so one left behind by a build that was
    killed is removed (:func:`_clear`)."""
    if wait is not None:
        _try_lock(fcntl, old, time.monotonic() + wait)
    if not _names_lock(dir_fd, info):
        return None
    name = f"gate.lock.{os.getpid()}-{threading.get_ident()}-{next(_COUNTER)}"
    fd = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=dir_fd,
    )
    try:
        start = _start_time(os.getpid())
        if start is not None:
            os.write(fd, str(start).encode())
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # no one else can open it
        os.rename(name, "gate.lock", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        current = _names_lock(dir_fd, os.fstat(fd))
    except BaseException:
        os.close(fd)
        with suppress(OSError):
            os.unlink(name, dir_fd=dir_fd)
        raise
    if not current:
        os.close(fd)  # replaced in turn by another build's: try that one
        return None
    return fd


def _try_lock(
    fcntl: ModuleType,
    fd: int,
    deadline: float,
    dir_fd: int | None = None,
    info: os.stat_result | None = None,
) -> bool:
    """Take the exclusive lock on *fd*, trying until *deadline* (of
    :func:`time.monotonic`): whether it was taken.  Given the directory
    *dir_fd* and the lock's *info*, it also gives up, every
    :data:`_NAMES_CHECK_S`, once ``gate.lock`` names another file (it was
    replaced: waiting for this one is no use)."""
    check = time.monotonic() + _NAMES_CHECK_S
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            now = time.monotonic()
            if now >= deadline:
                return False
            if dir_fd is not None and info is not None and now >= check:
                if not _names_lock(dir_fd, info):
                    return False
                check = now + _NAMES_CHECK_S
            time.sleep(_LOCK_POLL_S)


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


def _private_dir(configured: Path) -> tuple[Path, int, bool] | None:
    """*configured*, if it is (or can be made) a private directory of this
    user's; else the first per-user fallback that is; else ``None``.  With
    the directory comes a descriptor of it, which every later use of the
    directory goes through, so it cannot be swapped once checked, and whether
    it is one of the gate's own."""
    own = configured == _default_dir()
    opened = _open_private(configured, own=own)
    if isinstance(opened, int):
        return configured, opened, own
    for candidate in _fallback_dirs():
        if candidate == configured:
            continue
        fallback = _open_private(candidate, own=True)
        if isinstance(fallback, int):
            _warn(
                f"Codon memory gate: not using {configured} ({opened}); "
                f"using {candidate} instead"
            )
            return candidate, fallback, True
    _warn(
        f"Codon memory gate: not using {configured} ({opened}) and no "
        "private per-user directory is available; builds run ungated"
    )
    return None


def _open_private(where: Path, *, own: bool) -> int | str:
    """A descriptor of *where*, or why it cannot be the gate's directory: it
    is created (mode 0o700) if missing, and must be a real directory (not a
    symlink) owned by this user that its group and others cannot write to.

    One of the gate's own directories (*own*; never a configured one) its
    group or others may use at all -- as an older gate left it, 0o755 or,
    under a lax umask, 0o775 -- is made 0o700, if it holds nothing but the
    gate's files.  Every time, the gate's files other users own are removed
    from one of its own directories, which is used only if nothing of another
    user's survives and its ``gate.lock``, if any, is sound (:func:`_unsound`):
    a repair that failed once is not trusted the next time."""
    try:
        where.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(where, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        return str(error)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            problem = f"owned by uid {info.st_uid}, not {os.getuid()}"
        elif own and info.st_mode & 0o077:
            mode = f"mode {stat.S_IMODE(info.st_mode):o}"
            if not all(_is_gate_file(name) for name in os.listdir(fd)):
                problem = f"open to others ({mode}) and holds other files"
            else:
                problem = _repair(fd) or _clear(fd)
        elif info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            problem = f"writable by others (mode {stat.S_IMODE(info.st_mode):o})"
        elif own:
            problem = _clear(fd)
        else:
            problem = ""
    except (OSError, ValueError) as error:
        problem = f"cannot be checked or made private ({error})"
    if not problem:
        return fd
    with suppress(OSError):
        os.close(fd)
    return problem


def _is_gate_file(name: str) -> bool:
    """Whether *name* is one the gate's directory holds (the lock, a lock
    being put in its place, a hold)."""
    return (
        name == "gate.lock" or name.startswith("gate.lock.") or name.endswith(".hold")
    )


def _repair(dir_fd: int) -> str:
    """Make the gate's directory *dir_fd* private: why that failed, or ``""``."""
    os.fchmod(dir_fd, 0o700)
    info = os.fstat(dir_fd)
    if info.st_mode & 0o077:
        return f"open to others (mode {stat.S_IMODE(info.st_mode):o})"
    return ""


def _clear(dir_fd: int) -> str:
    """Remove from the gate's own (private) directory *dir_fd* the gate's
    files other users own and the fresh locks (:func:`_replace_lock`) of
    builds that died: why it cannot be used then, or ``""``.

    Other builds may swap ``gate.lock`` meanwhile, so it is never removed for
    its links (a hard-linked one is swapped out under the lock instead,
    :func:`_take_lock`), and one seen with no links left -- replaced while it
    was looked at -- is not judged: the build checks the one in place when it
    takes it."""
    uid = os.getuid()
    for name in os.listdir(dir_fd):
        if not _is_gate_file(name):
            continue
        with suppress(OSError):  # FileNotFoundError: gone meanwhile
            entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if entry.st_nlink == 0:
                continue  # being replaced or removed right now
            if entry.st_uid != uid:
                os.unlink(name, dir_fd=dir_fd)  # others could write here once
            elif name.startswith("gate.lock.") and _stale_lock(dir_fd, name):
                os.unlink(name, dir_fd=dir_fd)
    for name in os.listdir(dir_fd):
        try:
            entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue  # a hold or fresh lock another build just removed
        if entry.st_nlink == 0:
            continue  # being replaced or removed right now
        if entry.st_uid != uid:
            return f"{name} of uid {entry.st_uid} cannot be removed"
        if name == "gate.lock":
            problem = _unsound(entry, links=True)
            if problem:
                return f"gate.lock {problem}"
    return ""


def _stale_lock(dir_fd: int, name: str) -> bool:
    """Whether the fresh lock *name* (``gate.lock.<pid>-...``) was left by a
    build that is gone (its pid dead, or since reused by another process)."""
    try:
        pid = int(name[len("gate.lock.") :].split("-", 1)[0])
    except ValueError:
        return False  # not one of the gate's
    if not 0 < pid <= _PID_MAX:
        return False
    text = _read_hold(dir_fd, name)
    if text is None or text is _FOREIGN:
        return False
    if text is _LINKED:
        return True
    fields = text.split()
    try:
        start = int(fields[0]) if fields else None
    except ValueError:
        start = None  # not what the gate writes: judge by the pid alone
    return not _alive(pid, start)


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


def _held(dir_fd: int, own: bool) -> list[int]:
    """What the running builds hold, in the directory *dir_fd*; the holds of
    dead processes are removed (and those of a pid since reused by another
    process); a file whose name does not start with a pid, or that is not a
    regular file, is ignored, and so is one with another link (a hard link to
    some other file), which is removed in one of the gate's own directories
    (*own*)."""
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
            if text is _FOREIGN or (text is _LINKED and own):
                with suppress(OSError):
                    os.unlink(name, dir_fd=dir_fd)
                continue  # another user's, or a hard link: never trusted
            if text is _LINKED:
                continue
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

#: What :func:`_read_hold` returns for a hold with more than one link.
_LINKED = "\0linked"


def _read_hold(dir_fd: int, name: str) -> str | None:
    """The text of the hold *name*, ``None`` if it is not a regular file or
    has just been removed (no links left), :data:`_FOREIGN` if another user
    owns it, or :data:`_LINKED` if it has another link; never read through a
    symlink, and never waited on (a FIFO)."""
    fd = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd
    )
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink == 0:
            return None
        if info.st_uid != os.getuid():
            return _FOREIGN
        if info.st_nlink != 1:
            return _LINKED
        return os.read(fd, 256).decode(errors="replace")
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
