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
file of this user's with one link (no hard link elsewhere) that, for the lock,
no one else may open: a hold that is not is never counted (and removed if
another user owns it, or if it is a hard link in the gate's own directory).
A lock found replaced while it is checked is let go and the one in place
tried instead.

Only the gate's own directories (the default and the fallbacks) are ever
changed (a configured path that is one of them counts as one), and every
removal from one -- of what other users own, of a lock that is not trusted
-- is made under an exclusive lock on the directory itself, after checking
again what it removes, so no two builds change it at once.  The gate's builds
hold that lock for microseconds; no one else can open the directory once it
is 0o700, but a descriptor opened while it was open to others survives, so
one that cannot take it within :data:`_REPLACE_WAIT_S` is given up for the
rest of the process, which falls back to the next directory.  One its
group or others may use (an older gate's 0o755) is made 0o700, if it holds
nothing but the gate's files; on every use the gate's files other users own
are removed from it, and it is used only if nothing of another user's
remains and its lock, if any, is a regular file of this user's.  A lock there
that is not trusted (others may open an older gate's 0o644 one, or it is a
hard link) is removed -- after trying for it for :data:`_REPLACE_WAIT_S`, so
a build inside it finishes first -- and made afresh by the next build, so a
descriptor someone kept on the old one no longer contends; so is, once per
build, one held for over :data:`LOCK_TIMEOUT_S` (through a descriptor opened
before it was made private); a build that finds the lock replaced while it
waits gets a fresh :data:`LOCK_TIMEOUT_S` for the new one, at most
:data:`_DEADLINE_RESETS` times, so it waits at most about
``(1 + _DEADLINE_RESETS) * LOCK_TIMEOUT_S`` in all.  A configured directory
is never changed but
for the gate's lock and holds created in it and holds removed from it (of
builds that died, of other users): a lock there that is not trusted runs the
build ungated, as does one held for over :data:`LOCK_TIMEOUT_S`, after which
the process's later builds try for it only briefly until one takes it.  When
the directory is not private, the gate warns and falls back to a per-user
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

#: How long a build tries for an untrusted lock before it removes it (s): the
#: gate's own builds release it within milliseconds.
_REPLACE_WAIT_S = 1.0

#: How often a build waiting for the lock checks it was not replaced (s).
_NAMES_CHECK_S = 0.5

#: How many times a build's wait starts afresh because the lock was replaced.
_DEADLINE_RESETS = 2

#: How long a build pauses before it tries a lock it found replaced (s).
_RETRY_S = 0.002

#: The gate's own directories whose lock this process could not take in time
#: (:func:`_dir_locked`): given up on, for the next fallback.
_FORSAKEN: set[str] = set()
_FORSAKEN_LOCK = threading.Lock()

#: Set while the last try of this process for a configured directory's lock
#: timed out: later builds there then try for it only for
#: :data:`_REPLACE_WAIT_S` before they run ungated; cleared once one takes it.
#: (A lock in the gate's own directories held that long is removed instead.)
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
    own lock and holds.  A configured path that is the default directory or
    a fallback is treated as the gate's own (repaired and cleared)."""
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
    while True:  # once more for each directory given up on (_DirectoryHeld)
        gate = _private_dir(configured)
        if gate is None:
            yield  # no private directory to gate in: run ungated (warned)
            return
        where, dir_fd, own = gate
        try:
            name = f"{os.getpid()}-{threading.get_ident()}-{next(_COUNTER)}.hold"
            try:
                recorded = _wait_for_room(fcntl, dir_fd, own, name, need, budget, sleep)
            except _DirectoryHeld as error:
                _forsake(where, error)
                continue
            except (OSError, ValueError) as error:  # ValueError: a file's contents
                _warn(f"Codon memory gate: {where}: {error}; the build runs ungated")
                recorded = False
            if not recorded:
                yield  # the gate failed (warned): run ungated
                return
            try:
                yield
            finally:
                # Never masks what the block raised: a failure here is logged.
                try:
                    os.unlink(name, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    _warn(f"Codon memory gate: cannot remove {where / name} ({error})")
            return
        finally:
            with suppress(OSError):
                os.close(dir_fd)


class _DirectoryHeld(TimeoutError):
    """The lock of one of the gate's own directories was not taken in time."""


def _forsake(where: Path, error: OSError) -> None:
    """Give up on the gate's own directory *where* for the rest of the
    process (its lock is held elsewhere: *error*)."""
    with _FORSAKEN_LOCK:
        _FORSAKEN.add(str(where))
    _warn(f"Codon memory gate: not using {where} any more ({error})")


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
    locked, that still names it (created if missing; one found replaced
    since it was opened is let go, and the one in place tried instead).

    In one of the gate's own directories (*own*), a lock that is not trusted
    (:func:`_untrusted`), or -- once -- one held for :data:`LOCK_TIMEOUT_S`,
    is removed (:func:`_drop_lock`) and the fresh one taken.  In a configured
    directory, the first raises :class:`PermissionError`, and the second
    :class:`TimeoutError`, as does one still held after
    :data:`_REPLACE_WAIT_S` once a build of this process has timed out there
    (:data:`_LOCK_TIMED_OUT`).

    A lock found replaced (by another build, or removed here) starts the wait
    afresh, for the new lock, at most :data:`_DEADLINE_RESETS` times: the
    whole wait is bounded by ``(1 + _DEADLINE_RESETS)`` times the window
    (:data:`LOCK_TIMEOUT_S`, or the shorter one above), plus a second for
    each untrusted lock let finish.  Raises :class:`_DirectoryHeld` when the
    directory's own lock cannot be taken (:func:`_dir_locked`)."""
    flagged = not own and _LOCK_TIMED_OUT.is_set()
    window = _REPLACE_WAIT_S if flagged else LOCK_TIMEOUT_S
    deadline = time.monotonic() + window
    resets = 0
    dropped = False
    while True:
        fd = os.open(
            "gate.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
            dir_fd=dir_fd,
        )
        replaced = True  # unless it is taken, or found held too long
        try:
            info = os.fstat(fd)
            if info.st_nlink == 0 or not _names_lock(dir_fd, info):
                pass  # replaced since it was opened: try the one there now
            elif problem := _untrusted(info):
                if not own:
                    raise PermissionError(errno.EPERM, f"gate.lock {problem}")
                # An older gate's 0o644 (a build of this gate may be inside
                # it: let it finish), or a hard link (no build locks it).
                lax = info.st_nlink == 1 and stat.S_ISREG(info.st_mode)
                _drop_lock(fcntl, dir_fd, fd, info, _REPLACE_WAIT_S if lax else None)
            elif _try_lock(fcntl, fd, deadline, dir_fd, info):
                if _names_lock(dir_fd, info):
                    if not own:
                        _LOCK_TIMED_OUT.clear()
                    return fd
            elif not _names_lock(dir_fd, info):
                pass  # replaced while it was waited for: try the new one
            elif own and not dropped:
                # Held all this time: through a descriptor someone opened
                # before an earlier gate made it 0o600 in place.
                _drop_lock(fcntl, dir_fd, fd, info, None)
                dropped = True
            else:
                raise _timed_out(own, flagged)
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            raise
        os.close(fd)
        if replaced and resets < _DEADLINE_RESETS:
            # A new lock: wait for it afresh (bounded: see the docstring).
            resets += 1
            deadline = max(deadline, time.monotonic() + window)
        if time.monotonic() >= deadline:
            raise _timed_out(own, flagged)
        time.sleep(_RETRY_S)


def _timed_out(own: bool, flagged: bool) -> TimeoutError:
    """The error of a build that did not get the lock in time; in a
    configured directory (not *own*), remembered (:data:`_LOCK_TIMED_OUT`)
    for the process's later builds."""
    if not own:
        _LOCK_TIMED_OUT.set()
    if flagged:
        return TimeoutError(
            errno.ETIMEDOUT,
            "gate.lock still held elsewhere (an earlier build timed out)",
        )
    return TimeoutError(
        errno.ETIMEDOUT, f"gate.lock held elsewhere for over {LOCK_TIMEOUT_S:g} s"
    )


def _untrusted(info: os.stat_result) -> str:
    """Why the file *info* describes cannot be the gate's lock, or ``""``:
    it must be a regular file of this user's with no other link (a hard link
    could make it some other file of this user's) that no one else may open
    (an older gate's was 0o644, and a descriptor of it may have been kept)."""
    if not stat.S_ISREG(info.st_mode):
        return "is not a regular file"
    if info.st_uid != os.getuid():
        return f"is owned by uid {info.st_uid}, not {os.getuid()}"
    if info.st_nlink != 1:
        return f"has {info.st_nlink} links"
    if stat.S_IMODE(info.st_mode) & 0o077:
        return f"may be opened by others (mode {stat.S_IMODE(info.st_mode):o})"
    return ""


def _names_lock(dir_fd: int, info: os.stat_result) -> bool:
    """Whether ``gate.lock`` in the directory *dir_fd* is still the file
    *info* describes (another build may have replaced it meanwhile)."""
    try:
        now = os.stat("gate.lock", dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (now.st_dev, now.st_ino) == (info.st_dev, info.st_ino)


@contextmanager
def _dir_locked(fcntl: ModuleType, dir_fd: int) -> Iterator[None]:
    """Hold the exclusive lock on the gate's own directory *dir_fd*, which
    every removal from it is made under (for microseconds); raises
    :class:`_DirectoryHeld` if it is not taken within :data:`_REPLACE_WAIT_S`
    -- held through a descriptor of the directory someone opened while it was
    open to others, so the directory is given up on (:func:`_forsake`)."""
    try:
        taken = _try_lock(fcntl, dir_fd, time.monotonic() + _REPLACE_WAIT_S)
    except OSError as error:
        # A directory whose lock cannot be taken at all (NFS refuses an
        # exclusive flock on a read-only descriptor: EBADF) is given up on
        # like one held elsewhere, so the build falls back, gated.
        raise _DirectoryHeld(
            error.errno or errno.EIO, f"its lock cannot be taken ({error})"
        ) from error
    if not taken:
        raise _DirectoryHeld(
            errno.ETIMEDOUT,
            f"its lock held elsewhere for over {_REPLACE_WAIT_S:g} s",
        )
    try:
        yield
    finally:
        fcntl.flock(dir_fd, fcntl.LOCK_UN)


def _drop_lock(
    fcntl: ModuleType, dir_fd: int, fd: int, info: os.stat_result, wait: float | None
) -> None:
    """Remove ``gate.lock`` -- the file *fd*, which *info* describes -- from
    the gate's own directory *dir_fd*, so the next build makes it afresh; but
    only if it still names that file, checked under the directory's lock,
    which every removal from it is made under.  It is tried for first, for
    *wait* (``None``: not at all), so that a build inside it finishes first;
    a build that waited for it then finds it replaced and takes the new one.
    A build inside it for longer (a gate stalled for over *wait*, or an
    older gate's) may decide while the next build in the fresh one does."""
    if wait is not None:
        _try_lock(fcntl, fd, time.monotonic() + wait)  # outside: keep it short
    with _dir_locked(fcntl, dir_fd):
        if _names_lock(dir_fd, info):
            os.unlink("gate.lock", dir_fd=dir_fd)


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
    it is one of the gate's own (*configured* is when it is the default or a
    fallback path, however it was given)."""
    own = configured == _default_dir() or configured in _fallback_dirs()
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
    gate's files (setting the mode is the same whoever does it, so it needs
    no lock).  Every time, the gate's files other users own are removed from
    one of its own directories (:func:`_clear`), which is used only if
    nothing of another user's survives and its ``gate.lock``, if any, is a
    regular file of this user's: a repair that failed once is not trusted the
    next time."""
    with _FORSAKEN_LOCK:
        if str(where) in _FORSAKEN:
            return "its lock was held elsewhere earlier in this process"
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
    except _DirectoryHeld as error:
        _forsake(where, error)
        problem = str(error)
    except (OSError, ValueError) as error:
        problem = f"cannot be checked or made private ({error})"
    if not problem:
        return fd
    with suppress(OSError):
        os.close(fd)
    return problem


def _is_gate_file(name: str) -> bool:
    """Whether *name* is one the gate's directory holds (the lock, a hold)."""
    return name == "gate.lock" or name.endswith(".hold")


def _repair(dir_fd: int) -> str:
    """Make the gate's directory *dir_fd* private: why that failed, or ``""``."""
    os.fchmod(dir_fd, 0o700)
    info = os.fstat(dir_fd)
    if info.st_mode & 0o077:
        return f"open to others (mode {stat.S_IMODE(info.st_mode):o})"
    return ""


def _clear(dir_fd: int) -> str:
    """Remove the gate's files other users own from the gate's own (private)
    directory *dir_fd* -- under the directory's lock, checking each again
    there -- and say why it cannot be used then, or ``""``.  An entry gone,
    or seen with no links left (being replaced or removed right now), is
    skipped; ``gate.lock`` itself is judged when it is taken
    (:func:`_take_lock`)."""
    import fcntl  # noqa: PLC0415 - Linux's; the gate is off without it

    if _foreign_entries(dir_fd):
        with _dir_locked(fcntl, dir_fd):
            for name in _foreign_entries(dir_fd):
                with suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=dir_fd)  # others could write here once
    uid = os.getuid()
    for name in os.listdir(dir_fd):
        entry = _lstat(dir_fd, name)
        if entry is None:
            continue
        if entry.st_uid != uid:
            return f"{name} of uid {entry.st_uid} cannot be removed"
        if name == "gate.lock" and not stat.S_ISREG(entry.st_mode):
            return "gate.lock is not a regular file"
    return ""


def _foreign_entries(dir_fd: int) -> list[str]:
    """The gate's files in the directory *dir_fd* that other users own."""
    uid = os.getuid()
    names = []
    for name in os.listdir(dir_fd):
        if _is_gate_file(name):
            entry = _lstat(dir_fd, name)
            if entry is not None and entry.st_uid != uid:
                names.append(name)
    return names


def _lstat(dir_fd: int, name: str) -> os.stat_result | None:
    """*name* in the directory *dir_fd*, not followed if a symlink; ``None``
    if it is gone, or has no links left (being replaced or removed)."""
    try:
        entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return entry if entry.st_nlink else None


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
