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
symlink) owned by this user that no one else can write to, and the lock and
hold files are opened without following symlinks.  When it is not, the gate
warns and falls back to a per-user directory (under ``$XDG_RUNTIME_DIR``, then
``~/.cache``) and, failing that, admits the build ungated: the gate never
blocks or fails a check of its own accord.

The gate is Linux's (``fcntl``, ``/proc/meminfo``), as Codon is; elsewhere,
or with ``AUTOWARE_CARLA_SCENARIO_CODON_GATE=0``, every build is admitted at
once.
"""

from __future__ import annotations

import itertools
import logging
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

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
    where = _private_dir(configured)
    if where is None:
        yield  # no private directory to gate in: run ungated (warned)
        return
    hold = where / f"{os.getpid()}-{threading.get_ident()}-{next(_COUNTER)}.hold"
    start = _start_time(os.getpid())
    record = str(need) if start is None else f"{need} {start}"
    waited = False
    recorded = False
    while True:
        try:
            lock = os.open(
                where / "gate.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        except OSError as error:
            _warn(
                f"Codon memory gate: cannot open {where / 'gate.lock'} ({error}); the build runs ungated"
            )
            yield
            return
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            held = _held(where)
            room = budget()
            if not held or room <= 0 or sum(held) + need <= room:
                recorded = _write_hold(hold, record)
                break
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
    if not recorded:
        yield  # the hold could not be created (warned): run ungated
        return
    try:
        yield
    finally:
        hold.unlink(missing_ok=True)


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


def _private_dir(configured: Path) -> Path | None:
    """*configured*, if it is (or can be made) a private directory of this
    user's; else the first per-user fallback that is; else ``None``."""
    problem = _private_problem(configured)
    if problem is None:
        return configured
    for candidate in _fallback_dirs():
        if candidate == configured:
            continue
        if _private_problem(candidate) is None:
            _warn(
                f"Codon memory gate: not using {configured} ({problem}); "
                f"using {candidate} instead"
            )
            return candidate
    _warn(
        f"Codon memory gate: not using {configured} ({problem}) and no "
        "private per-user directory is available; builds run ungated"
    )
    return None


def _private_problem(where: Path) -> str | None:
    """Why *where* cannot be the gate's directory, or ``None`` if it can: it
    is created (mode 0o700) if missing, and must be a real directory owned by
    this user that neither its group nor others may write to."""
    try:
        where.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = os.lstat(where)
    except OSError as error:
        return str(error)
    if not stat.S_ISDIR(info.st_mode):
        return "not a directory"
    if info.st_uid != os.getuid():
        return f"owned by uid {info.st_uid}, not {os.getuid()}"
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"writable by others (mode {stat.S_IMODE(info.st_mode):o})"
    return None


def _write_hold(hold: Path, record: str) -> bool:
    """Create *hold* (never through a symlink, never over a file) with
    *record*; whether that worked."""
    try:
        fd = os.open(
            hold,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except OSError as error:
        _warn(
            f"Codon memory gate: cannot create {hold} ({error}); the build runs ungated"
        )
        return False
    try:
        os.write(fd, record.encode())
    finally:
        os.close(fd)
    return True


_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()


def _warn(message: str) -> None:
    """Log *message* as a warning, once per process."""
    with _WARNED_LOCK:
        if message in _WARNED:
            return
        _WARNED.add(message)
    logger.warning(message)


def _held(where: Path) -> list[int]:
    """What the running builds hold; the holds of dead processes are removed
    (and those of a pid since reused by another process); a file whose name
    does not start with a pid is ignored."""
    held: list[int] = []
    for path in where.glob("*.hold"):
        try:
            pid = int(path.name.split("-", 1)[0])
        except ValueError:
            continue  # not a hold of ours
        if pid <= 0:
            continue
        try:
            fields = _read_nofollow(path).split()
            need = int(fields[0]) if fields else 0
            start = int(fields[1]) if len(fields) > 1 else None
        except (OSError, ValueError):
            continue  # being written or removed right now
        if not _alive(pid, start):
            path.unlink(missing_ok=True)
            continue
        held.append(need)
    return held


def _read_nofollow(path: Path) -> str:
    """The text of *path*, which is not read through a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
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
    except ProcessLookupError:
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
