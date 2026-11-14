"""The files of the SLURM worker's start lock where a file system has no ``flock``.

The lock is then a file created with ``O_EXCL`` beside the start lock,
holding its holder's token; this module reads its identity
(:func:`_lock_identity`), creates it (:func:`_create_lock`), releases it
(:func:`_release_lock`) and removes it once judged stale
(:func:`_remove_stale`), judging its age by the file server's clock
(:class:`_ServerClock`), and takes over a stale one, one waiter at a time
(:func:`_take_over`, #2144).  Taking the lock is :mod:`.worker`'s
(``_exclusive_file()``), which re-exports these as its own.
"""

from __future__ import annotations

import contextlib
import errno
import os
import time
import uuid
from pathlib import Path

#: The errors of a ``flock`` the file system does not support (some NFS and
#: Lustre mounts): the start lock is then taken with an ``O_EXCL`` file.
_NO_FLOCK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK, errno.EINVAL}
)


#: A lock file's identity: its device, inode, mtime (ns) and token.  The
#: mtime tells a lock apart from a later one that reuses its inode: a lock
#: is only taken over once it is old, and a new one is not.
_LockId = tuple[int, int, int, str]


def _lock_sibling(path: Path, kind: str) -> Path:
    """A name beside *path* that no other process uses."""
    return path.with_name(f"{path.name}.{kind}.{uuid.uuid4().hex}")


def _server_now(path: Path) -> float:
    """The time by the clock of the file system holding *path*: the mtime of
    a file created beside it.

    A lock's mtime is set by that clock too (the NFS server's), so the age of
    a lock is measured with one clock, whatever the skew between this node's
    and the server's.
    """
    probe = _lock_sibling(path, "probe")
    # 0o644: Codon has no octal literal.
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 420)
    try:
        return os.fstat(fd).st_mtime
    finally:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(probe)


class _ServerClock:
    """The file server's clock, as seen from beside *path*.

    Its offset from this node's ``time.monotonic()`` (which no clock
    adjustment moves) is measured with a probe file (:func:`_server_now`)
    the first time it is read, and again whenever a file looks stale by it,
    so polling costs no probe file and nothing is judged stale by an
    estimate alone.
    """

    _path: Path
    _offset: float | None

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = None

    def now(self) -> float:
        """The server's time, measured now."""
        server = _server_now(self._path)
        self._offset = server - time.monotonic()
        return server

    def stale(self, mtime: float, after: float) -> bool:
        """Whether a file of *mtime* is more than *after* seconds old."""
        if self._offset is not None and (
            time.monotonic() + self._offset - mtime <= after
        ):
            return False
        return self.now() - mtime > after


def _lock_identity(path: Path) -> tuple[_LockId, float] | None:
    """The identity of the lock file at *path* and its mtime; ``None`` when
    there is none."""
    try:
        # Read as text decoded with "replace" and no newline translated
        # (newline=""): the str bytes.decode("utf-8", "replace") makes of
        # the whole file, which Codon, whose bytes is a str, cannot spell.
        with path.open(encoding="utf-8", errors="replace", newline="") as stream:
            token = stream.read()
            st = os.fstat(stream.fileno())
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_mtime_ns, token), st.st_mtime


def _create_lock(path: Path, token: str) -> bool:
    """Create *path* holding *token* with ``O_EXCL``; whether it was not
    there yet."""
    try:
        # 0o644: Codon has no octal literal.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 420)
    except FileExistsError:
        return False
    try:
        os.write(fd, token.encode("utf-8"))
    except BaseException:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(path)  # just made, and no one takes over a fresh lock
        raise
    os.close(fd)
    return True


def _release_lock(path: Path, token: str) -> None:
    """Remove *path* if it still holds *token*.

    Between the read and the unlink the lock can only change hands if it is
    taken over as stale in that moment, i.e. if it was held for longer than
    it may be.
    """
    held = _lock_identity(path)
    if held is not None and held[0][3] == token:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _remove_stale(path: Path, stale: _LockId) -> bool:
    """Unlink *path* if it is still the lock judged stale as *stale*; whether
    it was.

    Nothing is renamed or put back: a lock other than *stale* is left where
    it is.  Between the check and the unlink *path* can only change if the
    holder of *stale* is alive after all, or another waiter removes *stale*
    meanwhile, which :func:`_take_over` rules out but for a waiter
    that stalled past :data:`_TAKEOVER_STALE_AFTER`.
    """
    seen = _lock_identity(path)
    if seen is None or seen[0] != stale:
        return False
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    return True


#: How old (by the file server's clock) the takeover mutex of a lock file is
#: when the waiter holding it was killed: it is held for a few file
#: operations only.
_TAKEOVER_STALE_AFTER = 10.0


def _take_over(
    path: Path, stale: _LockId, clock: _ServerClock, stale_after: float, token: str
) -> bool:
    """Remove the lock at *path* judged stale as *stale*, if it still is it,
    one waiter at a time; whether this waiter removed it (and need not wait
    before trying the lock again).

    The waiters taking over one stale lock share a mutex keyed to its
    identity and made of generations: ``<path>.takeover.<ino>-<mtime>.<g>``,
    created with ``O_EXCL``.  A waiter creates generation 1, or generation
    g+1 once it has seen generation g and found it stale (its creator was
    killed: :data:`_TAKEOVER_STALE_AFTER` seconds by the server's clock).
    Creating never removes anything, so each generation has one winner, and
    a waiter that finds the newest generation live leaves the takeover to
    its creator.  The winner of generation g backs off, leaving its
    generation in place, if generation g+1 exists; otherwise it checks
    again that *path* is still *stale* and stale, unlinks it
    (:func:`_remove_stale`), and removes generations 1 to g.  Except in the
    cases below:

    - Of the waiters that judged lock L1 stale together, one removes it;
      the others find generation g live, or, once it is removed, win a
      generation of their own, find *path* free or holding a fresh lock L2
      (another identity), and leave it alone.  No waiter renames, puts back
      or removes a live lock.
    - A waiter killed while holding generation g leaves it and L1: after
      :data:`_TAKEOVER_STALE_AFTER` seconds one waiter wins g+1 and removes
      both; the waiters that arrive together find generation g stale at once
      and still only one of them wins g+1.  A missing generation below a
      dead one (its winner stalled, found itself superseded by a winner that
      was then killed) is won again, left in place, and climbed past once
      it is stale.  A mutex of a lock already removed only leads its winner
      to find the lock gone.

    The cases: a holder alive past its staleness.  The holder of L1 if it
    held it past *stale_after* (which the stagger's spacing + 60 s rules out
    but for a stalled node); or the winner of generation g if it stalled for
    :data:`_TAKEOVER_STALE_AFTER` seconds, so that g+1 was won as well (two
    winners of the mutex) -- right between its last check of *path* and its
    unlink while the winner of g+1 removed L1 and a task made L2, it removes
    L2.  Only then may two tasks hold the lock.
    """
    prefix = f"{path.name}.takeover.{stale[1]}-{stale[2]}."
    generation = 1
    while True:
        mutex = path.with_name(f"{prefix}{generation}")
        if _create_lock(mutex, token):
            break
        seen = _lock_identity(mutex)
        if seen is None:
            continue  # removed by a winner that found L1 gone: try it again
        if not clock.stale(seen[1], _TAKEOVER_STALE_AFTER):
            return False  # its winner is taking the lock over
        generation += 1
    if path.with_name(f"{prefix}{generation + 1}").exists():
        # Superseded (this generation was missing, or this waiter stalled):
        # generation g+1's winner removes 1 to g+1, or, if it was killed,
        # this generation goes stale and the next waiter climbs past both.
        # Removing it here would leave a gap below a dead g+1 that every
        # waiter would fill and back off from, for ever.
        return False
    seen = _lock_identity(path)
    removed = (
        seen is not None
        and seen[0] == stale
        and clock.stale(seen[1], stale_after)
        and _remove_stale(path, stale)
    )
    # The lock judged stale is gone: its mutex goes too.
    for old in range(generation, 0, -1):
        with contextlib.suppress(OSError):
            os.unlink(path.with_name(f"{prefix}{old}"))
    return removed
