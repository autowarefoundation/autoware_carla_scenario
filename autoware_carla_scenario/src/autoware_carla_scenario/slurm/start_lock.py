"""The files of the SLURM worker's start lock where a file system has no ``flock``.

The lock is then a file created with ``O_EXCL`` beside the start lock,
holding its holder's token; this module reads its identity
(:func:`_lock_identity`), creates it (:func:`_create_lock`), releases it
(:func:`_release_lock`) and removes it once judged stale
(:func:`_remove_stale`), judging its age by the file server's clock
(:class:`_ServerClock`).  Taking the lock, and over a stale one, is
:mod:`.worker`'s (``_exclusive_file()``, ``_take_over()``), which re-exports
these as its own.
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
    meanwhile, which :func:`.worker._take_over` rules out but for a waiter
    that stalled past :data:`.worker._TAKEOVER_STALE_AFTER`.
    """
    seen = _lock_identity(path)
    if seen is None or seen[0] != stale:
        return False
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    return True
