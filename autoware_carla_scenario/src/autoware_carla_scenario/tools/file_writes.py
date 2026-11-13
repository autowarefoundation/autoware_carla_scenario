"""How ``detect-no-3d-model`` writes the map configuration it edits.

:func:`_write_atomically` replaces a file's content through a temporary file
beside it, falling back to :func:`_write_in_place` where the directory allows
no new file; a write in place that fails after the file was emptied raises
:class:`_PartialWriteError`, and :func:`_describe_os_error` is an error as
the tool reports it.  :mod:`.detect_no_3d_model_lanelets` re-exports them
under the names they had there (#2135): free of CARLA, YAML and argparse,
they are compiled by the library check (docs/typecheck.md).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


class _PartialWriteError(OSError):
    """Writing a file in place failed after it was opened and emptied."""


def _write_in_place(path: Path, data: bytes) -> None:
    """Overwrite *path* with *data* without replacing the file.

    The file keeps its mode and owner.  Opening it does not truncate, so
    an error from the open leaves it unchanged and propagates as is; an
    error after that (a full disk, an I/O error, a quota) may leave it
    empty or partial and is raised as :class:`_PartialWriteError`.
    """
    fd = os.open(path, os.O_WRONLY)
    try:
        try:
            os.ftruncate(fd, 0)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)
    except OSError as exc:
        raise _PartialWriteError(exc.errno, exc.strerror, str(path)) from exc


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace *path*'s content with *data* through a temporary file.

    *path* must already be resolved: replacing a symlink would turn it into
    a regular file instead of writing through it.  The temporary file gets
    *path*'s mode and, where permitted, its owner.  When the directory does
    not allow creating it, *path* is written in place instead.
    """
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except PermissionError:
        _write_in_place(path, data)
        return
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, tmp_path)
        stat = path.stat()
        try:
            os.chown(tmp_path, stat.st_uid, stat.st_gid)
        except OSError:
            pass  # Only root (or the owner, for the group) may do this.
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _describe_os_error(path: Path, exc: OSError) -> str:
    """*exc* as text, prefixed with *path* unless it already names it."""
    if exc.filename is not None and os.fspath(exc.filename) in (
        str(path),
        os.path.realpath(path),
    ):
        return str(exc)
    return f"{path}: {exc}"
