"""How the viewer keeps its recordings and its copy of Rerun's web viewer on disk.

The name a recording is kept under in the viewer's cache directory when the
log's own directory cannot be written (:func:`cached_recording_name`, a hash
of the log's path), whether a recording is as new as its log
(:func:`_fresh`), how a recording just made is put in place, without
replacing one written there meanwhile unless asked (:func:`_publish`), and
how an unpacked copy of the web viewer replaces the one already in the cache
(:func:`_swap_into_place`).

Split out of ``ui.recordings`` (#2124), which re-exports every name moved
here and keeps ``_cached_recording()``, the path of that name under its own
``_cache_root()``, so that the library check compiles it
(``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

__all__ = ["cached_recording_name"]


def cached_recording_name(trajectory: Path) -> str:
    """The name of the recording of *trajectory* kept in the viewer's cache
    directory (under ``recordings/``) when the log's directory cannot be
    written: a hash of the log's resolved path."""
    digest = hashlib.sha256(str(trajectory.resolve()).encode()).hexdigest()[:24]
    return f"{digest}.rrd"


def _fresh(recording: Path, trajectory: Path) -> bool:
    try:
        return recording.stat().st_mtime >= trajectory.stat().st_mtime
    except OSError:
        return False


def _publish(tmp: Path, out: Path, replace: bool) -> None:
    """Rename *tmp* to *out*; without *replace*, only while there is no *out*
    (``FileExistsError`` otherwise), so a file written there meanwhile -- by a
    run converting its own log -- is kept."""
    if replace:
        os.replace(tmp, out)
        return
    try:
        os.link(tmp, out)  # fails, atomically, when out exists
    except FileExistsError:
        raise
    except OSError:  # a file system without hard links
        if os.path.lexists(out):
            raise FileExistsError(str(out)) from None
        os.replace(tmp, out)


def _swap_into_place(staging: Path, target: Path) -> None:
    """Move the directory *staging* to *target*, replacing a copy already
    there; the old copy is moved aside first, and back if the move fails."""
    old: Path | None = None
    if target.exists():
        old = Path(tempfile.mkdtemp(prefix=f".{target.name}.old.", dir=target.parent))
        try:
            os.replace(target, old)
        except OSError:
            old.rmdir()  # still the empty directory mkdtemp made
            raise
    try:
        os.replace(staging, target)
    except OSError:
        if old is not None:
            os.replace(old, target)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
