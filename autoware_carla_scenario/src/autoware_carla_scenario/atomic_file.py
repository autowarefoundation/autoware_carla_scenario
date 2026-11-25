"""Writing a file that others read while it is rewritten: in one rename.

``Path.write_text`` truncates before it writes, so a reader (the viewer, a
process sharing a cache) could find the file empty or half written, and a run
killed mid-write leaves it so for good: a truncated Protobuf message often
still parses, as a shorter one.  The data goes to a scratch file in the same
directory instead, closed (so flushed) and then renamed over the file: a
reader sees the old file or the new one, whole.

The file keeps what ``write_text`` would have kept: a symlink is written
through (the scratch file sits beside the file it names, which the rename
replaces), and an existing file keeps its permission bits.  A new file gets
the mode ``write_text`` would make, ``0o666`` less the umask: the scratch file
is created with ``0o666`` and the system takes the umask off, so the umask is
never read (and never set, as reading it takes in Python).

Free of imports of the package, so that any module can use it
(:mod:`~autoware_carla_scenario.maps.cache`,
:mod:`~autoware_carla_scenario.batch_report`,
:mod:`~autoware_carla_scenario.t4_scene.convert`,
:mod:`~autoware_carla_scenario.wosac.convert`,
:mod:`~autoware_carla_scenario.wosac.pack`).
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

# typing's, not collections.abc's: Codon has no collections.abc (the library
# check compiles this module, docs/typecheck.md).
from typing import Iterable  # noqa: UP035

__all__ = [
    "write_bytes_atomically",
    "write_chunks_atomically",
    "write_text_atomically",
]


def _scratch(path: str) -> str:
    """A new, empty scratch file beside *path* (resolved), unique whichever
    thread, process or host writes (``O_EXCL``, and a random name); its name
    starts with a dot and ends in ``.tmp``, so a glob for the file's own
    pattern does not find it.  It is created with the mode ``0o666``, less
    the umask, what ``write_text`` gives a new file, not ``mkstemp``'s
    ``0o600``."""
    name = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    # 0o666: Codon has no octal literal.
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 438)
    os.close(fd)
    return name


def _replace(name: str, path: str) -> None:
    """Give the scratch file *name* the permission bits of *path* (resolved),
    if it exists, and rename it over *path*."""
    try:
        shutil.copymode(path, name)
    except FileNotFoundError:
        pass  # A new file: it keeps the mode _scratch() made.
    os.replace(name, path)


def write_text_atomically(path: Path, text: str) -> None:
    """Replace *path* with a file holding *text* (UTF-8), in one rename.

    The scratch file is removed when the write fails (an exception, or
    Ctrl-C, before the rename), so nothing that looks for the file finds one
    half written.  A symlink is written through, and the file keeps its mode
    (a new one gets ``write_text``'s, ``0o666`` less the umask).
    """
    target = os.path.realpath(path)
    name = _scratch(target)
    try:
        Path(name).write_text(text, encoding="utf-8")
        _replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """Replace *path* with a file holding *data*, in one rename, as
    :func:`write_text_atomically` does text."""
    write_chunks_atomically(path, [data])


def write_chunks_atomically(path: Path, chunks: Iterable[bytes]) -> None:
    """Replace *path* with a file holding *chunks* one after the other, in one
    rename, as :func:`write_bytes_atomically` does one: for data too large to
    join first (a WOSAC submission's shard, written a scenario at a time)."""
    target = os.path.realpath(path)
    name = _scratch(target)
    try:
        with Path(name).open("wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
        _replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
