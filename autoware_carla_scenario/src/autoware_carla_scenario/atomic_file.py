"""Writing a file that others read while it is rewritten: in one rename.

``Path.write_text`` truncates before it writes, so a reader (the viewer, a
process sharing a cache) could find the file empty or half written, and a run
killed mid-write leaves it so for good: a truncated Protobuf message often
still parses, as a shorter one.  The data goes to a scratch file in the same
directory instead, closed (so flushed) and then renamed over the file: a
reader sees the old file or the new one, whole.

Free of imports of the package, so that any module can use it
(:mod:`~autoware_carla_scenario.maps.cache`,
:mod:`~autoware_carla_scenario.batch_report`,
:mod:`~autoware_carla_scenario.t4_scene.convert`,
:mod:`~autoware_carla_scenario.wosac.convert`,
:mod:`~autoware_carla_scenario.wosac.pack`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# typing's, not collections.abc's: Codon has no collections.abc (the library
# check compiles this module, docs/typecheck.md).
from typing import Iterable  # noqa: UP035

__all__ = [
    "write_bytes_atomically",
    "write_chunks_atomically",
    "write_text_atomically",
]


def _scratch(path: Path) -> str:
    """A new, empty scratch file beside *path*: ``mkstemp``'s, unique whichever
    thread, process or host writes; its name starts with a dot and ends in
    ``.tmp``, so a glob for the file's own pattern does not find it."""
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    return name


def _replace(name: str, path: Path) -> None:
    """Give the scratch file *name* ``write_text``'s mode and rename it over
    *path*."""
    # 0o644: Codon has no octal literal.
    os.chmod(name, 420)
    os.replace(name, str(path))


def write_text_atomically(path: Path, text: str) -> None:
    """Replace *path* with a file holding *text* (UTF-8), in one rename.

    The scratch file is removed when the write fails (an exception, or
    Ctrl-C, before the rename), so nothing that looks for the file finds one
    half written.  The file gets the mode ``0o644``, what ``write_text``
    would make under the usual umask, rather than ``mkstemp``'s ``0o600``.
    """
    name = _scratch(path)
    try:
        Path(name).write_text(text, encoding="utf-8")
        _replace(name, path)
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
    name = _scratch(path)
    try:
        with Path(name).open("wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
        _replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
