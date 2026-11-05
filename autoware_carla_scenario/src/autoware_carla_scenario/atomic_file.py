"""Writing a file that others read while it is rewritten: in one rename.

``Path.write_text`` truncates before it writes, so a reader (the viewer, a
process sharing a cache) could find the file empty or half written.  The text
goes to a scratch file in the same directory instead, closed (so flushed) and
then renamed over the file: a reader sees the old file or the new one, whole.

Free of imports of the package, so that any module can use it
(:mod:`~autoware_carla_scenario.maps.cache`,
:mod:`~autoware_carla_scenario.batch_report`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["write_text_atomically"]


def write_text_atomically(path: Path, text: str) -> None:
    """Replace *path* with a file holding *text* (UTF-8), in one rename.

    The scratch name is ``mkstemp``'s, unique whichever thread, process or
    host writes; it starts with a dot and ends in ``.tmp``, and is removed
    when the write fails (an exception, or Ctrl-C, before the rename), so
    nothing that looks for the file finds one half written.  The file gets
    the mode ``0o644``, what ``write_text`` would make under the usual umask,
    rather than ``mkstemp``'s ``0o600``.
    """
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        Path(name).write_text(text, encoding="utf-8")
        # 0o644: Codon has no octal literal.
        os.chmod(name, 420)
        os.replace(name, str(path))
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
