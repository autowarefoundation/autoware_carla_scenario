"""Where ``scenario-setup`` installs CARLA, and the paths it swaps an install by.

The path logic of :mod:`autoware_carla_scenario.carla_install`, which
re-exports every name it uses: the install directory and its launcher (which
:class:`~autoware_carla_scenario.server.CarlaServerManager` looks for), a
package given as a local path, the lock file and the room an install needs,
the renames that put a complete download in place of an old install, and
whether an install is current and the conditional request that asks the server
so. It needs nothing Codon lacks, so the library check compiles it from its
source (``docs/typecheck.md``); the downloading, the unpacking and the reading
of the stamp's JSON stay in ``carla_install``.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional

HOME_ENV = "AUTOWARE_CARLA_SCENARIO_HOME"
"""Overrides the framework's home directory, ``~/.autoware_carla_scenario``."""

_LAUNCHER = "Linux/CarlaUnreal.sh"
"""The server's launcher, relative to the unpacked package (``carla_install``'s
``LAUNCHER``, as a ``str``: a ``Path`` joined with either is the same path)."""

_STAMP = ".download.json"
"""What the install was downloaded from, to tell whether it is still current."""

_UNPACKED_PER_PACKED = 2.0
"""The nightly unpacks to about twice its archive."""

_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*://|file:)", re.IGNORECASE)
"""The ``scheme://`` (or ``file:``, which urllib takes without ``//``) that tells
a URL from a local path."""


def install_dir() -> Path:
    """Where ``scenario-setup`` puts CARLA: ``<home>/bin/carla``."""
    home = os.environ.get(HOME_ENV, "").strip() or "~/.autoware_carla_scenario"
    return Path(home).expanduser() / "bin" / "carla"


def installed_executable() -> Optional[Path]:
    """The launcher of the CARLA ``scenario-setup`` installed, if there is one."""
    launcher = install_dir() / _LAUNCHER
    return launcher if launcher.is_file() else None


def package_url(url: str) -> str:
    """*url* itself, or the ``file://`` URL of a local path given instead.

    A value without a ``scheme://`` (or ``file:``) is a path: ``~`` expands
    and a relative path is taken from the current directory.
    """
    if _SCHEME.match(url):
        return url
    return Path(url).expanduser().resolve().as_uri()


def _staging(target: Path) -> Path:
    """Where the download into *target* is unpacked until it is complete."""
    return target.with_name(target.name + ".partial")


def _replaced(target: Path) -> Path:
    """Where the install *target* held is moved while it is being deleted."""
    return target.with_name(target.name + ".replaced")


def _lock_path(target: Path) -> Path:
    """The file whose ``flock`` lets one run at a time install into *target*."""
    return target.with_name(target.name + ".lock")


def _room_needed(packed: int, per_packed: float = _UNPACKED_PER_PACKED) -> int:
    """Bytes a package of *packed* bytes needs unpacked, at *per_packed* to one.

    ``carla_install`` passes its own ``_UNPACKED_PER_PACKED``, which a test
    replaces, rather than letting this module read its constant.
    """
    return int(packed * per_packed)


def _swap_into_place(staging: Path, target: Path) -> None:
    """Replace the install at *target* with the complete one at *staging*.

    Deleting ~30 GB takes seconds, long enough for a Ctrl-C, and a tree deleted
    in place could be left with its launcher and stamp but not much else, which
    the next run would take for a current install. So the old tree is renamed
    aside first and deleted only once *staging* is in its place: *target* is
    always either a complete install or missing, never half of one. What an
    interrupt leaves aside, ``carla_install.install()`` deletes on its next run.
    """
    old = _replaced(target)
    shutil.rmtree(old, ignore_errors=True)
    if not target.exists():
        staging.rename(target)
        return
    target.rename(old)
    try:
        staging.rename(target)
    except OSError:
        old.rename(target)  # the previous install, as it was
        raise
    # Not an install any more, whatever is left of it if this is cut short.
    (old / _STAMP).unlink(missing_ok=True)
    shutil.rmtree(old, ignore_errors=True)


def _content_range(
    content_range: Optional[str],
) -> tuple[Optional[int], Optional[int]]:
    """First byte and size of a ``Content-Range: bytes <first>-<last>/<size>``."""
    unit, _, span = (content_range or "").strip().partition(" ")
    first, _, rest = span.partition("-")
    _, _, total = rest.partition("/")
    if unit != "bytes" or not (first.isascii() and first.isdigit()):
        return None, None
    if not (total.isascii() and total.isdigit()):
        return None, None
    # Named Optionals: Codon types a tuple display of two ints as tuple[int, int].
    start: Optional[int] = int(first)
    size: Optional[int] = int(total)
    return start, size


def _stamped(stamp: dict[str, Optional[str]], key: str) -> Optional[str]:
    """``stamp.get(key)``: Codon's ``dict.get`` takes no default of ``None``."""
    return stamp[key] if key in stamp else None


def _conditions_of(stamp: dict[str, Optional[str]], url: str) -> dict[str, str]:
    """Headers that make the server answer 304 for the download *stamp* describes.

    None unless *stamp* is of *url*: ``If-None-Match`` its ETag and
    ``If-Modified-Since`` its Last-Modified, each where it has one.
    """
    if _stamped(stamp, "url") != url:
        return {}
    headers: dict[str, str] = {}
    etag = _stamped(stamp, "etag")
    if etag is not None and etag:
        headers["If-None-Match"] = etag
    last_modified = _stamped(stamp, "last_modified")
    if last_modified is not None and last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _current_of(
    stamp: dict[str, Optional[str]], source: dict[str, Optional[str]]
) -> bool:
    """Whether *stamp* is of the same download as *source*.

    The same URL and ETag, or, where the server gives no ETag, the same URL,
    Last-Modified and size, all given.
    """
    if _stamped(stamp, "url") != source["url"]:
        return False
    etag = source["etag"]
    if etag is not None and etag:
        return _stamped(stamp, "etag") == etag
    for key in ("last_modified", "size"):
        value = source[key]
        if not (value is not None and value):
            return False
    for key in ("last_modified", "size"):
        if _stamped(stamp, key) != source[key]:
            return False
    return True
