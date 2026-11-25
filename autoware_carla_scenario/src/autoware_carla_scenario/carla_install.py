"""``scenario-setup``: download a CARLA server for the framework to launch.

By default it fetches the CARLA release the framework targets, CARLA 1.0
(Unreal Engine 5.8), and unpacks it into ``~/.autoware_carla_scenario/bin/carla``,
where :class:`~autoware_carla_scenario.server.CarlaServerManager` finds its
launcher when ``CARLA_EXECUTABLE`` is not set::

    scenario-setup            # download, or keep an install that is current
    scenario-setup --force    # download again regardless
    scenario-setup --nightly  # CARLA's nightly build (the ue58-dev branch) instead
    scenario-setup --nightly 20261124           # the last nightly built that day
    scenario-setup --nightly 23d487e3           # the nightly built from that commit
    scenario-setup --nightly 20261124_23d487e3  # a nightly by its own name

The client speaks the 1.0 release's RPC by default (``TYPESAFE_CARLA_CARLA_REF``,
see :mod:`autoware_carla_scenario.carla_release`); a nightly server wants
``TYPESAFE_CARLA_CARLA_REF=ue58-dev``.

The archive (~14 GB, ~30 GB unpacked) is unpacked while it downloads, so it is
never kept on disk; the new tree replaces the old one only once it is complete,
by renames, so an interrupted run never leaves a half-deleted install in place.
A connection that drops mid-download is picked up where it stopped with a range
request, so a flaky link costs a retry rather than the whole transfer.

One run at a time installs into a directory: each holds an exclusive ``flock``
on ``<target>.lock`` beside it from the clean-up of what an interrupted run left
to the swap, so a second ``scenario-setup`` for the same directory waits for the
first (and then usually finds its install up to date) instead of deleting the
first one's download. The lock file stays: deleting it would let a run lock the
deleted file while another locks its replacement. A run that cannot open the
lock file for writing (an install of another user's, or on a read-only file
system) can only check the install: it takes a shared lock, which still waits
for a writer, and reports an install it would have to replace rather than
trying to.
``AUTOWARE_CARLA_SCENARIO_HOME`` moves ``~/.autoware_carla_scenario``.

CARLA publishes its server for Linux x86_64 only, so on any other machine
(aarch64 included, where the framework and its client do run) the release and
the nightly are refused rather than downloaded: run the server on an x86_64 host and point
``server.host`` at it. ``--url`` still installs a package of one's choosing,
from a URL or from a local path (``--url ~/Downloads/CARLA.tar.gz``).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import errno
import functools
import gzip
import http.client
import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import time
import zlib
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import IO, Callable, Iterator, Optional, cast

# The path logic, which the library check compiles from its own module; every
# name of it is re-exported here, where it has always been imported from.
from .carla_install_paths import (
    _LAUNCHER,
    _STAMP,
    _conditions_of,
    _content_range,
    _current_of,
    _lock_path,
    _open_lock,
    _replaced,
    _room_needed,
    _staging,
    _swap_into_place,
)
from .carla_install_paths import _cannot_install as _cannot_install
from .carla_install_paths import _UNPACKED_PER_PACKED as _UNPACKED_PER_PACKED
from .carla_install_paths import _UNWRITABLE_ERRNOS as _UNWRITABLE_ERRNOS
from .carla_install_paths import HOME_ENV as HOME_ENV
from .carla_install_paths import install_dir as install_dir
from .carla_install_paths import installed_executable as installed_executable
from .carla_install_paths import launcher_in as launcher_in
from .carla_install_paths import package_url as package_url

# The errors of a ``flock`` the file system does not support (some NFS and
# Lustre mounts) are the checked slurm.start_lock's, whose start lock tells
# such a file system by them too.
from .slurm.start_lock import _NO_FLOCK_ERRNOS

logger = logging.getLogger(__name__)

RELEASE_URL = "https://downloads.carlasim.com/Linux/CARLA_1.0.tar.gz"
"""The CARLA release the framework targets (1.0), as linked from its release page."""

NIGHTLY_URL = "https://downloads.carlasim.com/Linux/Dev/CARLA_UE5_Latest.tar.gz"
"""CARLA's nightly Linux build (the ``ue58-dev`` branch), as linked from its
download page."""

NIGHTLY_DIR_URL = "https://downloads.carlasim.com/Linux/Dev/"
"""Where CARLA's CI keeps every nightly it built, ``<YYYYMMDD>_<commit>.tar.gz``:
the committer date and the abbreviated hash of the ``ue58-dev`` commit it built
(``Util/Tools/Deploy.sh`` in CARLA), beside the ``CARLA_UE5_Latest.tar.gz`` it
also replaces."""

NIGHTLY_COMMITS_API = "https://api.github.com/repos/carla-simulator/carla/commits"
"""The GitHub API a nightly given by date or commit is looked up in."""

NIGHTLY_BRANCH = "ue58-dev"
"""The branch CARLA's nightly is built from."""

USER_AGENT = "autoware-carla-scenario (scenario-setup)"
"""The ``User-Agent`` every download request sends.

``downloads.carlasim.com``, where CARLA's packages are served from, answers
urllib's own (``Python-urllib/3.x``) with 403 Forbidden whatever the file, so
a download that sent it could never start."""

PACKAGE_MACHINES = frozenset({"x86_64"})
"""The machines (``platform.machine()``) CARLA's own packages, the release and
the nightly, run on."""

NIGHTLY_MACHINES = PACKAGE_MACHINES
"""The name :data:`PACKAGE_MACHINES` had when only the nightly was offered."""

LAUNCHER = PurePosixPath(_LAUNCHER)
"""The server's launcher in a CARLA 0.10 package, relative to the unpacked
package. CARLA 1.0's is at the package's top (``CarlaUnreal.sh``): either is
accepted (:func:`launcher_in`)."""


def _launcher_of(target: Path) -> Path:
    """The launcher of the install at *target*, or where a 0.10 package's would be."""
    found = launcher_in(target)
    return found if found is not None else target / LAUNCHER


_TIMEOUT = 60.0
"""Seconds a read may stall before the connection counts as dropped."""

_RETRIES = 8
"""Reconnections in a row that may fail to deliver a byte before giving up."""

_RETRY_DELAY = 2.0
"""Seconds before the first reconnection; each failed one doubles it, up to 60."""


def install(
    url: str = RELEASE_URL, target: Optional[Path] = None, force: bool = False
) -> Path:
    """Download the CARLA package at *url* into *target* and return its launcher.

    *url* may also be a local path (see :func:`package_url`); it is read in place.
    An install already made from the same download (same ETag, or Last-Modified
    and size) is kept unless *force*. The package's single top-level directory
    is dropped, so the launcher is ``<target>/CarlaUnreal.sh`` (CARLA 1.0) or
    ``<target>/Linux/CarlaUnreal.sh`` (0.10).
    Another install into *target* running meanwhile is waited for (see
    :func:`_locked`), so the two never delete each other's download.

    Raises:
        RuntimeError: There is not room for the package, it has no launcher, or
            *target* holds something other than a CARLA this function installed.
            A previous install is left as it was, and so is anything else.
            Also raised, before any download, on a Python whose ``tarfile``
            lacks extraction filters (before 3.10.12 / 3.11.4).
        FileNotFoundError: *url* is a local path or ``file://`` URL naming no file.
        ValueError: *url* has a scheme urllib cannot open.
    """
    _require_extraction_filters()  # before anything is downloaded
    # Imported here: only a download needs them, not the commands that check an install.
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    url = package_url(url)
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() == "file" and parts.netloc in ("", "localhost"):
        # Said plainly; urllib's own report is "<urlopen error [Errno 2] ...>".
        local = Path(urllib.request.url2pathname(parts.path))
        if not local.is_file():
            raise FileNotFoundError(f"{local}: no such file")
    target = (target or install_dir()).expanduser().resolve()
    # Before the lock, so a directory refused gets no lock file beside it; a
    # run holding the lock only ever leaves *target* missing or stamped.
    if target.exists() and not (target / _STAMP).is_file() and any(target.iterdir()):
        raise RuntimeError(
            f"{target} is not empty and not a CARLA scenario-setup installed; "
            "pick an empty or new directory"
        )
    with _locked(target) as writable:
        return _install(url, target, force, writable)


def _install(url: str, target: Path, force: bool, writable: bool = True) -> Path:
    """:func:`install`'s work, with the lock on *target* held.

    Not *writable* (see :func:`_locked`), it only checks the install, and raises
    ``RuntimeError`` where it would have to download.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    if not writable and force:
        raise _cannot_install(target, force=True)
    if writable:
        # What an interrupted run left: an unfinished download and an old install
        # moved aside but not yet deleted. Neither is ever trusted as an install.
        for leftover in (_staging(target), _replaced(target)):
            shutil.rmtree(leftover, ignore_errors=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            **({} if force else _conditions(target, url)),
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=_TIMEOUT)  # noqa: S310 -- a URL the user chose
    except urllib.error.HTTPError as exc:
        if exc.code != 304:  # 304 Not Modified: the server says the install is current
            raise
        print(f"CARLA at {target} is up to date.")
        return _launcher_of(target)
    with response:
        source = {
            "url": url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "size": response.headers.get("Content-Length"),
        }
        if not force and _current(target, source):
            print(f"CARLA at {target} is up to date.")
            return _launcher_of(target)
        if not writable:
            raise _cannot_install(target)
        # Checked before the download, not before finding the install current
        # (a leftover that cannot be deleted does not stop a run that keeps the
        # install): the swap needs the name free, and would otherwise fail only
        # once the whole package is unpacked.
        if _replaced(target).exists():
            raise RuntimeError(
                f"{_replaced(target)}, a previous install moved aside, cannot be "
                "deleted (files held open by a CARLA server, perhaps on another "
                "node, or owned by another user such as root); delete it and run "
                "scenario-setup again"
            )
        size = int(source["size"]) if source["size"] else None
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = _staging(target)
        if size is not None:
            _check_room(target.parent, size)

        print(f"Downloading {url}\n         to {target}")
        stream = _ResumingStream(response, url, source, size, tqdm.write)
        try:
            with tqdm.wrapattr(
                stream,
                "read",
                total=size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as counted:
                _unpack(cast(IO[bytes], counted), staging)
            if launcher_in(staging) is None:
                raise RuntimeError(
                    f"{url} has no {LAUNCHER.name} (at its top, as CARLA 1.0 "
                    f"has it, or in {LAUNCHER.parent}/): not a CARLA Linux package"
                )
            (staging / _STAMP).write_text(json.dumps(source, indent=2) + "\n")
            _swap_into_place(staging, target)
        # Ctrl-C, a dead connection, a failed swap: no 30 GB left over.
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            stream.close()
    return _launcher_of(target)


@contextlib.contextmanager
def _locked(target: Path) -> Iterator[bool]:
    """Hold the lock on installing into *target*, waiting for it if taken.

    Yields whether the run may write *target*: ``False`` where the lock file
    could be opened read-only only.

    A ``flock`` on :func:`_lock_path`, released when the file is closed, so also
    when the process dies. The file is never deleted: a run that opened it just
    before the deletion would lock a file no later run sees. Waiting is a
    blocking ``flock``, which a Ctrl-C ends with ``KeyboardInterrupt``. Where
    the file system has no ``flock``, the install goes ahead unlocked.

    Checking an install needs no write access: where the lock file cannot be
    opened for writing (a read-only or another user's install), an existing one
    is opened read-only and locked shared, which still waits for a writer's
    exclusive lock: an NFS client emulates ``flock`` with byte-range locks, and
    refuses an exclusive one on a read-only descriptor (``EBADF``, which on such
    a descriptor counts as no ``flock``). Without a lock file the run goes ahead
    unlocked, as it can only find the install up to date or fail to write it.
    """
    import fcntl  # noqa: PLC0415

    lock = _lock_path(target)
    stream = _open_lock(lock)
    if stream is None:
        yield True
        return
    writable = stream.mode != "r"
    mode = fcntl.LOCK_EX if writable else fcntl.LOCK_SH
    with stream:
        try:
            fcntl.flock(stream.fileno(), mode | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"Waiting for another scenario-setup on {target} to finish "
                f"(lock {lock}) ...",
                flush=True,
            )
            fcntl.flock(stream.fileno(), mode)
        except OSError as exc:
            if exc.errno not in _NO_FLOCK_ERRNOS and (
                writable or exc.errno != errno.EBADF
            ):
                raise
            print(
                f"scenario-setup: no flock on {lock.parent} ({exc}); installing "
                "unlocked, so run only one scenario-setup for it at a time",
                file=sys.stderr,
            )
        yield writable


class _ResumingStream:
    """The body of *response*, reconnecting with a range request when it drops.

    The unpacking only reads forward, so when a read fails or the body ends
    short of its Content-Length, the rest is asked for with
    ``Range: bytes=<read so far>-`` and the stream carries on as if nothing
    happened. ``If-Range`` makes a download replaced in the meantime come back
    whole (200) instead of spliced onto the old one, which ends the install.
    A body that ends short and cannot be resumed raises ``RuntimeError``.
    """

    def __init__(
        self,
        response: IO[bytes],
        url: str,
        source: dict[str, Optional[str]],
        size: Optional[int],
        log: Callable[[str], None],
    ) -> None:
        self._response = response
        self._url = url
        self._size = size
        self._log = log
        self._offset = 0
        self._source = source
        etag = source["etag"]
        # If-Range takes a strong ETag or a date; a weak ETag never matches.
        self._validator = (
            etag if etag and not etag.startswith("W/") else source["last_modified"]
        )
        self._resumable = (
            url.lower().startswith(("http://", "https://"))
            and size is not None
            and self._validator is not None
        )

    def read(self, amount: Optional[int] = None) -> bytes:
        failures = 0
        while True:
            try:
                # read(-1) fails on 3.10/3.11's HTTPResponse; read() reads it all.
                if amount is None:
                    data = self._response.read()
                else:
                    data = self._response.read(amount)
            except (OSError, http.client.HTTPException) as exc:
                if not self._resumable:
                    raise
                error: Optional[BaseException] = exc
            else:
                if data or self._size is None or self._offset >= self._size:
                    self._offset += len(data)
                    return data
                if not self._resumable:
                    # Content-Length says there is more; urllib's read() just
                    # returns nothing when the connection closes early.
                    raise RuntimeError(
                        f"the download of {self._url} ended at {self._offset} of "
                        f"{self._size} bytes and cannot be resumed; run "
                        "scenario-setup again"
                    )
                error = None
            failures += 1
            self._reconnect(failures, error)

    def _reconnect(self, failures: int, error: Optional[BaseException]) -> None:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        while True:
            if failures > _RETRIES:
                raise RuntimeError(
                    f"connection lost at {self._offset} of {self._size} bytes; "
                    f"gave up after {_RETRIES} reconnections"
                ) from error
            reason = f" ({error})" if error is not None else ""
            self._log(
                f"Connection lost at {self._offset} of {self._size} bytes{reason}; "
                f"resuming (attempt {failures}/{_RETRIES})"
            )
            time.sleep(min(_RETRY_DELAY * 2 ** (failures - 1), 60.0))
            self.close()
            request = urllib.request.Request(
                self._url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Range": f"bytes={self._offset}-",
                    "If-Range": str(self._validator),
                },
            )
            try:
                response = urllib.request.urlopen(request, timeout=_TIMEOUT)  # noqa: S310
            except urllib.error.HTTPError as exc:
                exc.close()
                if exc.code < 500 and exc.code not in (408, 429):
                    raise  # 403, 404, 416...: retrying will not change the answer
                error = exc
                failures += 1
                continue
            except (OSError, http.client.HTTPException) as exc:
                error = exc
                failures += 1
                continue
            first, total = _content_range(response.headers.get("Content-Range"))
            if (
                response.status != 206
                or first != self._offset
                or total != self._size
                or not self._same_download(response.headers)
            ):
                response.close()
                raise RuntimeError(
                    f"{self._url} changed during the download or cannot resume "
                    f"(HTTP {response.status}); run scenario-setup again"
                ) from error
            self._response = response
            return

    def _same_download(self, headers: Message) -> bool:
        """Whether a 206 is of the download begun, for servers ignoring If-Range."""
        return all(
            headers.get(header) in (None, self._source[key])
            for header, key in (("ETag", "etag"), ("Last-Modified", "last_modified"))
            if self._source[key]
        )

    def close(self) -> None:
        try:
            self._response.close()
        except (OSError, http.client.HTTPException):
            pass


def _stamp(target: Path, shown: tuple[str, ...] = ()) -> dict[str, Optional[str]]:
    """What *target* was downloaded from, or nothing for an incomplete install.

    The values as the checked ``carla_install_paths`` takes them, strings or
    ``None``, converted so that it decides as it did on the JSON's own values.
    A stamp edited by hand can hold other types: such a value never equals the
    string it is compared with, so it becomes ``None``, which never does either
    (nor does a key missing). Under the keys in *shown*, whose value a request
    header carries, a value that is not falsy becomes its ``str()`` instead,
    which is what the header was given.
    """
    if launcher_in(target) is None:
        return {}
    try:
        stamp = json.loads((target / _STAMP).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(stamp, dict):
        return {}
    return {
        str(key): (
            value
            if value is None or isinstance(value, str)
            else str(value)
            if key in shown and value
            else None
        )
        for key, value in stamp.items()
    }


def _conditions(target: Path, url: str) -> dict[str, str]:
    """Headers that make the server answer 304 for the download *target* holds."""
    return _conditions_of(_stamp(target, ("etag", "last_modified")), url)


def _current(target: Path, source: dict[str, Optional[str]]) -> bool:
    """Whether *target* holds a complete install of the same download as *source*."""
    return _current_of(_stamp(target), source)


def _check_room(directory: Path, packed: int) -> None:
    needed = _room_needed(packed, _UNPACKED_PER_PACKED)
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise RuntimeError(
            f"CARLA needs about {needed / 2**30:.0f} GiB in {directory}, "
            f"which has {free / 2**30:.0f} GiB free"
        )


def _require_extraction_filters() -> None:
    """Refuse a Python whose ``tarfile`` has no extraction filters.

    ``extractall(filter=...)`` and ``tarfile.data_filter``, which keep the
    archive from writing outside the destination, came in Python 3.12 and in
    the 3.10.12 and 3.11.4 security releases; ``requires-python`` still admits
    the patch releases before those (Ubuntu 22.04 released with 3.10.4), where
    the extraction would otherwise fail with a ``TypeError`` traceback.

    Raises:
        RuntimeError: The running Python is one of those older patch releases.

    The filters themselves have known bypasses before 3.10.18, 3.11.13,
    3.12.11 and 3.13.4 (:data:`_FILTER_FIXES`): there, a warning is logged
    rather than the install refused, which would lock out Pythons still
    shipped (Ubuntu 24.04's 3.12.3).
    """
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError(
            f"scenario-setup needs tarfile's extraction filters, which Python "
            f"{platform.python_version()} lacks: upgrade Python to 3.10.12, "
            f"3.11.4, 3.12 or later"
        )
    major, minor, micro = sys.version_info[:3]
    _warn_of_filter_bypasses(major, minor, micro)


_FILTER_FIXES = {(3, 10): 18, (3, 11): 13, (3, 12): 11, (3, 13): 4}
"""The patch release of each minor version that fixed the extraction filters'
bypasses (CVE-2024-12718, CVE-2025-4138, CVE-2025-4330, CVE-2025-4517)."""


@functools.lru_cache(maxsize=None)
def _warn_of_filter_bypasses(major: int, minor: int, micro: int) -> None:
    """Warn, once per run, of a Python whose extraction filters can be bypassed."""
    fixed = _FILTER_FIXES.get((major, minor))
    if fixed is not None and micro < fixed:
        logger.warning(
            "Python %d.%d.%d's tarfile extraction filters can be bypassed by a "
            "crafted archive (CVE-2024-12718, CVE-2025-4138, CVE-2025-4330, "
            "CVE-2025-4517; fixed in %d.%d.%d): pass only a --url you trust, or "
            "upgrade Python",
            major,
            minor,
            micro,
            major,
            minor,
            fixed,
        )


class _ArchiveLayoutError(tarfile.TarError):
    """A CARLA package that is not one top directory holding the package."""


class _StripTop:
    """A tar extraction filter: drop each member's top directory, then the
    data filter.

    A CARLA package holds a single top directory
    (``Carla-<version>-Linux-Shipping/``, say) with the package in it, and
    that directory is what is dropped.  An archive with two top directories,
    or without the wrapper directory (a file at its top, or ``Linux/`` as its
    top directory), is refused with :class:`_ArchiveLayoutError` rather than
    merged or stripped of the wrong directory (#2233).  A filter sees the
    members one at a time, so an instance remembers the top directory of the
    first; use one per archive.

    A hard link names its target by the target's archive path, so that path
    loses the top directory too; otherwise extraction looks for the target
    under the dropped directory and, not finding it, re-reads the archive for
    it, which a stream (``r|gz``) cannot do. A hard link to anything outside
    the link's own top directory (or to that directory itself) has no place
    in the stripped tree and is refused, as the data filter refuses a link
    escaping the destination.

    A symbolic link's target is relative to the link's own directory, which
    moves with it, so it is left alone; the data filter refuses one that is
    absolute or reaches outside the destination.
    """

    def __init__(self) -> None:
        self.top: Optional[str] = None

    def __call__(self, member: tarfile.TarInfo, path: str) -> Optional[tarfile.TarInfo]:
        parts = PurePosixPath(member.name).parts
        if not parts:
            return None  # "./" itself
        if len(parts) == 1 and not member.isdir():
            raise _ArchiveLayoutError(
                f"the archive has {member.name!r} at its top, not in a single "
                "top directory: not a CARLA Linux package as released (one "
                f"directory holding {LAUNCHER.name})"
            )
        if self.top is None:
            if parts[0] == LAUNCHER.parts[0]:
                # The launcher's own directory: the package is not wrapped,
                # and stripping it would lose the launcher's place.
                raise _ArchiveLayoutError(
                    f"the archive's top directory is {parts[0]!r}, the one "
                    f"holding {LAUNCHER.name}, with no directory around it: not "
                    "a CARLA Linux package as released (one directory holding "
                    f"{LAUNCHER.name})"
                )
            self.top = parts[0]
        elif parts[0] != self.top:
            raise _ArchiveLayoutError(
                f"the archive has two top directories, {self.top!r} and "
                f"{parts[0]!r}: not a CARLA Linux package as released (one "
                f"directory holding {LAUNCHER.name})"
            )
        if len(parts) < 2:
            return None
        linkname = member.linkname
        if member.islnk():
            link_parts = PurePosixPath(linkname).parts
            if len(link_parts) < 2 or link_parts[0] != parts[0]:
                raise tarfile.LinkOutsideDestinationError(member, linkname)
            linkname = str(PurePosixPath(*link_parts[1:]))
        stripped = member.replace(
            name=str(PurePosixPath(*parts[1:])), linkname=linkname, deep=False
        )
        # The data filter refuses absolute paths, escapes and device files.
        return tarfile.data_filter(stripped, path)


def _unpack(stream: IO[bytes], destination: Path) -> None:
    """Unpack a gzipped tar *stream* into *destination*, dropping its top directory.

    ``tarfile``'s own ``r|gz`` takes a stream cut at a member boundary for the
    end of the archive and never checks the gzip trailer, so the gzip stream
    is read here, to its end once the archive is unpacked: one cut short
    raises ``EOFError``, a wrong CRC or length ``gzip.BadGzipFile``.
    """
    _require_extraction_filters()
    destination.mkdir(parents=True)
    # 1 MiB reads of the unpacked tar, not tarfile's default 10 KiB. The progress
    # bar counts gzip's reads of the download (8 KiB before 3.12, 128 KiB on).
    with gzip.GzipFile(fileobj=stream, mode="rb") as unzipped:
        with tarfile.open(fileobj=unzipped, mode="r|", bufsize=1 << 20) as archive:
            try:
                archive.extractall(destination, filter=_StripTop())
            except KeyError as exc:
                # tarfile's own report of a hard link whose target is not (yet)
                # on disk nor in the archive; a TarError, so main() reports it
                # cleanly.
                raise tarfile.ExtractError(
                    f"a hard link's target is not in the archive: {exc}"
                ) from exc
        while unzipped.read(1 << 20):  # the rest, up to the trailer's checks
            pass


_HASH_LENGTHS = (8, 7, 9, 10, 11, 12)
"""Abbreviated hash lengths a nightly's name may use, likeliest first: ``git log
%h`` abbreviates as the repository's size demands (8 for CARLA in 2026)."""

_DATE_SLACK_DAYS = 1
"""Days either side of a day a commit is looked for in: a nightly is named by
the committer's local date, the GitHub API answers in UTC."""

_ONE_DAY = datetime.timedelta(days=1)


def nightly_url(build: Optional[str] = None) -> str:
    """The URL of the CARLA nightly *build* names (``scenario-setup --nightly``).

    *build* is one of:

    - ``None`` or ``latest``: the latest nightly (:data:`NIGHTLY_URL`);
    - a nightly's own name, ``YYYYMMDD_<commit>`` (``20261124_23d487e3``), used
      as it is;
    - a date, ``YYYYMMDD`` or ``YYYY-MM-DD``: the last nightly built from a
      commit of that day;
    - a commit of ``ue58-dev`` (7 to 40 hex digits): the nightly built from it.

    A date or a commit is looked up in the branch's history (the GitHub API;
    ``GITHUB_TOKEN`` is sent if set) and the nightlies it names are asked for,
    since the download server lists none. CARLA's CI builds only some commits,
    and an old nightly may have been deleted, so the build asked for may not
    exist.

    Raises:
        ValueError: *build* is none of the above, or no nightly matches it.
        OSError: The GitHub API or the download server cannot be reached.
    """
    if build is None or build.strip().lower() in ("", "latest"):
        return NIGHTLY_URL
    name = build.strip().lower()
    if name.endswith(".tar.gz"):
        name = name[: -len(".tar.gz")]
    if re.fullmatch(r"\d{8}_[0-9a-f]{7,40}", name):
        _parse_day(name[:8], build)
        return f"{NIGHTLY_DIR_URL}{name}.tar.gz"
    day = _maybe_day(name)
    if day is not None:
        candidates = [
            (day, sha) for sha in _commits_between(day - _ONE_DAY, day + 2 * _ONE_DAY)
        ]
        what = f"of {day:%Y-%m-%d}"
    elif re.fullmatch(r"[0-9a-f]{7,40}", name):
        sha, committed = _commit(name)
        candidates = [
            (committed + offset * _ONE_DAY, sha)
            for offset in range(-_DATE_SLACK_DAYS, _DATE_SLACK_DAYS + 1)
        ]
        what = f"built from commit {sha[:12]}"
    else:
        raise ValueError(
            f"--nightly {build!r}: expected 'latest', a date (YYYYMMDD or "
            "YYYY-MM-DD), a commit hash, or a nightly's name (YYYYMMDD_<commit>)"
        )
    for length in _HASH_LENGTHS:
        for when, sha in candidates:
            url = f"{NIGHTLY_DIR_URL}{when:%Y%m%d}_{sha[:length]}.tar.gz"
            if _exists(url):
                return url
    raise ValueError(
        f"no CARLA nightly {what} on {NIGHTLY_DIR_URL}: CARLA's CI builds only "
        f"some commits of {NIGHTLY_BRANCH}, and old nightlies may be gone. Pick "
        "another date or commit, or name a nightly as YYYYMMDD_<commit>"
    )


def _parse_day(text: str, build: str) -> datetime.date:
    try:
        return datetime.datetime.strptime(text, "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"--nightly {build!r}: {text} is not a date") from None


def _maybe_day(name: str) -> Optional[datetime.date]:
    """The day *name* writes (``YYYYMMDD``, ``YYYY-MM-DD``), or ``None``.

    Eight digits that are no date are left to be read as a commit hash."""
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(name, pattern).date()
        except ValueError:
            continue
    return None


def _github(url: str) -> object:
    """The JSON the GitHub API answers *url* with."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310 -- a fixed https URL
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            raise ValueError(f"{url}: not found on GitHub") from None
        if exc.code in (403, 429):
            raise OSError(
                f"GitHub refused {url} (HTTP {exc.code}; anonymous requests are "
                "rate-limited): set GITHUB_TOKEN, or name the nightly as "
                "YYYYMMDD_<commit>"
            ) from None
        raise OSError(f"GitHub answered {url} with HTTP {exc.code}") from None


def _commits_between(first: datetime.date, end: datetime.date) -> list[str]:
    """The ``ue58-dev`` commits committed from *first* to before *end* (UTC),
    newest first."""
    import urllib.parse  # noqa: PLC0415

    query = urllib.parse.urlencode(
        {
            "sha": NIGHTLY_BRANCH,
            "since": f"{first:%Y-%m-%d}T00:00:00Z",
            "until": f"{end:%Y-%m-%d}T00:00:00Z",
            "per_page": "100",
        }
    )
    answer = _github(f"{NIGHTLY_COMMITS_API}?{query}")
    if not isinstance(answer, list):
        raise OSError(f"unexpected answer from {NIGHTLY_COMMITS_API}")
    return [str(entry["sha"]) for entry in answer if isinstance(entry, dict)]


def _commit(ref: str) -> tuple[str, datetime.date]:
    """The full hash of commit *ref* of CARLA and the day (UTC) it was committed."""
    answer = _github(f"{NIGHTLY_COMMITS_API}/{ref}")
    try:
        sha = str(answer["sha"])  # type: ignore[index]
        committed = str(answer["commit"]["committer"]["date"])  # type: ignore[index]
    except (KeyError, TypeError):
        raise OSError(f"unexpected answer for commit {ref} from GitHub") from None
    return sha, datetime.date.fromisoformat(committed[:10])


def _exists(url: str) -> bool:
    """Whether the download server has *url* (a ``HEAD`` request)."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT):  # noqa: S310 -- a fixed https URL
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return False
        raise OSError(f"{url}: HTTP {exc.code}") from None


def main(argv: Optional[list[str]] = None) -> int:
    """``scenario-setup``: download and unpack a CARLA server.

    Ctrl-C prints one line, not a traceback, and exits 130 (#2333), once the
    install has cleaned up after itself (its lock, a partial download).
    """
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("scenario-setup: interrupted", file=sys.stderr)
        return 130


def _main(argv: Optional[list[str]]) -> int:
    parser = argparse.ArgumentParser(
        prog="scenario-setup",
        description="Download a CARLA server for the scenario runner to launch.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url",
        default=None,
        help="CARLA Linux package (.tar.gz) to install, as a URL or a local path "
        "(default: the CARLA 1.0 release)",
    )
    source.add_argument(
        "--nightly",
        nargs="?",
        const="latest",
        default=None,
        metavar="BUILD",
        help="install CARLA's nightly build (the ue58-dev branch) instead of the "
        "release: the latest, or BUILD -- a date (YYYYMMDD: the last nightly of "
        "that day), a ue58-dev commit, or a nightly's name (YYYYMMDD_<commit>); "
        "run the scenarios with TYPESAFE_CARLA_CARLA_REF=ue58-dev",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"where to unpack it (default: {install_dir()})",
    )
    parser.add_argument(
        "--force", action="store_true", help="download even if the install is current"
    )
    args = parser.parse_args(argv)
    machine = platform.machine()
    official: Optional[str] = None
    if args.nightly is not None:
        official = "CARLA's nightly build"
    elif args.url is None:
        official = "CARLA 1.0"
    if official is not None and (
        not sys.platform.startswith("linux") or machine not in PACKAGE_MACHINES
    ):
        print(
            f"scenario-setup: {official} is for Linux x86_64 and does not "
            f"run on {sys.platform} {machine}. Run the CARLA server on an x86_64 host and point the "
            f"runner at it (server.host=<address>), or pass --url to install another "
            f"CARLA package.",
            file=sys.stderr,
        )
        return 1
    try:
        if args.nightly is not None:
            url = nightly_url(args.nightly)
            if url != NIGHTLY_URL:
                print(f"CARLA nightly: {url}")
        else:
            url = RELEASE_URL if args.url is None else args.url
        launcher = install(url, args.dir, args.force)
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,  # a gzip stream cut short (gzip.BadGzipFile is an OSError)
        zlib.error,
        tarfile.TarError,
        http.client.HTTPException,
    ) as exc:
        message = " ".join(str(exc).split())  # one line, whatever the exception says
        print(f"scenario-setup: {message}", file=sys.stderr)
        return 1
    print(f"CARLA launcher: {launcher}")
    if args.dir is not None:
        print(f"Set CARLA_EXECUTABLE={launcher} for the scenario runner to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
