"""``scenario-setup``: download a CARLA server for the framework to launch.

By default it fetches CARLA's nightly Linux build (the ``ue5-dev`` branch) and
unpacks it into ``~/.autoware_carla_scenario/bin/carla``, where
:class:`~autoware_carla_scenario.server.CarlaServerManager` finds its launcher
when ``CARLA_EXECUTABLE`` is not set::

    scenario-setup            # download, or keep an install that is current
    scenario-setup --force    # download again regardless

The archive (~16 GB, ~30 GB unpacked) is unpacked while it downloads, so it is
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
(aarch64 included, where the framework and its client do run) the nightly is
refused rather than downloaded: run the server on an x86_64 host and point
``server.host`` at it. ``--url`` still installs a package of one's choosing,
from a URL or from a local path (``--url ~/Downloads/CARLA.tar.gz``).
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import http.client
import json
import platform
import shutil
import sys
import tarfile
import time
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
    _replaced,
    _room_needed,
    _staging,
    _swap_into_place,
)
from .carla_install_paths import _SCHEME as _SCHEME
from .carla_install_paths import _UNPACKED_PER_PACKED as _UNPACKED_PER_PACKED
from .carla_install_paths import HOME_ENV as HOME_ENV
from .carla_install_paths import install_dir as install_dir
from .carla_install_paths import installed_executable as installed_executable
from .carla_install_paths import package_url as package_url

NIGHTLY_URL = (
    "https://s3.us-east-005.backblazeb2.com/carla-releases/Linux/Dev/"
    "CARLA_UE5_Latest.tar.gz"
)
"""CARLA's nightly Linux build, as linked from its download page."""

NIGHTLY_MACHINES = frozenset({"x86_64"})
"""The machines (``platform.machine()``) the nightly build runs on."""

LAUNCHER = PurePosixPath(_LAUNCHER)
"""The server's launcher, relative to the unpacked package."""

_TIMEOUT = 60.0
"""Seconds a read may stall before the connection counts as dropped."""

_RETRIES = 8
"""Reconnections in a row that may fail to deliver a byte before giving up."""

_RETRY_DELAY = 2.0
"""Seconds before the first reconnection; each failed one doubles it, up to 60."""


def install(
    url: str = NIGHTLY_URL, target: Optional[Path] = None, force: bool = False
) -> Path:
    """Download the CARLA package at *url* into *target* and return its launcher.

    *url* may also be a local path (see :func:`package_url`); it is read in place.
    An install already made from the same download (same ETag, or Last-Modified
    and size) is kept unless *force*. The package's single top-level directory
    is dropped, so the launcher is always ``<target>/Linux/CarlaUnreal.sh``.
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
        raise _cannot_install(target)
    if writable:
        # What an interrupted run left: an unfinished download and an old install
        # moved aside but not yet deleted. Neither is ever trusted as an install.
        for leftover in (_staging(target), _replaced(target)):
            shutil.rmtree(leftover, ignore_errors=True)
    request = urllib.request.Request(
        url, headers={} if force else _conditions(target, url)
    )
    try:
        response = urllib.request.urlopen(request, timeout=_TIMEOUT)  # noqa: S310 -- a URL the user chose
    except urllib.error.HTTPError as exc:
        if exc.code != 304:  # 304 Not Modified: the server says the install is current
            raise
        print(f"CARLA at {target} is up to date.")
        return target / LAUNCHER
    with response:
        source = {
            "url": url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "size": response.headers.get("Content-Length"),
        }
        if not force and _current(target, source):
            print(f"CARLA at {target} is up to date.")
            return target / LAUNCHER
        if not writable:
            raise _cannot_install(target)
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
            if not (staging / LAUNCHER).is_file():
                raise RuntimeError(
                    f"{url} has no {LAUNCHER}: not a CARLA Linux package"
                )
        except BaseException:  # Ctrl-C and a dead connection too: no 30 GB left over
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            stream.close()
    (staging / _STAMP).write_text(json.dumps(source, indent=2) + "\n")
    _swap_into_place(staging, target)
    return target / LAUNCHER


def _cannot_install(target: Path) -> RuntimeError:
    """The error of a run that can only check *target* but would have to install."""
    state = "out of date" if (target / LAUNCHER).is_file() else "not installed"
    return RuntimeError(
        f"CARLA at {target} is {state}, and this run cannot write there "
        f"({_lock_path(target)} cannot be opened for writing); run scenario-setup "
        "as a user who can write to it, or pass --dir to install elsewhere"
    )


#: The errors of a ``flock`` the file system does not support (some NFS mounts).
_NO_FLOCK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK, errno.EINVAL}
)


#: The errors of a lock file that cannot be created or written: a read-only
#: file system, or a directory of another user's.
_UNWRITABLE_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


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


def _open_lock(lock: Path) -> Optional[IO[str]]:
    """*lock* opened to ``flock``: for writing, else read-only, else nothing.

    Raises:
        OSError: *lock* cannot be created for a reason other than permissions
            or a read-only file system.
    """
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        return open(lock, "a")  # noqa: SIM115 -- the caller closes it
    except OSError as exc:
        if exc.errno not in _UNWRITABLE_ERRNOS:
            raise
        error = exc
    try:
        return open(lock)  # noqa: SIM115 -- for a shared lock, see _locked
    except OSError:
        print(
            f"scenario-setup: cannot create {lock} ({error}); going ahead "
            "unlocked, so run only one scenario-setup for it at a time",
            file=sys.stderr,
        )
        return None


class _ResumingStream:
    """The body of *response*, reconnecting with a range request when it drops.

    ``tarfile`` in ``r|gz`` mode only reads forward, so when a read fails or the
    body ends short of its Content-Length, the rest is asked for with
    ``Range: bytes=<read so far>-`` and the stream carries on as if nothing
    happened. ``If-Range`` makes a download replaced in the meantime come back
    whole (200) instead of spliced onto the old one, which ends the install.
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
                    return data  # tarfile reports the archive as truncated
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
    if not (target / LAUNCHER).is_file():
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
    """
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError(
            f"scenario-setup needs tarfile's extraction filters, which Python "
            f"{platform.python_version()} lacks: upgrade Python to 3.10.12, "
            f"3.11.4, 3.12 or later"
        )


def _strip_top(member: tarfile.TarInfo, path: str) -> Optional[tarfile.TarInfo]:
    """A tar extraction filter: drop *member*'s top directory, then the data filter.

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
    parts = PurePosixPath(member.name).parts
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
    """Unpack a gzipped tar *stream* into *destination*, dropping its top directory."""
    _require_extraction_filters()
    destination.mkdir(parents=True)
    # 1 MiB reads: the default 10 KiB makes 1.6 M calls through the progress bar.
    with tarfile.open(fileobj=stream, mode="r|gz", bufsize=1 << 20) as archive:
        try:
            archive.extractall(destination, filter=_strip_top)
        except KeyError as exc:
            # tarfile's own report of a hard link whose target is not (yet) on
            # disk nor in the archive; a TarError, so main() reports it cleanly.
            raise tarfile.ExtractError(
                f"a hard link's target is not in the archive: {exc}"
            ) from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scenario-setup",
        description="Download a CARLA server for the scenario runner to launch.",
    )
    parser.add_argument(
        "--url",
        default=NIGHTLY_URL,
        help="CARLA Linux package (.tar.gz) to install, as a URL or a local path "
        "(default: the nightly build)",
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
    if args.url == NIGHTLY_URL and (
        not sys.platform.startswith("linux") or machine not in NIGHTLY_MACHINES
    ):
        print(
            f"scenario-setup: CARLA's nightly build is for Linux x86_64 and does not "
            f"run on {sys.platform} {machine}. Run the CARLA server on an x86_64 host and point the "
            f"runner at it (server.host=<address>), or pass --url to install another "
            f"CARLA package.",
            file=sys.stderr,
        )
        return 1
    try:
        launcher = install(args.url, args.dir, args.force)
    except (
        OSError,
        RuntimeError,
        ValueError,
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
