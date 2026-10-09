"""``scenario-setup``: download a CARLA server for the framework to launch.

By default it fetches CARLA's nightly Linux build (the ``ue5-dev`` branch) and
unpacks it into ``~/.autoware_carla_scenario/bin/carla``, where
:class:`~autoware_carla_scenario.server.CarlaServerManager` finds its launcher
when ``CARLA_EXECUTABLE`` is not set::

    scenario-setup            # download, or keep an install that is current
    scenario-setup --force    # download again regardless

The archive (~16 GB, ~30 GB unpacked) is unpacked while it downloads, so it is
never kept on disk; the new tree replaces the old one only once it is complete.
A connection that drops mid-download is picked up where it stopped with a range
request, so a flaky link costs a retry rather than the whole transfer.
``AUTOWARE_CARLA_SCENARIO_HOME`` moves ``~/.autoware_carla_scenario``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import sys
import tarfile
import time
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import IO, Callable, Optional, cast

HOME_ENV = "AUTOWARE_CARLA_SCENARIO_HOME"
"""Overrides the framework's home directory, ``~/.autoware_carla_scenario``."""

NIGHTLY_URL = (
    "https://s3.us-east-005.backblazeb2.com/carla-releases/Linux/Dev/"
    "CARLA_UE5_Latest.tar.gz"
)
"""CARLA's nightly Linux build, as linked from its download page."""

LAUNCHER = PurePosixPath("Linux/CarlaUnreal.sh")
"""The server's launcher, relative to the unpacked package."""

_STAMP = ".download.json"
"""What the install was downloaded from, to tell whether it is still current."""

_UNPACKED_PER_PACKED = 2.0
"""The nightly unpacks to about twice its archive."""

_TIMEOUT = 60.0
"""Seconds a read may stall before the connection counts as dropped."""

_RETRIES = 8
"""Reconnections in a row that may fail to deliver a byte before giving up."""

_RETRY_DELAY = 2.0
"""Seconds before the first reconnection; each failed one doubles it, up to 60."""


def install_dir() -> Path:
    """Where ``scenario-setup`` puts CARLA: ``<home>/bin/carla``."""
    home = os.environ.get(HOME_ENV, "").strip() or "~/.autoware_carla_scenario"
    return Path(home).expanduser() / "bin" / "carla"


def installed_executable() -> Optional[Path]:
    """The launcher of the CARLA ``scenario-setup`` installed, if there is one."""
    launcher = install_dir() / LAUNCHER
    return launcher if launcher.is_file() else None


def install(
    url: str = NIGHTLY_URL, target: Optional[Path] = None, force: bool = False
) -> Path:
    """Download the CARLA package at *url* into *target* and return its launcher.

    An install already made from the same download (same ETag, or Last-Modified
    and size) is kept unless *force*. The package's single top-level directory
    is dropped, so the launcher is always ``<target>/Linux/CarlaUnreal.sh``.

    Raises:
        RuntimeError: There is not room for the package, it has no launcher, or
            *target* holds something other than a CARLA this function installed.
            A previous install is left as it was, and so is anything else.
    """
    if not hasattr(tarfile, "data_filter"):  # added in 3.10.12 / 3.11.4
        raise RuntimeError("scenario-setup needs Python 3.10.12, 3.11.4 or later")
    # The runner imports this module for install_dir() alone; these it does not need.
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    target = (target or install_dir()).expanduser().resolve()
    if target.exists() and not (target / _STAMP).is_file() and any(target.iterdir()):
        raise RuntimeError(
            f"{target} is not empty and not a CARLA scenario-setup installed; "
            "pick an empty or new directory"
        )
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
        size = int(source["size"]) if source["size"] else None
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)  # an interrupted run's
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
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target / LAUNCHER


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


def _content_range(
    content_range: Optional[str],
) -> tuple[Optional[int], Optional[int]]:
    """First byte and size of a ``Content-Range: bytes <first>-<last>/<size>``."""
    unit, _, span = (content_range or "").strip().partition(" ")
    first, _, rest = span.partition("-")
    _, _, total = rest.partition("/")
    if unit != "bytes" or not first.isdigit() or not total.isdigit():
        return None, None
    return int(first), int(total)


def _stamp(target: Path) -> dict[str, Optional[str]]:
    """What *target* was downloaded from, or nothing for an incomplete install."""
    if not (target / LAUNCHER).is_file():
        return {}
    try:
        stamp = json.loads((target / _STAMP).read_text())
    except (OSError, ValueError):
        return {}
    return stamp if isinstance(stamp, dict) else {}


def _conditions(target: Path, url: str) -> dict[str, str]:
    """Headers that make the server answer 304 for the download *target* holds."""
    stamp = _stamp(target)
    if stamp.get("url") != url:
        return {}
    headers = {}
    if stamp.get("etag"):
        headers["If-None-Match"] = str(stamp["etag"])
    if stamp.get("last_modified"):
        headers["If-Modified-Since"] = str(stamp["last_modified"])
    return headers


def _current(target: Path, source: dict[str, Optional[str]]) -> bool:
    """Whether *target* holds a complete install of the same download as *source*."""
    stamp = _stamp(target)
    if stamp.get("url") != source["url"]:
        return False
    if source["etag"]:
        return bool(stamp.get("etag") == source["etag"])
    keys = ("last_modified", "size")
    return all(source[k] for k in keys) and all(stamp.get(k) == source[k] for k in keys)


def _check_room(directory: Path, packed: int) -> None:
    needed = int(packed * _UNPACKED_PER_PACKED)
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise RuntimeError(
            f"CARLA needs about {needed / 2**30:.0f} GiB in {directory}, "
            f"which has {free / 2**30:.0f} GiB free"
        )


def _unpack(stream: IO[bytes], destination: Path) -> None:
    """Unpack a gzipped tar *stream* into *destination*, dropping its top directory."""

    def strip_top(member: tarfile.TarInfo, path: str) -> Optional[tarfile.TarInfo]:
        inner = PurePosixPath(*PurePosixPath(member.name).parts[1:])
        if not inner.parts:
            return None
        # The data filter refuses absolute paths, escapes and device files.
        return tarfile.data_filter(member.replace(name=str(inner), deep=False), path)

    destination.mkdir(parents=True)
    # 1 MiB reads: the default 10 KiB makes 1.6 M calls through the progress bar.
    with tarfile.open(fileobj=stream, mode="r|gz", bufsize=1 << 20) as archive:
        archive.extractall(destination, filter=strip_top)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scenario-setup",
        description="Download a CARLA server for the scenario runner to launch.",
    )
    parser.add_argument(
        "--url",
        default=NIGHTLY_URL,
        help="CARLA Linux package (.tar.gz) to install (default: the nightly build)",
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
    try:
        launcher = install(args.url, args.dir, args.force)
    except (OSError, RuntimeError, tarfile.TarError, http.client.HTTPException) as exc:
        print(f"scenario-setup: {exc}", file=sys.stderr)
        return 1
    print(f"CARLA launcher: {launcher}")
    if args.dir is not None:
        print(f"Set CARLA_EXECUTABLE={launcher} for the scenario runner to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
