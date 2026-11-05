"""Unit tests for ``scenario-setup``, against small packages shaped like CARLA's."""

from __future__ import annotations

import errno
import fcntl
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Iterator
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Optional
import urllib.error

import pytest

from autoware_carla_scenario import carla_install
from autoware_carla_scenario.carla_install import (
    HOME_ENV,
    LAUNCHER,
    install,
    install_dir,
    installed_executable,
)

_TOP = "Carla-0.10.0-Linux-Shipping"


def _package(path: Path, files: dict[str, bytes]) -> str:
    """A gzipped tar of *files* (names inside the archive) and its file:// URL."""
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
    return path.as_uri()


def _carla(path: Path, version: bytes = b"nightly") -> str:
    return _package(
        path,
        {f"{_TOP}/{LAUNCHER}": b"#!/bin/sh\n", f"{_TOP}/VERSION": version},
    )


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "home"))
    return tmp_path / "home"


def test_it_installs_under_the_framework_home(home: Path, tmp_path: Path) -> None:
    assert install_dir() == home / "bin" / "carla"
    assert installed_executable() is None
    launcher = install(_carla(tmp_path / "carla.tar.gz"))
    assert launcher == install_dir() / LAUNCHER
    assert installed_executable() == launcher
    assert launcher.stat().st_mode & 0o111, "the launcher stays executable"
    assert (install_dir() / "VERSION").read_bytes() == b"nightly"


def test_a_current_install_is_kept(home: Path, tmp_path: Path) -> None:
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    (install_dir() / "VERSION").write_bytes(b"touched")
    install(url)
    assert (install_dir() / "VERSION").read_bytes() == b"touched"
    install(url, force=True)
    assert (install_dir() / "VERSION").read_bytes() == b"nightly"


def test_a_newer_download_replaces_the_install(home: Path, tmp_path: Path) -> None:
    archive = tmp_path / "carla.tar.gz"
    install(_carla(archive))
    _carla(archive, b"a later night" * 10)  # another size: another download
    install(archive.as_uri())
    assert (install_dir() / "VERSION").read_bytes() == b"a later night" * 10


def test_a_package_without_a_launcher_leaves_the_install_alone(
    home: Path, tmp_path: Path
) -> None:
    install(_carla(tmp_path / "carla.tar.gz"))
    other = _package(tmp_path / "other.tar.gz", {f"{_TOP}/README": b"not CARLA"})
    with pytest.raises(RuntimeError, match="CarlaUnreal.sh"):
        install(other)
    assert installed_executable() is not None
    assert not install_dir().with_name("carla.partial").exists()


def test_members_escaping_the_install_are_refused(home: Path, tmp_path: Path) -> None:
    evil = _package(tmp_path / "evil.tar.gz", {f"{_TOP}/../../outside": b"x"})
    with pytest.raises(tarfile.TarError):
        install(evil)
    assert not (home / "outside").exists()


def _linked_package(links: dict[str, tuple[bytes, str]]) -> bytes:
    """A gzipped tar of ``Top/Linux/a`` plus *links*: ``name -> (type, linkname)``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        target = tarfile.TarInfo(f"{_TOP}/Linux/a")
        target.size = 1
        archive.addfile(target, io.BytesIO(b"x"))
        for name, (kind, linkname) in links.items():
            link = tarfile.TarInfo(name)
            link.type = kind
            link.linkname = linkname
            archive.addfile(link)
    return buffer.getvalue()


def _unpack_as(mode: str, data: bytes, destination: Path) -> None:
    """Unpack *data* as ``scenario-setup`` does (``stream``) or from a seekable file."""
    if mode == "stream":
        carla_install._unpack(io.BytesIO(data), destination)
        return
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(destination, filter=carla_install._strip_top)


_MODES = pytest.mark.parametrize("mode", ["stream", "seekable"])


@_MODES
def test_hard_links_lose_the_top_directory_too(mode: str, tmp_path: Path) -> None:
    data = _linked_package(
        {
            f"{_TOP}/Linux/b": (tarfile.LNKTYPE, f"{_TOP}/Linux/a"),
            f"{_TOP}/c": (tarfile.LNKTYPE, f"./{_TOP}/Linux/./a"),
        }
    )
    _unpack_as(mode, data, tmp_path / "out")
    a = tmp_path / "out" / "Linux" / "a"
    for link in (tmp_path / "out" / "Linux" / "b", tmp_path / "out" / "c"):
        assert link.read_bytes() == b"x"
        assert os.path.samefile(link, a), "a hard link, not a copy"


@_MODES
def test_symbolic_links_keep_their_relative_target(mode: str, tmp_path: Path) -> None:
    data = _linked_package(
        {
            f"{_TOP}/Linux/b": (tarfile.SYMTYPE, "a"),
            f"{_TOP}/c": (tarfile.SYMTYPE, "Linux/a"),
        }
    )
    _unpack_as(mode, data, tmp_path / "out")
    out = tmp_path / "out"
    # Relative to the link's own directory, which moved with it: not rewritten.
    assert os.readlink(out / "Linux" / "b") == "a"
    assert os.readlink(out / "c") == "Linux/a"
    assert (out / "Linux" / "b").read_bytes() == (out / "c").read_bytes() == b"x"


@_MODES
@pytest.mark.parametrize(
    "link",
    [
        (tarfile.LNKTYPE, "Elsewhere/Linux/a"),  # another top directory
        (tarfile.LNKTYPE, "a"),  # a top-level member
        (tarfile.LNKTYPE, _TOP),  # the top directory itself
        (tarfile.LNKTYPE, f"{_TOP}/../secret"),  # outside the destination
        (tarfile.LNKTYPE, f"{_TOP}/Linux/../../secret"),
        (tarfile.LNKTYPE, "/etc/passwd"),
        (tarfile.SYMTYPE, "../../secret"),
        (tarfile.SYMTYPE, "/etc/passwd"),
    ],
)
def test_links_escaping_the_tree_are_refused(
    mode: str, link: tuple[bytes, str], tmp_path: Path
) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"secret")
    data = _linked_package({f"{_TOP}/Linux/b": link})
    with pytest.raises(tarfile.FilterError):
        _unpack_as(mode, data, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out" / "Linux" / "b")
    assert secret.stat().st_nlink == 1


def test_a_hard_link_to_a_missing_member_is_reported(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "carla.tar.gz"
    archive.write_bytes(
        _linked_package({f"{_TOP}/Linux/b": (tarfile.LNKTYPE, f"{_TOP}/missing")})
    )
    assert carla_install.main(["--url", archive.as_uri()]) == 1
    assert "hard link's target is not in the archive" in capsys.readouterr().err
    assert not install_dir().exists()


def test_it_installs_a_package_with_hard_links(home: Path, tmp_path: Path) -> None:
    archive = tmp_path / "carla.tar.gz"
    with tarfile.open(archive, "w:gz") as packed:
        launcher = tarfile.TarInfo(f"{_TOP}/{LAUNCHER}")
        launcher.size = 10
        launcher.mode = 0o755
        packed.addfile(launcher, io.BytesIO(b"#!/bin/sh\n"))
        link = tarfile.TarInfo(f"{_TOP}/Linux/CarlaUnreal-Linux-Shipping.sh")
        link.type = tarfile.LNKTYPE
        link.linkname = f"{_TOP}/{LAUNCHER}"
        packed.addfile(link)
    assert carla_install.main(["--url", archive.as_uri()]) == 0
    installed = install_dir() / "Linux" / "CarlaUnreal-Linux-Shipping.sh"
    assert os.path.samefile(installed, install_dir() / LAUNCHER)


def test_it_refuses_a_disk_too_small(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(carla_install, "_UNPACKED_PER_PACKED", 1e12)
    with pytest.raises(RuntimeError, match="GiB"):
        install(_carla(tmp_path / "carla.tar.gz"))
    assert installed_executable() is None


def test_the_cli_reports_failures(home: Path, tmp_path: Path) -> None:
    assert carla_install.main(["--url", (tmp_path / "missing.tar.gz").as_uri()]) == 1
    assert carla_install.main(["--url", _carla(tmp_path / "carla.tar.gz")]) == 0
    assert installed_executable() is not None


@pytest.fixture
def no_extraction_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``tarfile`` as Python 3.10.0-3.10.11 and 3.11.0-3.11.3 have it."""
    monkeypatch.delattr(tarfile, "data_filter")
    monkeypatch.setattr(carla_install.platform, "python_version", lambda: "3.10.4")


@pytest.mark.usefixtures("no_extraction_filters")
def test_a_python_without_extraction_filters_is_reported_on_one_line(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert carla_install.main(["--url", _carla(tmp_path / "carla.tar.gz")]) == 1
    captured = capsys.readouterr()
    assert captured.err == (
        "scenario-setup: scenario-setup needs tarfile's extraction filters, which "
        "Python 3.10.4 lacks: upgrade Python to 3.10.12, 3.11.4, 3.12 or later\n"
    )
    assert "Downloading" not in captured.out  # refused before anything is fetched
    assert installed_executable() is None
    assert not install_dir().with_name("carla.partial").exists()


@pytest.mark.usefixtures("no_extraction_filters")
def test_unpacking_without_extraction_filters_extracts_nothing(
    tmp_path: Path,
) -> None:
    _carla(tmp_path / "carla.tar.gz")
    data = (tmp_path / "carla.tar.gz").read_bytes()
    with pytest.raises(RuntimeError, match=r"upgrade Python to 3\.10\.12"):
        carla_install._unpack(io.BytesIO(data), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_missing_local_path_is_reported_on_one_line(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.tar.gz"
    assert carla_install.main(["--url", str(missing)]) == 1
    err = capsys.readouterr().err
    assert err == f"scenario-setup: {missing}: no such file\n"
    assert installed_executable() is None


def test_an_unknown_url_type_is_reported_on_one_line(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert carla_install.main(["--url", "nosuch://host/carla.tar.gz"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("scenario-setup: ") and err.count("\n") == 1
    assert "nosuch" in err
    # A malformed URL is a ValueError, not urllib's URLError.
    assert carla_install.main(["--url", "http://[::1/carla.tar.gz"]) == 1
    err = capsys.readouterr().err
    assert err == "scenario-setup: Invalid IPv6 URL\n"


def _local_carla(path: Path) -> None:
    _package(path, {f"top/{LAUNCHER}": b"#!/bin/sh\n", "top/VERSION": b"local"})


def test_it_installs_from_an_absolute_local_path(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "CARLA.tar.gz"
    _local_carla(archive)
    assert carla_install.main(["--url", str(archive)]) == 0
    assert installed_executable() == install_dir() / LAUNCHER
    assert (install_dir() / "VERSION").read_bytes() == b"local"
    stamp = json.loads((install_dir() / ".download.json").read_text())
    assert stamp["url"] == archive.as_uri()
    assert stamp["size"] == str(archive.stat().st_size)
    # The same path again is the same download: kept, not unpacked anew.
    (install_dir() / "VERSION").write_bytes(b"touched")
    assert carla_install.main(["--url", str(archive)]) == 0
    assert "up to date" in capsys.readouterr().out
    assert (install_dir() / "VERSION").read_bytes() == b"touched"


def test_it_installs_from_a_relative_local_path(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "downloads").mkdir()
    _local_carla(tmp_path / "downloads" / "CARLA.tar.gz")
    monkeypatch.chdir(tmp_path)
    assert install("./downloads/CARLA.tar.gz") == install_dir() / LAUNCHER
    assert (install_dir() / "VERSION").read_bytes() == b"local"
    monkeypatch.chdir(tmp_path / "downloads")
    assert carla_install.main(["--url", "CARLA.tar.gz", "--force"]) == 0
    assert installed_executable() is not None


def test_a_local_path_expands_the_home_directory(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _local_carla(tmp_path / "CARLA.tar.gz")
    assert carla_install.package_url("~/CARLA.tar.gz") == (
        (tmp_path / "CARLA.tar.gz").resolve().as_uri()
    )
    assert install("~/CARLA.tar.gz") == install_dir() / LAUNCHER


def test_a_url_is_passed_on_unchanged(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    url = "http://downloads.invalid/Carla/CARLA.tar.gz?token=a b"
    assert carla_install.package_url(url) == url
    assert carla_install.package_url("HTTPS://x.invalid/c.tar.gz") == (
        "HTTPS://x.invalid/c.tar.gz"
    )
    # urllib opens file: without the //; it is not a relative path "file:".
    assert carla_install.package_url("file:/tmp/c.tar.gz") == "file:/tmp/c.tar.gz"
    assert carla_install.package_url("C:/c.tar.gz").startswith("file:///")
    assert carla_install.package_url("host:c.tar.gz").startswith("file:///")
    asked: list[str] = []

    def refused(request: urllib.request.Request, **_: object) -> None:
        asked.append(request.full_url)
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(urllib.error.HTTPError):
        install(url)
    assert asked == [url]


def test_the_nightly_is_refused_where_it_cannot_run(
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(carla_install.platform, "machine", lambda: "aarch64")
    assert carla_install.main([]) == 1
    assert "server.host" in capsys.readouterr().err
    assert installed_executable() is None
    # A package of one's own choosing is still installed.
    assert carla_install.main(["--url", _carla(tmp_path / "carla.tar.gz")]) == 0
    assert installed_executable() is not None


def test_an_unchanged_download_is_not_fetched(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp makes the request conditional; a 304 leaves the install alone."""
    import urllib.error
    import urllib.request

    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    stamp = json.loads((install_dir() / ".download.json").read_text())
    asked: list[urllib.request.Request] = []

    def not_modified(request: urllib.request.Request, **_: object) -> None:
        asked.append(request)
        raise urllib.error.HTTPError(url, 304, "Not Modified", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", not_modified)
    assert install(url) == install_dir() / LAUNCHER
    assert asked[0].get_header("If-modified-since") == stamp["last_modified"]


def test_a_directory_it_did_not_install_is_left_alone(
    home: Path, tmp_path: Path
) -> None:
    mine = tmp_path / "simulators"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    with pytest.raises(RuntimeError, match="not empty"):
        install(_carla(tmp_path / "carla.tar.gz"), mine)
    assert (mine / "notes.txt").read_text() == "keep me"


def test_an_interrupted_download_leaves_nothing_behind(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(stream: object, destination: Path) -> None:
        destination.mkdir(parents=True)
        (destination / "half").write_bytes(b"x")
        raise KeyboardInterrupt

    monkeypatch.setattr(carla_install, "_unpack", interrupted)
    with pytest.raises(KeyboardInterrupt):
        install(_carla(tmp_path / "carla.tar.gz"))
    assert not install_dir().with_name("carla.partial").exists()


class _Origin:
    """A local server for one archive that answers range requests like CARLA's.

    ``cut`` lists, per request, after how many bytes of the body the connection
    drops (``None``: it does not); ``etag`` can change mid-download.
    """

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.etag = '"nightly-1"'
        self.cut: list[Optional[int]] = []
        self.ranges: list[Optional[str]] = []
        self.honours_if_range = True
        self.gone = False  # resumes answer 404
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                wanted = self.headers.get("Range")
                origin.ranges.append(wanted)
                if wanted and origin.gone:
                    self.send_error(404)
                    return
                start = 0
                matches = self.headers.get("If-Range") in (None, origin.etag)
                if wanted and (matches or not origin.honours_if_range):
                    start = int(wanted.removeprefix("bytes=").rstrip("-"))
                part = origin.body[start:]
                self.send_response(206 if start else 200)
                if start:
                    last = len(origin.body) - 1
                    self.send_header(
                        "Content-Range", f"bytes {start}-{last}/{len(origin.body)}"
                    )
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", origin.etag)
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                cut = origin.cut.pop(0) if origin.cut else None
                self.wfile.write(part if cut is None else part[:cut])
                self.wfile.flush()
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/carla.tar.gz"


@pytest.fixture
def origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Origin]:
    for proxy in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(proxy, raising=False)
    monkeypatch.setattr(carla_install, "_RETRY_DELAY", 0.0)
    # Incompressible, so the archive is big enough to cut in the middle.
    noise = os.urandom(256 * 1024)
    archive = tmp_path / "served.tar.gz"
    _package(
        archive,
        {f"{_TOP}/{LAUNCHER}": b"#!/bin/sh\n", f"{_TOP}/noise": noise},
    )
    served = _Origin(archive.read_bytes())
    thread = threading.Thread(target=served.server.serve_forever, daemon=True)
    thread.start()
    yield served
    served.server.shutdown()
    served.server.server_close()


def test_a_dropped_download_resumes_where_it_stopped(
    home: Path, origin: _Origin
) -> None:
    half = len(origin.body) // 2
    origin.cut = [half, 1000]
    install(origin.url)
    assert installed_executable() is not None
    assert len((install_dir() / "noise").read_bytes()) == 256 * 1024
    assert origin.ranges == [None, f"bytes={half}-", f"bytes={half + 1000}-"]


def test_a_download_replaced_midway_is_not_spliced(home: Path, origin: _Origin) -> None:
    origin.cut = [len(origin.body) // 2]
    real_read = carla_install._ResumingStream.read

    def read(stream: carla_install._ResumingStream, amount: int = -1) -> bytes:
        origin.etag = '"nightly-2"'  # a new nightly is published meanwhile
        return real_read(stream, amount)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(carla_install._ResumingStream, "read", read)
        with pytest.raises(RuntimeError, match="changed during the download"):
            install(origin.url)
    assert installed_executable() is None
    assert not install_dir().with_name("carla.partial").exists()


def test_a_dead_link_gives_up(
    home: Path, origin: _Origin, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(carla_install, "_RETRIES", 3)
    origin.cut = [100, *[0] * 10]
    with pytest.raises(RuntimeError, match="gave up after 3"):
        install(origin.url)
    assert len(origin.ranges) == 4
    assert not install_dir().with_name("carla.partial").exists()


def test_a_server_ignoring_if_range_is_not_spliced(home: Path, origin: _Origin) -> None:
    origin.honours_if_range = False
    test_a_download_replaced_midway_is_not_spliced(home, origin)


def test_a_permanent_http_error_is_not_retried(home: Path, origin: _Origin) -> None:
    origin.cut = [100]
    origin.gone = True
    with pytest.raises(urllib.error.HTTPError, match="404"):
        install(origin.url)
    assert len(origin.ranges) == 2
    assert not install_dir().with_name("carla.partial").exists()


def test_an_interrupted_swap_never_passes_for_a_current_install(
    home: Path,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ctrl-C while the old tree is deleted: the next run ends complete (#1567)."""
    install(origin.url)
    real_rmtree = shutil.rmtree

    def interrupted(path: Path, ignore_errors: bool = False) -> None:
        tree = Path(path)
        if (tree / "noise").is_file():  # an install, old or new, being deleted
            (tree / "noise").unlink()  # part way through: launcher and stamp remain
            raise KeyboardInterrupt
        real_rmtree(tree, ignore_errors)

    monkeypatch.setattr(shutil, "rmtree", interrupted)
    with pytest.raises(KeyboardInterrupt):
        install(origin.url, force=True)  # a reinstall, Ctrl-C mid-swap
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    capsys.readouterr()

    install(origin.url)  # the next run
    assert installed_executable() is not None
    assert len((install_dir() / "noise").read_bytes()) == 256 * 1024
    assert (install_dir() / ".download.json").is_file()
    assert sorted(p.name for p in install_dir().parent.iterdir()) == [
        "carla",
        "carla.lock",
    ]


def test_an_interrupt_between_the_renames_leaves_no_install_to_trust(
    home: Path, origin: _Origin, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(origin.url)
    real_rename = Path.rename

    def interrupted(self: Path, target: Path) -> Path:
        if self.name == "carla.partial":
            raise KeyboardInterrupt
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt):
        install(origin.url, force=True)
    monkeypatch.setattr(Path, "rename", real_rename)
    assert installed_executable() is None  # moved aside, not half there

    install(origin.url)
    assert len((install_dir() / "noise").read_bytes()) == 256 * 1024
    assert sorted(p.name for p in install_dir().parent.iterdir()) == [
        "carla",
        "carla.lock",
    ]


def test_a_failed_swap_puts_the_previous_install_back(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "carla.tar.gz"
    install(_carla(archive))
    real_rename = Path.rename

    def failing(self: Path, target: Path) -> Path:
        if self.name == "carla.partial":
            raise OSError("rename failed")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing)
    _carla(archive, b"a later night" * 10)
    with pytest.raises(OSError, match="rename failed"):
        install(archive.as_uri())
    assert (install_dir() / "VERSION").read_bytes() == b"nightly"
    assert (install_dir() / ".download.json").is_file()


def test_concurrent_installs_wait_for_each_other(
    home: Path,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second run waits for the first instead of deleting its download (#1570)."""
    downloading = threading.Event()  # the first run is mid-download
    waiting = threading.Event()  # the second run blocks on the lock
    real_flock = fcntl.flock
    real_read = carla_install._ResumingStream.read
    staged: list[bool] = []

    def flock(fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:  # the blocking wait, after LOCK_NB failed
            waiting.set()
        real_flock(fd, operation)

    def read(stream: carla_install._ResumingStream, amount: int = -1) -> bytes:
        if not downloading.is_set():
            downloading.set()
            assert waiting.wait(10), "the second run never waited"
            time.sleep(0.2)  # time for it to be blocked in flock, not just about to be
        staged.append(install_dir().with_name("carla.partial").is_dir())
        return real_read(stream, amount)

    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(carla_install._ResumingStream, "read", read)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            install(origin.url)
        except BaseException as exc:  # noqa: BLE001 -- reported by the test
            errors.append(exc)

    first = threading.Thread(target=run)
    first.start()
    assert downloading.wait(10)
    second = threading.Thread(target=run)
    second.start()
    first.join(30)
    second.join(30)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert all(staged), "the first run's download was deleted under it"
    out = capsys.readouterr().out
    assert "Waiting for another scenario-setup" in out
    assert "up to date" in out  # the second found the first one's install
    assert origin.ranges == [None, None]  # one download, one check
    assert len((install_dir() / "noise").read_bytes()) == 256 * 1024
    assert sorted(p.name for p in install_dir().parent.iterdir()) == [
        "carla",
        "carla.lock",
    ]


def test_ctrl_c_while_waiting_for_the_lock_stops_the_run(
    home: Path, tmp_path: Path
) -> None:
    url = _carla(tmp_path / "carla.tar.gz")
    lock = install_dir().with_name("carla.lock")
    lock.parent.mkdir(parents=True)

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    try:
        with open(lock, "a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)  # another run's install
            signal.setitimer(signal.ITIMER_REAL, 0.3)
            with pytest.raises(KeyboardInterrupt):
                install(url)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert installed_executable() is None
    assert install(url) == install_dir() / LAUNCHER  # the lock is free again


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_a_read_only_install_is_still_checked(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    parent = install_dir().parent
    parent.chmod(0o555)
    try:
        for lock_left in (True, False):
            if not lock_left:
                parent.chmod(0o755)
                install_dir().with_name("carla.lock").unlink()
                parent.chmod(0o555)
            capsys.readouterr()
            assert install(url) == install_dir() / LAUNCHER
            assert "up to date" in capsys.readouterr().out
    finally:
        parent.chmod(0o755)


@pytest.mark.parametrize("lock_left", [True, False])
def test_an_unwritable_lock_file_does_not_stop_a_check(
    lock_left: bool,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """As the test above, for root too: opening the lock for writing is refused."""
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    lock = install_dir().with_name("carla.lock")
    if not lock_left:
        lock.unlink()
    opened: list[str] = []

    def read_only_open(file: Path, mode: str = "r") -> IO[str]:
        opened.append(mode)
        if mode != "r":
            raise PermissionError(errno.EACCES, "Permission denied", str(file))
        return open(file, mode)  # noqa: SIM115

    monkeypatch.setattr(carla_install, "open", read_only_open, raising=False)
    capsys.readouterr()
    assert install(url) == install_dir() / LAUNCHER
    captured = capsys.readouterr()
    assert "up to date" in captured.out
    assert opened == ["a", "r"]
    assert ("going ahead unlocked" in captured.err) is not lock_left
    assert lock.exists() is lock_left


def _read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lock file can be opened read-only only, as in another user's install."""

    def read_only_open(file: Path, mode: str = "r") -> IO[str]:
        if mode != "r":
            raise PermissionError(errno.EACCES, "Permission denied", str(file))
        return open(file, mode)  # noqa: SIM115

    monkeypatch.setattr(carla_install, "open", read_only_open, raising=False)


def _nfs_flock(monkeypatch: pytest.MonkeyPatch) -> None:
    """``flock`` as an NFS client has it: byte-range locks (``lockf``), whose
    exclusive lock needs a descriptor open for writing (``EBADF`` otherwise)."""
    monkeypatch.setattr(fcntl, "flock", fcntl.lockf)


def test_a_read_only_check_on_nfs_takes_a_shared_lock(
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1577: an exclusive lock on the read-only lock file failed with EBADF."""
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    _read_only(monkeypatch)
    _nfs_flock(monkeypatch)
    capsys.readouterr()
    assert carla_install.main(["--url", url]) == 0
    captured = capsys.readouterr()
    assert "up to date" in captured.out
    assert captured.err == ""  # locked, not "going ahead unlocked"


def test_a_read_only_lock_without_byte_range_locks_goes_ahead_unlocked(
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    _read_only(monkeypatch)

    def flock(fd: int, operation: int) -> None:
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(fcntl, "flock", flock)
    capsys.readouterr()
    assert carla_install.main(["--url", url]) == 0
    captured = capsys.readouterr()
    assert "up to date" in captured.out
    assert "installing unlocked" in captured.err


@pytest.mark.parametrize("nfs", [False, True], ids=["flock", "nfs"])
def test_a_read_only_check_waits_for_a_writer(
    nfs: bool,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    lock = install_dir().with_name("carla.lock")
    # Another process: byte-range locks do not conflict within one.
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys\n"
            f"f = open(sys.argv[1], 'a'); fcntl.{'lockf' if nfs else 'flock'}"
            "(f.fileno(), fcntl.LOCK_EX); print('locked', flush=True)\n"
            "sys.stdin.read()",
            str(lock),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert writer.stdin is not None and writer.stdout is not None
    try:
        assert writer.stdout.readline() == "locked\n"
        _read_only(monkeypatch)
        if nfs:
            _nfs_flock(monkeypatch)
        real_flock = fcntl.flock
        waiting = threading.Event()
        events: list[str] = []

        def flock(fd: int, operation: int) -> None:
            if operation == fcntl.LOCK_SH:  # the blocking wait, after LOCK_NB failed
                waiting.set()
            real_flock(fd, operation)
            events.append(f"locked {operation}")

        monkeypatch.setattr(fcntl, "flock", flock)

        def release() -> None:
            # Released even if the check never waited, so a regression fails
            # (on the events) rather than hanging the test on the writer's lock.
            if waiting.wait(10):
                time.sleep(
                    0.2
                )  # time for the check to be blocked, not just about to be
            events.append("released")
            assert writer.stdin is not None
            writer.stdin.close()

        releaser = threading.Thread(target=release)
        releaser.start()
        capsys.readouterr()
        assert carla_install.main(["--url", url]) == 0
        releaser.join(10)
        assert events == ["released", f"locked {fcntl.LOCK_SH}"]
        out = capsys.readouterr().out
        assert "Waiting for another scenario-setup" in out
        assert "up to date" in out
    finally:
        if writer.stdin is not None and not writer.stdin.closed:
            writer.stdin.close()
        writer.wait(10)


@pytest.mark.parametrize("state", ["out of date", "not installed", "forced"])
def test_a_read_only_run_does_not_install(
    state: str,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = _carla(tmp_path / "carla.tar.gz")
    install(url)
    if state == "out of date":
        url = _carla(tmp_path / "new.tar.gz", b"newer")
    elif state == "not installed":
        shutil.rmtree(install_dir())
    _read_only(monkeypatch)
    _nfs_flock(monkeypatch)

    def no_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("a read-only run downloaded")

    monkeypatch.setattr(carla_install, "_unpack", no_download)
    capsys.readouterr()
    argv = ["--url", url] + (["--force"] if state == "forced" else [])
    assert carla_install.main(argv) == 1
    err = capsys.readouterr().err
    expected = "not installed" if state == "not installed" else "out of date"
    assert f"is {expected}, and this run cannot write there" in err
    if state != "not installed":
        assert (install_dir() / "VERSION").read_bytes() == b"nightly"
