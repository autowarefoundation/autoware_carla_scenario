"""Unit tests for ``scenario-setup``, against small packages shaped like CARLA's."""

from __future__ import annotations

import io
import json
import os
import tarfile
import threading
from collections.abc import Iterator
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

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
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                wanted = self.headers.get("Range")
                origin.ranges.append(wanted)
                start = 0
                if wanted and self.headers.get("If-Range") in (None, origin.etag):
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
