"""Unit tests for ``scenario-setup``, against small packages shaped like CARLA's."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

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
    assert launcher == home / "bin" / "carla" / LAUNCHER
    assert installed_executable() == launcher
    assert launcher.stat().st_mode & 0o111, "the launcher stays executable"
    assert (home / "bin" / "carla" / "VERSION").read_bytes() == b"nightly"


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
