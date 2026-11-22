"""The automatic lock directory of the port slots is private (#2325).

``/dev/shm/acs-slurm-ports-<uid>`` (and the ``/tmp`` one it falls back to) is
used only if it is a real directory of this user's that no one else can write
to; ``launcher.port_lock_dir`` is taken as configured.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from autoware_carla_scenario.slurm import ports


def _name() -> str:
    return f"acs-slurm-ports-{os.getuid()}"


@pytest.fixture
def parents(tmp_path: Path) -> tuple[Path, Path]:
    shm, tmp = tmp_path / "shm", tmp_path / "tmp"
    shm.mkdir()
    tmp.mkdir()
    return shm, tmp


def test_a_private_directory_is_made_and_taken(parents: tuple[Path, Path]) -> None:
    shm, tmp = parents
    path = ports.default_lock_dir(shm, tmp)
    assert path == shm / _name()
    info = os.lstat(path)
    assert stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
    # ... and taken again as it is.
    assert ports.default_lock_dir(shm, tmp) == path
    assert not (tmp / _name()).exists()


def test_a_group_readable_one_of_the_users_is_taken(
    parents: tuple[Path, Path],
) -> None:
    shm, tmp = parents
    (shm / _name()).mkdir(mode=0o755)
    (shm / _name()).chmod(0o755)
    assert ports.default_lock_dir(shm, tmp) == shm / _name()


def _squat(path: Path, how: str, elsewhere: Path) -> None:
    if how == "symlink":
        elsewhere.mkdir(mode=0o700)
        path.symlink_to(elsewhere)
    elif how == "file":
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir()
        path.chmod(0o777 if how == "world-writable" else 0o770)


@pytest.mark.parametrize("how", ["symlink", "file", "world-writable", "group-writable"])
def test_a_squatted_directory_is_passed_over_for_tmp(
    parents: tuple[Path, Path],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    shm, tmp = parents
    _squat(shm / _name(), how, tmp_path / "theirs")
    with caplog.at_level(logging.WARNING, logger=ports.__name__):
        path = ports.default_lock_dir(shm, tmp)
    assert path == tmp / _name()
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o700
    assert f"Not taking the port slots' locks in {shm / _name()}" in caplog.text
    if how != "file":
        # Left as it is: nothing is changed in another user's directory.
        assert os.lstat(shm / _name()).st_mode != os.lstat(path).st_mode


def test_a_directory_another_user_owns_is_passed_over(
    parents: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shm, tmp = parents
    theirs = shm / _name()
    theirs.mkdir(mode=0o700)
    real_lstat = os.lstat

    def lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        info = real_lstat(path, *args, **kwargs)  # type: ignore[arg-type]
        if Path(str(path)) == theirs:
            fields = list(info)
            fields[stat.ST_UID] = os.getuid() + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(os, "lstat", lstat)
    with caplog.at_level(logging.WARNING, logger=ports.__name__):
        assert ports.default_lock_dir(shm, tmp) == tmp / _name()
    assert f"is owned by uid {os.getuid() + 1}, not {os.getuid()}" in caplog.text


def test_the_fallback_is_checked_too(
    parents: tuple[Path, Path], tmp_path: Path
) -> None:
    shm, tmp = parents
    _squat(shm / _name(), "world-writable", tmp_path / "unused")
    _squat(tmp / _name(), "symlink", tmp_path / "theirs")
    with pytest.raises(RuntimeError) as raised:
        ports.default_lock_dir(shm, tmp)
    message = str(raised.value)
    assert str(shm / _name()) in message and "writable by others" in message
    assert str(tmp / _name()) in message and "not a directory" in message
    assert "launcher.port_lock_dir" in message


def test_without_shm_the_tmp_one_is_taken_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=ports.__name__):
        path = ports.default_lock_dir(tmp_path / "missing", tmp_path)
    assert path == tmp_path / _name()
    assert "Not taking" not in caplog.text  # no /dev/shm: nothing to warn of


def test_a_configured_shared_directory_is_taken_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A group's own (2770, docs/slurm.md "Ports"): not checked.
    shared = tmp_path / "group-locks"
    shared.mkdir()
    shared.chmod(0o2770)

    def unexpected(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("a configured directory is not the default")

    monkeypatch.setattr(ports, "default_lock_dir", unexpected)
    domains = ports.RosDomains(shared)
    try:
        domain = domains.take(7)
    finally:
        domains.release()
    assert (shared / f"autoware_carla_scenario-ros-domain-{domain}.lock").exists()


def test_a_symlinked_lock_file_is_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "autoware_carla_scenario-ros-domain-7.lock"
    link.symlink_to(target)
    assert ports._try_lock(link) is None
    # A regular one is taken.
    stream = ports._try_lock(tmp_path / "plain.lock")
    assert stream is not None
    stream.close()
