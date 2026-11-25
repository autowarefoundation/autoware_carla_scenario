"""atomic_file: a file replaced in one rename, text, bytes or chunks."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterator

import pytest

from autoware_carla_scenario import atomic_file
from autoware_carla_scenario.atomic_file import (
    write_bytes_atomically,
    write_chunks_atomically,
    write_text_atomically,
)


def _kill_the_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    def replace(src: str, dst: str) -> None:
        raise OSError("killed before the rename")

    monkeypatch.setattr(atomic_file.os, "replace", replace)


def test_bytes_replace_the_file_whole_with_write_texts_mode(tmp_path: Path) -> None:
    path = tmp_path / "x.pb"
    path.write_bytes(b"old")
    write_bytes_atomically(path, b"\x00new\xff")
    assert path.read_bytes() == b"\x00new\xff"
    assert path.stat().st_mode & 0o777 == 0o644
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.pb"]


def test_chunks_are_written_one_after_the_other_from_a_generator(
    tmp_path: Path,
) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"head"
        yield b"-one"
        yield b"-two"

    path = tmp_path / "shard"
    write_chunks_atomically(path, chunks())
    assert path.read_bytes() == b"head-one-two"


@pytest.mark.parametrize(
    "write",
    [
        lambda path: write_bytes_atomically(path, b"new"),
        lambda path: write_chunks_atomically(path, [b"n", b"ew"]),
        lambda path: write_text_atomically(path, "new"),
    ],
    ids=["bytes", "chunks", "text"],
)
def test_a_write_killed_before_the_rename_keeps_the_old_file_and_no_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: object
) -> None:
    path = tmp_path / "x.pb"
    path.write_bytes(b"old")
    _kill_the_rename(monkeypatch)
    with pytest.raises(OSError, match="killed"):
        write(path)  # type: ignore[operator]
    assert path.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.pb"]


def test_a_chunk_that_fails_leaves_no_file(tmp_path: Path) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"half"
        raise RuntimeError("the producer failed")

    path = tmp_path / "shard"
    with pytest.raises(RuntimeError, match="producer"):
        write_chunks_atomically(path, chunks())
    assert list(tmp_path.iterdir()) == []


def test_the_scratch_file_does_not_match_the_files_own_glob(tmp_path: Path) -> None:
    """The scratch name starts with a dot: a glob for ``name-*`` (pack's check
    for a submission already there) does not find one left by a crash."""
    seen: list[str] = []
    real = atomic_file.os.replace

    def replace(src: str, dst: str) -> None:
        seen.extend(p.name for p in tmp_path.glob("shard-*"))
        real(src, dst)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(atomic_file.os, "replace", replace)
        write_bytes_atomically(tmp_path / "shard-0", b"x")
    assert seen == []


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize(
    "write",
    [
        lambda path: write_bytes_atomically(path, b"new"),
        lambda path: write_chunks_atomically(path, [b"n", b"ew"]),
        lambda path: write_text_atomically(path, "new"),
    ],
    ids=["bytes", "chunks", "text"],
)
def test_a_symlink_is_written_through_and_kept(tmp_path: Path, write: object) -> None:
    """The link stays a link and its target gets the new content (#2507):
    the scratch file sits beside the target, not beside the link."""
    real = tmp_path / "real" / "document.yaml"
    real.parent.mkdir()
    real.write_text("old", encoding="utf-8")
    link = tmp_path / "document.yaml"
    link.symlink_to(Path("real") / "document.yaml")
    write(link)  # type: ignore[operator]
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new"
    assert sorted(p.name for p in real.parent.iterdir()) == ["document.yaml"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["document.yaml", "real"]


def test_a_dangling_symlink_creates_its_target(tmp_path: Path) -> None:
    link = tmp_path / "document.yaml"
    link.symlink_to("target.yaml")
    write_text_atomically(link, "new")
    assert link.is_symlink()
    assert (tmp_path / "target.yaml").read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o755])
def test_an_existing_file_keeps_its_mode(tmp_path: Path, mode: int) -> None:
    """A file the user made 0o600 does not become 0o644 (#2507)."""
    path = tmp_path / "secret.json"
    path.write_text("x", encoding="utf-8")
    os.chmod(path, mode)
    write_text_atomically(path, "y")
    assert path.read_text(encoding="utf-8") == "y"
    assert _mode(path) == mode


def test_a_new_file_gets_write_texts_mode_under_the_umask(tmp_path: Path) -> None:
    """``0o666`` less the umask, as ``write_text`` makes it: 0o600 under 077,
    not 0o644 (#2507); and the umask is left as it was."""
    old = os.umask(0o077)
    try:
        write_text_atomically(tmp_path / "new.json", "z")
        write_bytes_atomically(tmp_path / "new.pb", b"z")
        (tmp_path / "plain.txt").write_text("w", encoding="utf-8")
        assert os.umask(0o077) == 0o077
    finally:
        os.umask(old)
    assert _mode(tmp_path / "new.json") == 0o600
    assert _mode(tmp_path / "new.pb") == 0o600
    assert _mode(tmp_path / "plain.txt") == 0o600


def test_a_new_file_under_umask_002_is_group_writable(tmp_path: Path) -> None:
    old = os.umask(0o002)
    try:
        write_text_atomically(tmp_path / "shared.json", "z")
    finally:
        os.umask(old)
    assert _mode(tmp_path / "shared.json") == 0o664
