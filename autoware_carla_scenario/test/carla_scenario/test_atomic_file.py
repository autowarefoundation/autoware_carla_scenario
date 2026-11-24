"""atomic_file: a file replaced in one rename, text, bytes or chunks."""

from __future__ import annotations

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
