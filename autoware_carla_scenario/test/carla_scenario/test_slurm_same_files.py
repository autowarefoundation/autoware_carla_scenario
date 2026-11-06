"""slurm.same_files' SameFiles: whether a config string that differs on the
node names the same thing as where the case was submitted (#1720)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_carla_scenario.slurm import worker
from autoware_carla_scenario.slurm.same_files import SameFiles
from autoware_carla_scenario.slurm.scripts import file_sha256


def test_worker_uses_the_moved_class() -> None:
    assert worker.SameFiles is SameFiles


def test_no_submit_cwd_has_no_prefixes() -> None:
    assert SameFiles({}, None).prefixes == []
    assert SameFiles({}, "").prefixes == []


def test_prefixes_are_the_job_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", "/logical/here")
    same = SameFiles({}, "/submit/dir/")
    assert same.prefixes == sorted(
        [("/submit/dir/", "/logical/here"), ("/submit/dir/", str(tmp_path))],
        key=lambda pair: pair[1],
    )


def test_pwd_unset_or_empty_falls_back_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PWD", raising=False)
    assert SameFiles({}, "/submit").prefixes == [("/submit/", str(tmp_path))]
    monkeypatch.setenv("PWD", "")
    assert SameFiles({}, "/submit").prefixes == [("/submit/", str(tmp_path))]


def test_the_submitting_directory_itself_is_no_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", str(tmp_path))
    assert SameFiles({}, str(tmp_path)).prefixes == []


def test_a_moved_submitting_directory_is_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", str(tmp_path))
    same = SameFiles({}, "/submit")
    assert same.why_not("a", "/submit/x/y.osm", str(tmp_path / "x/y.osm")) is None
    assert same.why_not("a", "/submit/x/y.osm", "/elsewhere/x/y.osm") == ""
    assert same.why_not("a", "plain", "plain") is None


def test_a_recorded_file_is_compared_by_content(tmp_path: Path) -> None:
    there = tmp_path / "there.osm"
    there.write_text("map")
    here = tmp_path / "here.osm"
    here.write_text("map")
    same = SameFiles({"map.path": file_sha256(there)}, None)
    assert same.why_not("map.path", str(there), str(here)) is None
    here.write_text("changed")
    # the digest of a path is read once
    assert same.why_not("map.path", str(there), str(here)) is None
    other = tmp_path / "other.osm"
    other.write_text("changed")
    assert same.why_not("map.path", str(there), str(other)) == (
        f"{there} there, {other} here: content differs"
    )
    missing = tmp_path / "missing.osm"
    assert same.why_not("map.path", str(there), str(missing)) == (
        f"{there} there, {missing} here: missing on the node"
    )
