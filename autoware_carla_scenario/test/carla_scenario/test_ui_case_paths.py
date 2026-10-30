"""ui.case_paths, split out of ui.cases so the library check compiles it
(#741): the viewer's path guard still refuses every way out of its base, and
ui.cases still exports every name it did, as the same objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoware_carla_scenario.ui import case_paths, cases

#: What ui.cases held before the split that now lives in ui.case_paths.
_MOVED = (
    "SESSION_TYPES",
    "_COMPONENT",
    "_TRAJECTORY",
    "_RESULT",
    "_STEM_NUMBER",
    "CaseFiles",
    "_safe",
    "contained",
    "session_dir",
    "job_dir",
    "stem_order",
    "_stems_in",
)


@pytest.mark.parametrize("name", _MOVED)
def test_cases_re_exports_the_moved_name(name: str) -> None:
    assert getattr(cases, name) is getattr(case_paths, name)


def test_cases_still_exports_every_name() -> None:
    assert cases.__all__ == [
        "SESSION_TYPES",
        "CaseFiles",
        "batch_case_files",
        "batch_entries",
        "case_files",
        "contained",
        "job_dir",
        "match_batch_stems",
        "session_dir",
        "stem_order",
    ]
    for name in cases.__all__:
        assert hasattr(cases, name), name


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """``base/root`` with a directory ``a``, a secret beside the root, and
    symlinks in the root to the secret and to ``a``."""
    root = tmp_path / "base" / "root"
    (root / "a" / "b").mkdir(parents=True)
    (tmp_path / "base" / "secret").mkdir()
    (tmp_path / "base" / "secret" / "key.txt").write_text("secret")
    (root / "out").symlink_to(tmp_path / "base" / "secret", target_is_directory=True)
    (root / "in").symlink_to(root / "a", target_is_directory=True)
    return root


@pytest.mark.parametrize(
    "parts",
    [
        ("..",),
        ("..", "secret"),
        ("..", "..", "..", "etc", "passwd"),
        ("a", "..", ".."),
        ("a", "b", "..", "..", "..", "secret"),
        ("a/../../secret",),
        ("../secret/key.txt",),
        ("/etc",),
        ("/etc/passwd",),
        ("a", "/etc"),
        ("out",),
        ("out", "key.txt"),
        ("in", "..", "..", "secret"),
        ("in/../../secret",),
        # A sibling named like the root but for its case.
        ("..", "Root"),
        ("..", "root", "..", "secret"),
    ],
)
def test_contained_refuses_a_way_out_of_the_root(
    root: Path, parts: tuple[str, ...]
) -> None:
    assert case_paths.contained(root, *parts) is None
    assert cases.contained(root, *parts) is None


@pytest.mark.parametrize(
    ("parts", "joined"),
    [
        ((), ""),
        ((".",), "."),
        (("a",), "a"),
        (("a", "b"), "a/b"),
        (("a", "..", "a"), "a/../a"),
        (("missing", "x.json"), "missing/x.json"),
        (("in",), "in"),
        (("in", "b"), "in/b"),
        (("a/",), "a"),
        (("..", "root", "a"), "../root/a"),
        # Windows separators are a name's characters on POSIX, not a way up.
        (("a\\..\\..\\secret",), "a\\..\\..\\secret"),
        (("x" * 300,), "x" * 300),
    ],
)
def test_contained_keeps_a_path_under_the_root(
    root: Path, parts: tuple[str, ...], joined: str
) -> None:
    # The path as joined, not resolved: the symlink stays the route's.
    expected = root / joined if joined else root
    assert case_paths.contained(root, *parts) == expected


def test_contained_refuses_a_symlink_loop(root: Path) -> None:
    (root / "loop").symlink_to(root / "loop")
    # A loop raises (RuntimeError before Python 3.13, OSError since) or
    # resolves to itself under the root, depending on the Python version;
    # either way it does not leave the root.
    found = case_paths.contained(root, "loop")
    assert found is None or found == root / "loop"


@pytest.mark.parametrize(
    ("session_type", "date", "time"),
    [
        ("multirun", "..", "00-00-00"),
        ("multirun", "2026-01-01", ".."),
        ("batch", ".", "00-00-00"),
        ("single", "2026-01-01/..", "x"),
        ("single", "a/b", "00-00-00"),
        ("batch", "a\\b", "00-00-00"),
        ("batch", "", "00-00-00"),
        ("batch", "2026-01-01", "x" * 256),
        ("batch", "2026-01-01", "00\x0000"),
        ("batch", "2026-01-01", "00 00"),
        ("outputs", "2026-01-01", "00-00-00"),
        ("", "2026-01-01", "00-00-00"),
        # Escapes a URL or another script spells ".." with: no such name.
        ("batch", "%2e%2e", "00-00-00"),
        ("batch", "..%2f", "00-00-00"),
        ("batch", "\uff0e\uff0e", "00-00-00"),
        ("batch", "\u2025", "00-00-00"),
        ("batch", "..\u0338", "00-00-00"),
        ("batch", "..\r", "00-00-00"),
        ("batch", "2026-01-01/", "00-00-00"),
        ("batch", "C:\\x", "00-00-00"),
        ("batch", "x" * 256, "00-00-00"),
        # The type is matched as written.
        ("Batch", "2026-01-01", "00-00-00"),
        ("MULTIRUN", "2026-01-01", "00-00-00"),
        (" multirun", "2026-01-01", "00-00-00"),
        ("multirun\n", "2026-01-01", "00-00-00"),
    ],
)
def test_session_dir_refuses_names_that_make_no_session(
    tmp_path: Path, session_type: str, date: str, time: str
) -> None:
    assert case_paths.session_dir(tmp_path, session_type, date, time) is None


@pytest.mark.parametrize("name", ["..\n", ".\n", "...", "x" * 255])
def test_session_dir_keeps_an_odd_name_under_the_base(
    tmp_path: Path, name: str
) -> None:
    # ``$`` matches before a final newline, so "..\n" passes the name test,
    # but it names a directory "..\n", not the parent: the guard still keeps
    # the session under ``outputs/``.
    found = case_paths.session_dir(tmp_path, "batch", name, "00-00-00")
    assert found is None or found.resolve().is_relative_to(
        (tmp_path / "outputs").resolve()
    )
    job = case_paths.job_dir(tmp_path, name, "00-00-00", 10**30)
    assert job is None or job.resolve().is_relative_to(
        (tmp_path / "multirun").resolve()
    )


def test_session_dir_refuses_a_session_symlinked_out_of_the_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    (base / "outputs").mkdir(parents=True)
    (tmp_path / "elsewhere" / "00-00-00").mkdir(parents=True)
    (base / "outputs" / "2026-01-01").symlink_to(
        tmp_path / "elsewhere", target_is_directory=True
    )
    assert case_paths.session_dir(base, "batch", "2026-01-01", "00-00-00") is None


def test_session_and_job_dirs_under_the_base(tmp_path: Path) -> None:
    assert case_paths.session_dir(tmp_path, "multirun", "2026-01-01", "00-00-00") == (
        tmp_path / "multirun" / "2026-01-01" / "00-00-00"
    )
    assert case_paths.session_dir(tmp_path, "single", "2026-01-01", "12:00:00") == (
        tmp_path / "outputs" / "2026-01-01" / "12:00:00"
    )
    assert case_paths.job_dir(tmp_path, "2026-01-01", "00-00-00", 3) == (
        tmp_path / "multirun" / "2026-01-01" / "00-00-00" / "3"
    )
    assert case_paths.job_dir(tmp_path, "2026-01-01", "00-00-00", -1) is None
    assert case_paths.job_dir(tmp_path, "..", "00-00-00", 0) is None


def test_the_stems_of_a_directory_in_the_runner_s_order(tmp_path: Path) -> None:
    for name in (
        "Cut-10_result.json",
        "Cut-2.trajectory.jsonl",
        "Cut_result.json",
        "Cut.trajectory.jsonl",
        "Cut-1_coverage.json",
        "_result.json",
        ".trajectory.jsonl",
        "Other_result.json",
    ):
        (tmp_path / name).write_text("{}")
    assert case_paths._stems_in(tmp_path) == ["Cut", "Cut-2", "Cut-10", "Other"]
    assert case_paths._stems_in(tmp_path / "missing") == []
    files = case_paths.CaseFiles(tmp_path, "Cut-2")
    assert files.trajectory == tmp_path / "Cut-2.trajectory.jsonl"
    assert files.result == tmp_path / "Cut-2_result.json"
    assert files.coverage == tmp_path / "Cut-2_coverage.json"
    assert files.odd_archive == tmp_path / "Cut-2_odd.tar.gz"
    assert files.environment == tmp_path / "Cut-2_environment.json"
    assert files.recording == tmp_path / "Cut-2.rrd"
