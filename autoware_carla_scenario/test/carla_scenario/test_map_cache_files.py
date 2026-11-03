"""The files a map cache entry keeps: ``meta.json`` and ``tip.json`` (maps.cache).

The module writes them by hand (the library check cannot compile a mixed-value
``dict``), so these hold the bytes to what ``json.dumps(..., indent=2)`` wrote
when they were dicts, and the reader to the files that wrote.  Both are read
without a lock while another thread or process rewrites them, so they are
replaced whole, never truncated and written in place (#1261).
"""

from __future__ import annotations

import inspect
import json
import os
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
from autoware_carla_scenario.maps.cache import GitMapCache, _EntryMeta, _json_object
from autoware_carla_scenario.maps.source import MapSource

_SOURCE = MapSource(repo_url="https://example.invalid/maps.git", ref="main", path="a")
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def cache(tmp_path: Path) -> GitMapCache:
    return GitMapCache(tmp_path / "map_root")


def _entry(cache: GitMapCache) -> Path:
    entry = cache.entry_dir(_SOURCE)
    entry.mkdir(parents=True)
    return entry


def _old_meta_text(**fields: Any) -> str:
    """What the dict-based module wrote to ``meta.json``."""
    return json.dumps(fields, indent=2)


# -- byte for byte what json.dumps wrote --------------------------------------


def test_a_new_entry_writes_the_meta_json_the_dict_module_wrote(
    cache: GitMapCache,
) -> None:
    entry = _entry(cache)
    GitMapCache._write_meta(entry, _SOURCE, _COMMIT)

    written = (entry / "meta.json").read_text(encoding="utf-8")
    fetched_at = json.loads(written)["fetched_at"]
    assert isinstance(fetched_at, float)
    assert written == _old_meta_text(
        repo_url=_SOURCE.repo_url,
        ref="main",
        commit=_COMMIT,
        fetched_at=fetched_at,
        materialised=None,
    )


def test_recording_the_checked_out_directory_writes_what_the_merge_wrote(
    cache: GitMapCache,
) -> None:
    entry = _entry(cache)
    GitMapCache._write_meta(entry, _SOURCE, _COMMIT)
    fetched_at = json.loads((entry / "meta.json").read_text())["fetched_at"]

    GitMapCache._write_entry_meta(
        entry, replace(GitMapCache._read_meta(entry), materialised=("b", "a"))
    )

    assert (entry / "meta.json").read_text(encoding="utf-8") == _old_meta_text(
        repo_url=_SOURCE.repo_url,
        ref="main",
        commit=_COMMIT,
        fetched_at=fetched_at,
        materialised=["a", "b"],
    )


def test_a_meta_json_with_only_the_checked_out_directory_is_what_the_merge_wrote(
    cache: GitMapCache,
) -> None:
    # The entry's meta.json was missing: the old merge started from {}.
    entry = _entry(cache)
    GitMapCache._write_entry_meta(
        entry, replace(GitMapCache._read_meta(entry), materialised=("a",))
    )

    assert (entry / "meta.json").read_text(encoding="utf-8") == _old_meta_text(
        materialised=["a"]
    )


def test_a_tip_json_is_what_the_dict_module_wrote(cache: GitMapCache) -> None:
    cache.remember_tip(_SOURCE, _COMMIT)

    written = cache.tip_file(_SOURCE).read_text(encoding="utf-8")
    resolved_at = json.loads(written)["resolved_at"]
    assert written == json.dumps(
        {
            "repo_url": _SOURCE.repo_url,
            "ref": "main",
            "commit": _COMMIT,
            "resolved_at": resolved_at,
        },
        indent=2,
    )


@pytest.mark.parametrize(
    "value", ["plain", 'quote " and \\ slash', "café ☃", "line\nbreak", ""]
)
def test_strings_are_escaped_as_json_dumps_escapes_them(
    cache: GitMapCache, value: str
) -> None:
    entry = _entry(cache)
    GitMapCache._write_entry_meta(
        entry,
        _EntryMeta(repo_url=value, ref=value, commit=value, materialised=(value,)),
    )

    assert (entry / "meta.json").read_text(encoding="utf-8") == _old_meta_text(
        repo_url=value, ref=value, commit=value, materialised=[value]
    )
    assert GitMapCache._read_meta(entry).materialised == (value,)


def test_an_object_without_fields_is_what_json_dumps_wrote() -> None:
    assert _json_object([]) == json.dumps({}, indent=2)


# -- files the dict module wrote still load -----------------------------------


def test_a_meta_json_the_dict_module_wrote_is_read(cache: GitMapCache) -> None:
    entry = _entry(cache)
    (entry / "repo" / ".git").mkdir(parents=True)
    # Written out by hand, as an older version left it on disk.
    (entry / "meta.json").write_text(
        "{\n"
        '  "repo_url": "https://example.invalid/maps.git",\n'
        '  "ref": "main",\n'
        f'  "commit": "{_COMMIT}",\n'
        '  "fetched_at": 1700000000.5,\n'
        '  "materialised": "a"\n'
        "}",
        encoding="utf-8",
    )

    assert GitMapCache._read_meta(entry) == _EntryMeta(
        repo_url="https://example.invalid/maps.git",
        ref="main",
        commit=_COMMIT,
        fetched_at=1700000000.5,
        # A single directory, as an entry recorded before the checkout grew.
        materialised=("a",),
    )
    peeked = cache.peek(_SOURCE)
    assert peeked is not None
    assert peeked.commit == _COMMIT
    assert peeked.materialised == ("a",)


def test_a_meta_json_that_was_never_checked_out_reads_as_not_materialised(
    cache: GitMapCache,
) -> None:
    entry = _entry(cache)
    (entry / "meta.json").write_text(
        '{\n  "commit": "abc",\n  "materialised": null\n}', encoding="utf-8"
    )

    meta = GitMapCache._read_meta(entry)
    assert meta.commit == "abc"
    assert meta.materialised == ()
    assert meta.repo_url is None


def test_a_tip_json_the_dict_module_wrote_is_read(cache: GitMapCache) -> None:
    path = cache.tip_file(_SOURCE)
    path.parent.mkdir(parents=True)
    path.write_text(
        "{\n"
        '  "repo_url": "https://example.invalid/maps.git",\n'
        '  "ref": "main",\n'
        f'  "commit": "{_COMMIT}",\n'
        '  "resolved_at": 1700000000.25\n'
        "}",
        encoding="utf-8",
    )

    assert cache.last_tip(_SOURCE) == _COMMIT


def test_what_is_written_is_read_back(cache: GitMapCache) -> None:
    entry = _entry(cache)
    GitMapCache._write_meta(entry, _SOURCE, _COMMIT)
    meta = GitMapCache._read_meta(entry)
    assert (meta.repo_url, meta.ref, meta.commit) == (
        _SOURCE.repo_url,
        "main",
        _COMMIT,
    )
    assert meta.materialised == ()
    assert isinstance(meta.fetched_at, float)

    cache.remember_tip(_SOURCE, _COMMIT)
    assert cache.last_tip(_SOURCE) == _COMMIT


# -- corrupt files read as empty ----------------------------------------------

_CORRUPT = [
    "",
    "{bad",
    "[1]",
    '"commit"',
    "5",
    "null",
    "true",
    "[]",
    '{"commit": '.strip(),
]


@pytest.mark.parametrize("text", _CORRUPT)
def test_a_corrupt_meta_json_reads_as_empty(cache: GitMapCache, text: str) -> None:
    entry = _entry(cache)
    (entry / "meta.json").write_text(text, encoding="utf-8")

    assert GitMapCache._read_meta(entry) == _EntryMeta()


def test_a_missing_meta_json_reads_as_empty(cache: GitMapCache) -> None:
    assert GitMapCache._read_meta(_entry(cache)) == _EntryMeta()


@pytest.mark.parametrize("text", _CORRUPT)
def test_a_corrupt_tip_json_reads_as_no_commit(cache: GitMapCache, text: str) -> None:
    path = cache.tip_file(_SOURCE)
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")

    assert cache.last_tip(_SOURCE) == ""


def test_a_tip_json_naming_no_commit_reads_as_no_commit(cache: GitMapCache) -> None:
    path = cache.tip_file(_SOURCE)
    path.parent.mkdir(parents=True)
    path.write_text('{"commit": null}', encoding="utf-8")
    assert cache.last_tip(_SOURCE) == ""

    path.write_text('{"commit": "not-a-commit"}', encoding="utf-8")
    assert cache.last_tip(_SOURCE) == ""


# -- written whole: a reader never sees half a file (#1261) -------------------


def _pause_halfway(
    monkeypatch: pytest.MonkeyPatch, read: Callable[[], Any]
) -> list[Any]:
    """Make every ``Path.write_text`` stop halfway and *read*, as a reader
    racing the write would; returns what each *read* saw."""
    seen: list[Any] = []

    def write_text(
        self: Path,
        data: str,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> int:
        with self.open("w", encoding=encoding, errors=errors, newline=newline) as f:
            f.write(data[: len(data) // 2])
            f.flush()
            seen.append(read())
            f.write(data[len(data) // 2 :])
        return len(data)

    monkeypatch.setattr(Path, "write_text", write_text)
    return seen


def test_a_meta_json_being_rewritten_reads_as_the_old_one(
    cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry(cache)
    GitMapCache._write_meta(entry, _SOURCE, _COMMIT)
    before = GitMapCache._read_meta(entry)

    seen = _pause_halfway(monkeypatch, lambda: GitMapCache._read_meta(entry))
    GitMapCache._write_entry_meta(entry, replace(before, materialised=("a", "b")))

    assert seen == [before]
    assert seen[0].commit == _COMMIT
    assert GitMapCache._read_meta(entry) == replace(before, materialised=("a", "b"))


def test_a_tip_json_being_rewritten_reads_as_the_old_one(
    cache: GitMapCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    newer = "fedcba9876543210fedcba9876543210fedcba98"
    cache.remember_tip(_SOURCE, _COMMIT)

    seen = _pause_halfway(monkeypatch, lambda: cache.last_tip(_SOURCE))
    cache.remember_tip(_SOURCE, newer)

    assert seen == [_COMMIT]
    assert cache.last_tip(_SOURCE) == newer


def test_meta_and_tip_leave_no_scratch_file_behind(cache: GitMapCache) -> None:
    entry = _entry(cache)
    GitMapCache._write_meta(entry, _SOURCE, _COMMIT)
    GitMapCache._write_entry_meta(
        entry, replace(GitMapCache._read_meta(entry), materialised=("a",))
    )
    cache.remember_tip(_SOURCE, _COMMIT)
    cache.remember_tip(_SOURCE, _COMMIT)

    assert [p.name for p in entry.iterdir()] == ["meta.json"]
    assert sorted(p.name for p in entry.parent.iterdir()) == sorted(
        [entry.name, cache.tip_file(_SOURCE).name]
    )
    # Readable by others sharing the cache, as write_text left it, rather than
    # the scratch file's 0o600.
    for written in (entry / "meta.json", cache.tip_file(_SOURCE)):
        assert stat.S_IMODE(written.stat().st_mode) == 0o644


@pytest.mark.parametrize("failing", ["write_text", "replace"])
def test_a_failed_write_keeps_the_old_file_and_no_scratch_file(
    cache: GitMapCache, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    entry = _entry(cache)
    GitMapCache._write_meta(entry, _SOURCE, _COMMIT)
    cache.remember_tip(_SOURCE, _COMMIT)
    meta_before = (entry / "meta.json").read_text(encoding="utf-8")
    tip_before = cache.tip_file(_SOURCE).read_text(encoding="utf-8")

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    if failing == "write_text":
        monkeypatch.setattr(Path, "write_text", broken)
    else:
        monkeypatch.setattr(os, "replace", broken)

    with pytest.raises(OSError, match="disk full"):
        GitMapCache._write_meta(entry, _SOURCE, "f" * 40)
    with pytest.raises(OSError, match="disk full"):
        cache.remember_tip(_SOURCE, "f" * 40)
    monkeypatch.undo()

    assert (entry / "meta.json").read_text(encoding="utf-8") == meta_before
    assert cache.tip_file(_SOURCE).read_text(encoding="utf-8") == tip_before
    assert [p.name for p in entry.iterdir()] == ["meta.json"]
    assert sorted(p.name for p in entry.parent.iterdir()) == sorted(
        [entry.name, cache.tip_file(_SOURCE).name]
    )


# -- include: None and () mean the same ---------------------------------------


def test_checkout_takes_no_include_by_default() -> None:
    assert inspect.signature(GitMapCache.checkout).parameters["include"].default is None


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


@pytest.mark.parametrize("include", [None, (), [], [""]])
def test_no_include_pattern_checks_the_directory_out_without_lfs(
    tmp_path: Path, include: Any
) -> None:
    origin = tmp_path / "origin"
    (origin / "map").mkdir(parents=True)
    (origin / "map" / "lanelet2_map.osm").write_text("<osm/>")
    _git(origin, "init", "--quiet", "--initial-branch", "main", ".")
    _git(origin, "add", "-A")
    _git(origin, "commit", "--quiet", "-m", "map")
    source = MapSource.parse(f"git+file://{origin}@main#map")

    kwargs = {} if include is None else {"include": include}
    cache = GitMapCache(tmp_path / "root")
    repo = cache.checkout(source, **kwargs)

    assert (repo.worktree / "map" / "lanelet2_map.osm").read_text() == "<osm/>"
    peeked = cache.peek(source)
    assert peeked is not None
    assert peeked.materialised == ("map",)
    # Nothing but the entry (no clone staging, no scratch meta.json) is left.
    assert [p.name for p in peeked.root.parent.iterdir()] == [peeked.root.name]
    assert sorted(p.name for p in peeked.root.iterdir()) == ["meta.json", "repo"]
