"""The viewer's file helpers (``ui.recordings``): where a recording is kept
in the cache, whether one is fresh, putting a recording in place, and
swapping a copy of the web viewer into place.

The ``_old_*`` oracles below are ``ui/recordings.py``'s
``_cached_recording()``, ``_fresh()``, ``_publish()`` and
``_swap_into_place()`` as they were before they moved into the library check
(#2124), copied verbatim but for their names (``_old_`` in front).  What the
viewer has now must do the same: return the same, raise the same (type,
arguments, cause and context suppression), make the same ``os.replace``,
``os.link`` and ``os.path.lexists`` calls in the same order, and leave the
same files, whether a call succeeds or fails part-way.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.ui import recording_files, recordings
from autoware_carla_scenario.ui.web_viewer_paths import _cache_root

# -- the oracles: ui/recordings.py as it was -----------------------------------


def _old_cached_recording(trajectory: Path) -> Path:
    """Where a recording is kept when the log's directory cannot be written."""
    digest = hashlib.sha256(str(trajectory.resolve()).encode()).hexdigest()[:24]
    return _cache_root() / "recordings" / f"{digest}.rrd"


def _old_fresh(recording: Path, trajectory: Path) -> bool:
    try:
        return recording.stat().st_mtime >= trajectory.stat().st_mtime
    except OSError:
        return False


def _old_publish(tmp: Path, out: Path, replace: bool) -> None:
    """Rename *tmp* to *out*; without *replace*, only while there is no *out*
    (``FileExistsError`` otherwise), so a file written there meanwhile -- by a
    run converting its own log -- is kept."""
    if replace:
        os.replace(tmp, out)
        return
    try:
        os.link(tmp, out)  # fails, atomically, when out exists
    except FileExistsError:
        raise
    except OSError:  # a file system without hard links
        if os.path.lexists(out):
            raise FileExistsError(str(out)) from None
        os.replace(tmp, out)


def _old_swap_into_place(staging: Path, target: Path) -> None:
    """Move the directory *staging* to *target*, replacing a copy already
    there; the old copy is moved aside first, and back if the move fails."""
    old: Path | None = None
    if target.exists():
        old = Path(tempfile.mkdtemp(prefix=f".{target.name}.old.", dir=target.parent))
        try:
            os.replace(target, old)
        except OSError:
            old.rmdir()  # still the empty directory mkdtemp made
            raise
    try:
        os.replace(staging, target)
    except OSError:
        if old is not None:
            os.replace(old, target)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


# -- helpers --------------------------------------------------------------------

#: mkdtemp's random suffix, which differs between two runs.
_RANDOM = re.compile(r"(\.old\.)[A-Za-z0-9_]{8}")


def _norm(text: str, base: Path) -> str:
    return _RANDOM.sub(r"\1XXXXXXXX", text.replace(str(base), "<base>"))


def _tree(root: Path) -> dict[str, tuple[str, bytes, int]]:
    out: dict[str, tuple[str, bytes, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            mode = path.lstat().st_mode
            key = _norm(str(path.relative_to(root)), root)
            if stat.S_ISLNK(mode):
                out[key] = ("link", os.readlink(path).encode(), 0)
            elif stat.S_ISDIR(mode):
                out[key] = ("dir", b"", stat.S_IMODE(mode))
            else:
                out[key] = ("file", path.read_bytes(), stat.S_IMODE(mode))
    return out


def _outcome(base: Path, call: Callable[[], Any]) -> tuple[Any, ...]:
    try:
        result = call()
    except BaseException as exc:  # noqa: BLE001 -- compared, old and new
        return (
            "raised",
            type(exc),
            tuple(_norm(str(a), base) for a in exc.args),
            type(exc.__cause__),
            exc.__suppress_context__,
        )
    if isinstance(result, Path):
        return ("returned", _norm(str(result), base))
    return ("returned", result)


class _Calls:
    """os.replace, os.link and os.path.lexists, recorded, any made to fail."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.made: list[tuple[str, ...]] = []
        self.base = Path("/")
        #: (name, call number from 1) -> the error that call raises.
        self.fail: dict[tuple[str, int], BaseException] = {}
        self._counts: dict[str, int] = {}
        for owner, name in ((os, "replace"), (os, "link"), (os.path, "lexists")):
            real = getattr(owner, name)
            monkeypatch.setattr(owner, name, self._spy(name, real))

    def reset(self, base: Path) -> None:
        self.made.clear()
        self._counts.clear()
        self.base = base

    def _spy(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kw: Any) -> Any:
            self._counts[name] = self._counts.get(name, 0) + 1
            self.made.append(
                (name, *(_norm(str(a), self.base) for a in args), repr(sorted(kw)))
            )
            error = self.fail.get((name, self._counts[name]))
            if error is not None:
                raise error
            return real(*args, **kw)

        return spy


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Calls]:
    yield _Calls(monkeypatch)


def _both(
    tmp_path: Path,
    calls: _Calls,
    prepare: Callable[[Path], None],
    old: Callable[[Path], Any],
    new: Callable[[Path], Any],
) -> None:
    results = []
    for name, run in (("old", old), ("new", new)):
        base = tmp_path / name
        base.mkdir(parents=True)
        prepare(base)
        calls.reset(base)
        outcome = _outcome(base, partial(run, base))
        results.append((outcome, list(calls.made), _tree(base)))
    assert results[0] == results[1]


# -- _cached_recording() ---------------------------------------------------------


@pytest.mark.parametrize(
    "trajectory",
    [
        "run.trajectory.jsonl",
        "a/b/../run.trajectory.jsonl",
        "missing/x.trajectory.jsonl",
        "é中.trajectory.jsonl",
        "link.trajectory.jsonl",
        "loop.trajectory.jsonl",
        "",
    ],
)
@pytest.mark.parametrize(
    "env",
    [
        {"VIEWER_CACHE_DIR": "~/cache"},
        {"VIEWER_CACHE_DIR": "", "XDG_CACHE_HOME": "/xdg"},
        {"XDG_CACHE_HOME": ""},
        {},
    ],
)
def test_the_cached_recording_is_where_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trajectory: str,
    env: dict[str, str],
) -> None:
    for name in ("VIEWER_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "run.trajectory.jsonl").write_text("{}\n")
    (tmp_path / "link.trajectory.jsonl").symlink_to("run.trajectory.jsonl")
    (tmp_path / "loop.trajectory.jsonl").symlink_to("loop.trajectory.jsonl")
    monkeypatch.chdir(tmp_path)
    for path in (Path(trajectory), tmp_path / trajectory):
        assert _outcome(tmp_path, partial(recordings._cached_recording, path)) == (
            _outcome(tmp_path, partial(_old_cached_recording, path))
        )


def test_a_patch_of_recordings_cache_root_reaches_the_cached_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recordings, "_cache_root", lambda: tmp_path / "patched")
    trajectory = tmp_path / "run.trajectory.jsonl"
    cached = recordings._cached_recording(trajectory)
    assert cached.parent == tmp_path / "patched" / "recordings"
    assert cached.name == _old_cached_recording(trajectory).name


# -- _fresh() --------------------------------------------------------------------

#: The mtimes (ns) of the recording and of the log; None: no such file.
_TIMES = [
    (None, None),
    (None, 10**18),
    (10**18, None),
    (10**18, 10**18),
    (10**18 + 1, 10**18),
    (10**18, 10**18 + 1),
    (0, 0),
    (10**18 + 999, 10**18),  # the same float second, apart in ns
    (10**18, 10**18 + 999),
]


@pytest.mark.parametrize(("recording_ns", "trajectory_ns"), _TIMES)
@pytest.mark.parametrize("kind", ["file", "dir", "symlink", "dangling"])
def test_freshness_is_judged_as_it_was(
    tmp_path: Path,
    calls: _Calls,
    recording_ns: int | None,
    trajectory_ns: int | None,
    kind: str,
) -> None:
    def prepare(base: Path) -> None:
        for name, ns in (("x.rrd", recording_ns), ("x.jsonl", trajectory_ns)):
            if ns is None:
                continue
            path = base / name
            if kind == "dir" and name == "x.rrd":
                path.mkdir()
            elif kind in ("symlink", "dangling") and name == "x.rrd":
                (base / "target.rrd").write_bytes(b"r")
                os.utime(base / "target.rrd", ns=(ns, ns))
                path.symlink_to("target.rrd" if kind == "symlink" else "none.rrd")
                continue
            else:
                path.write_bytes(b"x")
            os.utime(path, ns=(ns, ns))

    _both(
        tmp_path,
        calls,
        prepare,
        lambda base: _old_fresh(base / "x.rrd", base / "x.jsonl"),
        lambda base: recordings._fresh(base / "x.rrd", base / "x.jsonl"),
    )


@pytest.mark.parametrize(
    "error", [OSError(errno.EIO, "io"), PermissionError(13, "no"), KeyboardInterrupt()]
)
@pytest.mark.parametrize("which", ["x.rrd", "x.jsonl"])
def test_a_failing_stat_is_judged_as_it_was(
    tmp_path: Path,
    calls: _Calls,
    monkeypatch: pytest.MonkeyPatch,
    error: Any,
    which: str,
) -> None:
    real = Path.stat
    active = [False]

    def stat_(self: Path, **kw: Any) -> Any:
        if active[0] and self.name == which:
            raise error
        return real(self, **kw)

    monkeypatch.setattr(Path, "stat", stat_)

    def prepare(base: Path) -> None:
        (base / "x.rrd").write_bytes(b"r")
        (base / "x.jsonl").write_bytes(b"j")

    def failing(fresh: Callable[[Path, Path], bool]) -> Callable[[Path], bool]:
        def run(base: Path) -> bool:
            active[0] = True
            try:
                return fresh(base / "x.rrd", base / "x.jsonl")
            finally:
                active[0] = False

        return run

    _both(tmp_path, calls, prepare, failing(_old_fresh), failing(recordings._fresh))


# -- _publish() ------------------------------------------------------------------


def _publish_setup(out_kind: str, tmp_there: bool) -> Callable[[Path], None]:
    def prepare(base: Path) -> None:
        if tmp_there:
            (base / "t.tmp").write_bytes(b"new recording")
            (base / "t.tmp").chmod(0o640)
        out = base / "out.rrd"
        if out_kind == "file":
            out.write_bytes(b"old recording")
        elif out_kind == "dir":
            out.mkdir()
        elif out_kind == "full-dir":
            out.mkdir()
            (out / "f").write_bytes(b"f")
        elif out_kind == "dangling":
            out.symlink_to("nowhere")
        elif out_kind == "symlink":
            (base / "pointed").write_bytes(b"p")
            out.symlink_to("pointed")

    return prepare


_LINK_ERRORS: list[BaseException | None] = [
    None,
    OSError(errno.EPERM, "no hard links"),
    OSError(errno.EXDEV, "cross-device"),
    FileExistsError(errno.EEXIST, "exists"),
    PermissionError(errno.EACCES, "denied"),
    KeyboardInterrupt(),
]


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "out_kind", ["none", "file", "dir", "full-dir", "dangling", "symlink"]
)
@pytest.mark.parametrize("tmp_there", [True, False])
@pytest.mark.parametrize("link_error", _LINK_ERRORS)
@pytest.mark.parametrize(
    "replace_error", [None, OSError(errno.ENOSPC, "no space"), KeyboardInterrupt()]
)
def test_a_recording_is_published_as_it_was(
    tmp_path: Path,
    calls: _Calls,
    replace: bool,
    out_kind: str,
    tmp_there: bool,
    link_error: BaseException | None,
    replace_error: BaseException | None,
) -> None:
    if link_error is not None:
        calls.fail[("link", 1)] = link_error
    if replace_error is not None:
        calls.fail[("replace", 1)] = replace_error
    _both(
        tmp_path,
        calls,
        _publish_setup(out_kind, tmp_there),
        lambda base: _old_publish(base / "t.tmp", base / "out.rrd", replace),
        lambda base: recordings._publish(base / "t.tmp", base / "out.rrd", replace),
    )


def test_a_lexists_failure_is_raised_as_it_was(tmp_path: Path, calls: _Calls) -> None:
    calls.fail[("link", 1)] = OSError(errno.EPERM, "no hard links")
    calls.fail[("lexists", 1)] = OSError(errno.EIO, "io")
    _both(
        tmp_path,
        calls,
        _publish_setup("none", True),
        lambda base: _old_publish(base / "t.tmp", base / "out.rrd", False),
        lambda base: recordings._publish(base / "t.tmp", base / "out.rrd", False),
    )


# -- _swap_into_place() ----------------------------------------------------------


def _swap_setup(target_kind: str, staging_there: bool) -> Callable[[Path], None]:
    def prepare(base: Path) -> None:
        cache = base / "cache"
        cache.mkdir()
        if staging_there:
            staging = cache / ".0.38.1.staging"
            staging.mkdir()
            staging.chmod(0o755)
            (staging / "index.js").write_bytes(b"new")
        target = cache / "0.38.1"
        if target_kind == "dir":
            target.mkdir()
            (target / "index.js").write_bytes(b"old")
            (target / "sub").mkdir()
            (target / "sub" / "x").write_bytes(b"x")
        elif target_kind == "empty-dir":
            target.mkdir()
        elif target_kind == "file":
            target.write_bytes(b"a file")
        elif target_kind == "dangling":
            target.symlink_to("nowhere")

    return prepare


@pytest.mark.parametrize(
    "target_kind", ["none", "dir", "empty-dir", "file", "dangling"]
)
@pytest.mark.parametrize("staging_there", [True, False])
@pytest.mark.parametrize(
    "failing",
    [
        {},
        {("replace", 1): OSError(errno.ENOSPC, "no room")},
        {("replace", 2): OSError(errno.ENOSPC, "no room")},
        {
            ("replace", 2): OSError(errno.ENOSPC, "no room"),
            ("replace", 3): OSError(errno.EIO, "restore failed"),
        },
        {("replace", 1): KeyboardInterrupt()},
        {("replace", 2): KeyboardInterrupt()},
    ],
)
def test_a_web_viewer_copy_is_swapped_as_it_was(
    tmp_path: Path,
    calls: _Calls,
    target_kind: str,
    staging_there: bool,
    failing: dict[tuple[str, int], BaseException],
) -> None:
    calls.fail.update(failing)
    staging = Path("cache") / ".0.38.1.staging"
    target = Path("cache") / "0.38.1"
    _both(
        tmp_path,
        calls,
        _swap_setup(target_kind, staging_there),
        lambda base: _old_swap_into_place(base / staging, base / target),
        lambda base: recordings._swap_into_place(base / staging, base / target),
    )


@pytest.mark.parametrize(
    "error", [OSError(errno.ENOSPC, "no room"), KeyboardInterrupt()]
)
def test_a_failing_mkdtemp_is_raised_as_it_was(
    tmp_path: Path, calls: _Calls, monkeypatch: pytest.MonkeyPatch, error: Any
) -> None:
    asked: list[tuple[Any, ...]] = []

    def mkdtemp(*args: Any, **kw: Any) -> str:
        asked.append((args, kw["prefix"], Path(kw["dir"]).name, sorted(kw)))
        raise error

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    staging = Path("cache") / ".0.38.1.staging"
    target = Path("cache") / "0.38.1"
    _both(
        tmp_path,
        calls,
        _swap_setup("dir", True),
        lambda base: _old_swap_into_place(base / staging, base / target),
        lambda base: recordings._swap_into_place(base / staging, base / target),
    )
    assert len(asked) == 2
    assert asked[0] == asked[1]


def test_recordings_re_exports_the_checked_helpers() -> None:
    # The library check compiles ui.recording_files (#2124).
    for name in ("_fresh", "_publish", "_swap_into_place"):
        assert getattr(recordings, name) is getattr(recording_files, name)
