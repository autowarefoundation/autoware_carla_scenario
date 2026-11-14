"""The file writers of ``detect-no-3d-model`` (``tools.detect_no_3d_model_lanelets``).

The ``_old_*`` oracles below are ``tools/detect_no_3d_model_lanelets.py``'s
``_PartialWriteError``, ``_write_in_place()``, ``_write_atomically()`` and
``_describe_os_error()`` as they were before they moved into the library
check (#2135), copied verbatim but for their names (``_old`` in front, and in
the calls between them).  What the tool has now must do the same: return the
same, raise the same (type, args, errno, strerror, filename, str, and the
``__cause__`` and ``__context__`` chain), leave the same files with the same
bytes and modes, and make the same calls (``tempfile.mkstemp``,
``os.open``, ``os.ftruncate``, ``os.write``, ``os.close``, ``os.fdopen``,
``os.fsync``, ``shutil.copymode``, ``Path.stat``, ``os.chown``,
``os.replace``, ``Path.unlink``) with the same arguments in the same order,
closing every descriptor it opens, whether each call succeeds or one or two
of them fail (an ``OSError`` of any form, a ``PermissionError``, a
``KeyboardInterrupt``) and whether ``os.write`` writes all it is given or
less.
"""

from __future__ import annotations

import errno
import os
import pickle
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.tools import detect_no_3d_model_lanelets as tool
from autoware_carla_scenario.tools import file_writes

# -- the oracles: tools/detect_no_3d_model_lanelets.py as it was ---------------


class _OldPartialWriteError(OSError):
    """Writing a file in place failed after it was opened and emptied."""


def _old_write_in_place(path: Path, data: bytes) -> None:
    """Overwrite *path* with *data* without replacing the file.

    The file keeps its mode and owner.  Opening it does not truncate, so
    an error from the open leaves it unchanged and propagates as is; an
    error after that (a full disk, an I/O error, a quota) may leave it
    empty or partial and is raised as :class:`_OldPartialWriteError`.
    """
    fd = os.open(path, os.O_WRONLY)
    try:
        try:
            os.ftruncate(fd, 0)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)
    except OSError as exc:
        raise _OldPartialWriteError(exc.errno, exc.strerror, str(path)) from exc


def _old_write_atomically(path: Path, data: bytes) -> None:
    """Replace *path*'s content with *data* through a temporary file.

    *path* must already be resolved: replacing a symlink would turn it into
    a regular file instead of writing through it.  The temporary file gets
    *path*'s mode and, where permitted, its owner.  When the directory does
    not allow creating it, *path* is written in place instead.
    """
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except PermissionError:
        _old_write_in_place(path, data)
        return
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, tmp_path)
        stat = path.stat()
        try:
            os.chown(tmp_path, stat.st_uid, stat.st_gid)
        except OSError:
            pass  # Only root (or the owner, for the group) may do this.
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _old_describe_os_error(path: Path, exc: OSError) -> str:
    """*exc* as text, prefixed with *path* unless it already names it."""
    if exc.filename is not None and os.fspath(exc.filename) in (
        str(path),
        os.path.realpath(path),
    ):
        return str(exc)
    return f"{path}: {exc}"


# -- what is compared ----------------------------------------------------------


_PARTIAL = (_OldPartialWriteError, tool._PartialWriteError)


def _type_name(kind: type) -> str:
    # The two partial-write classes are the same class, old and new.
    return "PartialWriteError" if kind in _PARTIAL else kind.__qualname__


def _value(value: object, base: Path) -> object:
    """*value* with the run's directory and temporary names made neutral."""
    if isinstance(value, os.PathLike):
        return ("path", type(value).__name__, _value(os.fspath(value), base))
    if isinstance(value, (bytes, bytearray, memoryview)):
        # os.write is handed a memoryview: what it holds is compared.
        return ("bytes", bytes(value))
    if isinstance(value, str):
        text = value.replace(str(base), "<dir>")
        return re.sub(r"\.([^/]*)\.[A-Za-z0-9_]{8}\.tmp", r".\1.<random>.tmp", text)
    if isinstance(value, tuple):
        return tuple(_value(v, base) for v in value)
    if isinstance(value, os.stat_result):
        return ("stat", value.st_mode, value.st_uid, value.st_gid, value.st_size)
    return value


def _exception(exc: BaseException | None, base: Path, depth: int = 0) -> object:
    if exc is None:
        return None
    described: tuple[object, ...] = (
        _type_name(type(exc)),
        _value(exc.args, base),
        _value(str(exc), base),
        _value(repr(exc).replace(type(exc).__name__, _type_name(type(exc))), base),
    )
    if isinstance(exc, OSError):
        described += (
            _value(exc.errno, base),
            _value(exc.strerror, base),
            _value(exc.filename, base),
            _value(exc.filename2, base),
        )
    if depth < 4:
        described += (
            _exception(exc.__cause__, base, depth + 1),
            _exception(exc.__context__, base, depth + 1),
            exc.__suppress_context__,
        )
    return described


def _open_fds() -> set[int]:
    # The listing's own descriptor is closed again by the time it returns.
    listed = {int(name) for name in os.listdir("/proc/self/fd")}
    open_now: set[int] = set()
    for fd in listed:
        try:
            os.fstat(fd)
        except OSError:
            continue
        open_now.add(fd)
    return open_now


class _Calls:
    """Records the patched calls, failing the ones whose index is in *fail*
    and shortening ``os.write`` to the sizes *writes* gives."""

    def __init__(
        self,
        base: Path,
        fail: dict[int, Callable[[], BaseException]],
        writes: Iterator[int] | None,
    ) -> None:
        self.base = base
        self.fail = fail
        self.writes = writes
        self.log: list[tuple[object, ...]] = []

    def wrap(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            index = len(self.log)
            self.log.append(
                (
                    name,
                    _value(args, self.base),
                    _value(tuple(sorted(kwargs.items())), self.base),
                )
            )
            if index in self.fail:
                raise self.fail[index]()
            if name == "os.write" and self.writes is not None:
                fd, data = args
                size = next(self.writes)
                return real(fd, bytes(data[:size]))
            return real(*args, **kwargs)

        return call


_PATCHED: tuple[tuple[Any, str, str], ...] = (
    (tempfile, "mkstemp", "tempfile.mkstemp"),
    (os, "open", "os.open"),
    (os, "ftruncate", "os.ftruncate"),
    (os, "write", "os.write"),
    (os, "close", "os.close"),
    (os, "fdopen", "os.fdopen"),
    (os, "fsync", "os.fsync"),
    (shutil, "copymode", "shutil.copymode"),
    (Path, "stat", "Path.stat"),
    (os, "chown", "os.chown"),
    (os, "replace", "os.replace"),
    (Path, "unlink", "Path.unlink"),
)


_Outcome = tuple[
    object,
    tuple[tuple[object, ...], ...],
    list[tuple[object, bytes | None, int, int, int]],
    list[int],
]


def _run(
    function: Callable[..., object],
    base: Path,
    target: str,
    data: bytes,
    content: bytes | None,
    mode: int,
    umask: int,
    fail: dict[int, Callable[[], BaseException]],
    writes: list[int] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> _Outcome:
    """Run *function* on *base*/*target*, prepared as described; what it
    returned or raised, the calls it made and the files it left."""
    base.mkdir()
    path = base / target
    if content is not None:
        path.write_bytes(content)
        path.chmod(mode)
    calls = _Calls(base, fail, iter(writes) if writes is not None else None)
    before = _open_fds()
    old_umask = os.umask(umask)
    try:
        with monkeypatch.context() as patch:
            for owner, attribute, name in _PATCHED:
                patch.setattr(
                    owner, attribute, calls.wrap(name, getattr(owner, attribute))
                )
            try:
                outcome: object = ("returned", function(path, data))
            except BaseException as exc:  # noqa: BLE001 - compared below
                outcome = ("raised", _exception(exc, base))
    finally:
        os.umask(old_umask)
    leaked = _open_fds() - before
    for fd in leaked:
        os.close(fd)
    files = sorted(
        (
            _value(str(entry.relative_to(base)), base),
            entry.read_bytes() if entry.is_file() else None,
            os.lstat(entry).st_mode,
            os.lstat(entry).st_uid,
            os.lstat(entry).st_gid,
        )
        for entry in base.rglob("*")
    )
    return outcome, tuple(calls.log), files, sorted(leaked)


def _compare(
    old: Callable[..., object],
    new: Callable[..., object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    **prepared: Any,
) -> _Outcome:
    expected = _run(old, tmp_path / f"{label}-old", monkeypatch=monkeypatch, **prepared)
    actual = _run(new, tmp_path / f"{label}-new", monkeypatch=monkeypatch, **prepared)
    assert actual == expected, label
    if not prepared["fail"]:
        # A failing os.close leaves its descriptor open, old and new alike.
        assert not expected[3], f"{label}: leaked descriptors"
    return expected


def _failures() -> list[tuple[str, Callable[[], BaseException]]]:
    return [
        ("enospc", lambda: OSError(errno.ENOSPC, "No space left on device")),
        ("eio-named", lambda: OSError(errno.EIO, "I/O error", "/some/file")),
        ("eacces", lambda: PermissionError(errno.EACCES, "Permission denied")),
        ("perm-message", lambda: PermissionError("read-only directory")),
        ("enoent", lambda: FileNotFoundError(errno.ENOENT, "No such file", "x")),
        ("bare", lambda: OSError()),
        ("message", lambda: OSError("plain message")),
        ("five", lambda: OSError(errno.EXDEV, "Cross-device", "a", None, "b")),
        ("interrupt", lambda: KeyboardInterrupt()),
        ("value", lambda: ValueError("not an OSError")),
    ]


_DATA = [
    b"",
    b"x",
    b"map:\n  name: M\n  no_3d_model_lanelet_ids:\n  - 7\n  - 8\n",
    bytes(range(256)) * 300,
]

_UMASKS = [0o022, 0o077, 0o000]


def _writers() -> list[tuple[str, Callable[[Path, bytes], None], Callable[..., None]]]:
    return [
        ("in_place", _old_write_in_place, tool._write_in_place),
        ("atomically", _old_write_atomically, tool._write_atomically),
    ]


# -- the writers ---------------------------------------------------------------


@pytest.mark.parametrize("which", ["in_place", "atomically"])
def test_the_writers_write_and_call_as_they_did(
    which: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = {name: (o, n) for name, o, n in _writers()}[which]
    case = 0
    for data in _DATA:
        for content in (None, b"", b"old content that is longer\n" * 50):
            for mode in (0o640, 0o600, 0o444):
                for umask in _UMASKS:
                    case += 1
                    _compare(
                        old,
                        new,
                        tmp_path,
                        monkeypatch,
                        f"c{case}",
                        target="map.yaml",
                        data=data,
                        content=content,
                        mode=mode,
                        umask=umask,
                        fail={},
                        writes=None,
                    )


@pytest.mark.parametrize("which", ["in_place", "atomically"])
def test_the_writers_fail_as_they_did_at_every_call(
    which: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = {name: (o, n) for name, o, n in _writers()}[which]
    case = 0
    for data in (_DATA[2], b""):
        base = dict(
            target="map.yaml",
            data=data,
            content=b"before\n",
            mode=0o640,
            umask=0o022,
            writes=None,
        )
        case += 1
        _, calls, _, _ = _compare(
            old, new, tmp_path, monkeypatch, f"c{case}", fail={}, **base
        )
        for index in range(len(calls)):
            for label, failure in _failures():
                case += 1
                _compare(
                    old,
                    new,
                    tmp_path,
                    monkeypatch,
                    f"c{case}-{label}",
                    fail={index: failure},
                    **base,
                )


@pytest.mark.parametrize("which", ["in_place", "atomically"])
def test_the_writers_fail_as_they_did_at_two_calls(
    which: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = {name: (o, n) for name, o, n in _writers()}[which]
    base = dict(
        target="map.yaml",
        data=_DATA[2],
        content=b"before\n",
        mode=0o640,
        umask=0o022,
        writes=None,
    )
    _, calls, _, _ = _compare(old, new, tmp_path, monkeypatch, "c0", fail={}, **base)
    failures = _failures()
    rng = random.Random(2135)
    case = 0
    for first in range(len(calls)):
        for second in range(first + 1, len(calls) + 4):
            label_a, a = rng.choice(failures)
            label_b, b = rng.choice(failures)
            case += 1
            _compare(
                old,
                new,
                tmp_path,
                monkeypatch,
                f"c{case}-{label_a}-{label_b}",
                fail={first: a, second: b},
                **base,
            )


def test_the_in_place_fallback_is_taken_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tempfile.mkstemp refused (PermissionError of any form): written in
    # place, failing at each call after it, with short writes.
    refusals: list[Callable[[], BaseException]] = [
        lambda: PermissionError(errno.EACCES, "Permission denied"),
        lambda: PermissionError("read-only directory"),
        lambda: PermissionError(),
    ]
    rng = random.Random(1741)
    case = 0
    for refusal in refusals:
        for data in _DATA:
            case += 1
            base = dict(
                target="map.yaml",
                data=data,
                content=b"old\n" * 100,
                mode=0o604,
                umask=0o022,
            )
            _, calls, _, _ = _compare(
                _old_write_atomically,
                tool._write_atomically,
                tmp_path,
                monkeypatch,
                f"c{case}",
                fail={0: refusal},
                writes=None,
                **base,
            )
            for index in range(1, len(calls)):
                for label, failure in _failures():
                    case += 1
                    _compare(
                        _old_write_atomically,
                        tool._write_atomically,
                        tmp_path,
                        monkeypatch,
                        f"c{case}-{label}",
                        fail={0: refusal, index: failure},
                        writes=None,
                        **base,
                    )
            case += 1
            sizes = [rng.randint(1, 7000) for _ in range(len(data) + 1)]
            _compare(
                _old_write_atomically,
                tool._write_atomically,
                tmp_path,
                monkeypatch,
                f"c{case}-short",
                fail={0: refusal},
                writes=sizes,
                **base,
            )
    # A file the fallback cannot open (missing): its error as it is.
    case += 1
    _compare(
        _old_write_atomically,
        tool._write_atomically,
        tmp_path,
        monkeypatch,
        f"c{case}-missing",
        target="map.yaml",
        data=b"x",
        content=None,
        mode=0o600,
        umask=0o022,
        fail={0: refusals[0]},
        writes=None,
    )


def test_short_writes_are_finished_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(449)
    case = 0
    for _ in range(40):
        data = rng.randbytes(rng.choice([1, 2, 17, 4096, 70000]))
        sizes = [rng.choice([1, 2, 3, 100, 5000, 1 << 20]) for _ in range(len(data))]
        # Some writes are cut and then fail.
        fail: dict[int, Callable[[], BaseException]] = {}
        if rng.random() < 0.4:
            fail = {rng.randint(2, 6): rng.choice(_failures())[1]}
        case += 1
        _compare(
            _old_write_in_place,
            tool._write_in_place,
            tmp_path,
            monkeypatch,
            f"c{case}",
            target="map.yaml",
            data=data,
            content=rng.randbytes(rng.randint(0, 9000)),
            mode=0o640,
            umask=0o022,
            fail=fail,
            writes=sizes,
        )


def test_the_writers_take_a_bytearray_and_refuse_a_str_as_they_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = 0
    for _, old, new in _writers():
        for data in (bytearray(b"abc\n"), "text", "", memoryview(b"view")):
            case += 1
            _compare(
                old,
                new,
                tmp_path,
                monkeypatch,
                f"c{case}",
                target="map.yaml",
                data=data,  # type: ignore[arg-type]
                content=b"before\n",
                mode=0o640,
                umask=0o022,
                fail={},
                writes=None,
            )


def test_the_writers_write_through_into_a_directory_target_as_they_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The target is a directory: each fails at its own call.
    for name, old, new in _writers():
        _compare(
            old,
            new,
            tmp_path,
            monkeypatch,
            f"dir-{name}",
            target="sub",
            data=b"x",
            content=None,
            mode=0o600,
            umask=0o022,
            fail={},
            writes=None,
        )


def test_the_atomic_writer_gives_the_temporary_file_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The owner Path.stat() reads is what os.chown is given; its failure is
    # swallowed, anything else is not.
    for owner in ((0, 0), (12345, 678), (1, 2)):
        real_stat = Path.stat

        def stat(
            self: Path,
            *args: Any,
            _owner: tuple[int, int] = owner,
            **kwargs: Any,
        ) -> Any:
            # **kwargs: Python 3.13's Path.is_file() calls
            # self.stat(follow_symlinks=...).
            found = real_stat(self, *args, **kwargs)
            if self.name != "map.yaml":
                return found
            fields = list(found)
            fields[4], fields[5] = _owner
            return os.stat_result(fields)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", stat)
            _compare(
                _old_write_atomically,
                tool._write_atomically,
                tmp_path,
                monkeypatch,
                f"owner-{owner[0]}",
                target="map.yaml",
                data=b"x\n",
                content=b"y\n",
                mode=0o640,
                umask=0o022,
                fail={},
                writes=None,
            )


# -- the partial write's error and its description ------------------------------


_ARGUMENTS: list[tuple[object, ...]] = [
    (),
    ("message",),
    (errno.ENOSPC, "No space left on device"),
    (errno.ENOSPC, "No space left on device", "/a/map.yaml"),
    (None, None, "/a/map.yaml"),
    (errno.EIO, None, "x"),
    (None, "strerror only", "x"),
    (errno.EIO, "I/O", "a", None, "b"),
    (errno.EIO, "I/O", "a", None, "b", "extra"),
    ("a", "b"),
    (1.5, "float", Path("/p")),
]


@pytest.mark.parametrize("args", _ARGUMENTS, ids=repr)
def test_the_partial_write_error_is_built_as_it_was(args: tuple[object, ...]) -> None:
    base = Path("/nonexistent-base")
    old = _OldPartialWriteError(*args)
    new = tool._PartialWriteError(*args)
    assert _exception(new, base) == _exception(old, base)
    assert isinstance(new, OSError) and type(new).__mro__[1:] == (
        OSError,
        Exception,
        BaseException,
        object,
    )
    assert type(new).__doc__ == _OldPartialWriteError.__doc__
    copied = pickle.loads(pickle.dumps(new))
    assert _exception(copied, base) == _exception(old, base)
    with pytest.raises(TypeError):
        tool._PartialWriteError(errno.EIO, "x", filename="f")  # type: ignore[call-arg]


def _errors(base: Path) -> list[OSError]:
    target = base / "map.yaml"
    link = base / "link.yaml"
    return [
        OSError(),
        OSError("message"),
        OSError(errno.EIO, "I/O"),
        OSError(errno.EIO, "I/O", str(target)),
        OSError(errno.EIO, "I/O", str(link)),
        OSError(errno.EIO, "I/O", target),
        OSError(errno.EIO, "I/O", link),
        OSError(errno.EIO, "I/O", os.fsencode(str(target))),
        OSError(errno.EIO, "I/O", "elsewhere"),
        OSError(errno.EIO, "I/O", ""),
        OSError(errno.EIO, "I/O", 5),
        OSError(errno.EXDEV, "x", str(target), None, "other"),
        PermissionError(errno.EACCES, "Permission denied", str(target)),
        FileNotFoundError(errno.ENOENT, "No such file", str(link)),
        tool._PartialWriteError(errno.ENOSPC, "No space", str(target)),
        tool._PartialWriteError(errno.ENOSPC, "No space", str(link)),
        _OldPartialWriteError(None, None, str(target)),
    ]


def _outcome(call: Callable[[], object], base: Path) -> object:
    try:
        return ("returned", _value(call(), base))
    except BaseException as exc:  # noqa: BLE001 - compared
        return ("raised", _exception(exc, base))


def test_the_error_is_described_as_it_was(tmp_path: Path) -> None:
    base = tmp_path / "real"
    base.mkdir()
    (base / "map.yaml").write_text("x")
    (base / "link.yaml").symlink_to(base / "map.yaml")
    alias = tmp_path / "alias"
    alias.symlink_to(base)
    for named in (
        base / "map.yaml",
        base / "link.yaml",
        alias / "map.yaml",
        base / "missing.yaml",
        Path("relative.yaml"),
    ):
        for error in _errors(base):
            expected = _outcome(lambda: _old_describe_os_error(named, error), base)
            actual = _outcome(lambda: tool._describe_os_error(named, error), base)
            assert actual == expected, (named, error)
    # A str path, which os.path.realpath() and str() take too.
    for error in _errors(base):
        named_str: Any = str(base / "link.yaml")
        assert _outcome(lambda: tool._describe_os_error(named_str, error), base) == (
            _outcome(lambda: _old_describe_os_error(named_str, error), base)
        )


def test_the_tool_re_exports_the_checked_writers() -> None:
    # The same objects, under the names they had in the tool (#2135).
    for name in (
        "_PartialWriteError",
        "_write_in_place",
        "_write_atomically",
        "_describe_os_error",
    ):
        assert getattr(tool, name) is getattr(file_writes, name), name
