"""The file helpers of the SLURM worker's start lock (``slurm.worker``).

The ``_old_*`` oracles below are ``slurm/worker.py``'s ``_NO_FLOCK_ERRNOS``,
``_LockId``, ``_lock_sibling()``, ``_server_now()``, ``_ServerClock``,
``_lock_identity()``, ``_create_lock()``, ``_release_lock()`` and
``_remove_stale()`` as they were before they moved into the library check
(#2116), copied verbatim but for their names (``_old_`` in front, and in the
calls between them).  What the worker has now must do the same: return the
same, raise the same, leave the same files with the same bytes and modes, and
make the same system calls (``os.open`` with its flags and mode,
``os.write``, ``os.fstat``, ``os.close``, ``os.unlink``) in the same order,
closing every descriptor it opens, whether a call succeeds or fails part-way
(an ``OSError`` or a ``KeyboardInterrupt`` from any of them).

Lock files hold any bytes: empty, ASCII tokens, invalid UTF-8, a BOM, CR, LF
and CRLF, sequences cut at the end, and random bytes longer than a read
buffer.  The server clock runs on a fake ``time.monotonic()`` and a fake probe
(``_server_now``), over random sequences of reads, NaN and infinities among
them, comparing every answer, every probe and every clock read.
"""

from __future__ import annotations

import contextlib
import errno
import math
import os
import random
import re
import time
import uuid
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import worker

# -- the oracles: slurm/worker.py as it was ------------------------------------

#: The errors of a ``flock`` the file system does not support (some NFS and
#: Lustre mounts): the start lock is then taken with an ``O_EXCL`` file.
_OLD_NO_FLOCK_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK, errno.EINVAL}
)


#: A lock file's identity: its device, inode, mtime (ns) and token.  The
#: mtime tells a lock apart from a later one that reuses its inode: a lock
#: is only taken over once it is old, and a new one is not.
_OldLockId = tuple[int, int, int, str]


def _old_lock_sibling(path: Path, kind: str) -> Path:
    """A name beside *path* that no other process uses."""
    return path.with_name(f"{path.name}.{kind}.{uuid.uuid4().hex}")


def _old_server_now(path: Path) -> float:
    """The time by the clock of the file system holding *path*: the mtime of
    a file created beside it.

    A lock's mtime is set by that clock too (the NFS server's), so the age of
    a lock is measured with one clock, whatever the skew between this node's
    and the server's.
    """
    probe = _old_lock_sibling(path, "probe")
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        return os.fstat(fd).st_mtime
    finally:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(probe)


class _OldServerClock:
    """The file server's clock, as seen from beside *path*.

    Its offset from this node's ``time.monotonic()`` (which no clock
    adjustment moves) is measured with a probe file (:func:`_server_now`)
    the first time it is read, and again whenever a file looks stale by it,
    so polling costs no probe file and nothing is judged stale by an
    estimate alone.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset: float | None = None

    def now(self) -> float:
        """The server's time, measured now."""
        server = _old_server_now(self._path)
        self._offset = server - time.monotonic()
        return server

    def stale(self, mtime: float, after: float) -> bool:
        """Whether a file of *mtime* is more than *after* seconds old."""
        if self._offset is not None and (
            time.monotonic() + self._offset - mtime <= after
        ):
            return False
        return self.now() - mtime > after


def _old_lock_identity(path: Path) -> tuple[_OldLockId, float] | None:
    """The identity of the lock file at *path* and its mtime; ``None`` when
    there is none."""
    try:
        with open(path, "rb") as stream:
            token = stream.read().decode("utf-8", "replace")
            st = os.fstat(stream.fileno())
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_mtime_ns, token), st.st_mtime


def _old_create_lock(path: Path, token: str) -> bool:
    """Create *path* holding *token* with ``O_EXCL``; whether it was not
    there yet."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, token.encode("utf-8"))
    except BaseException:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(path)  # just made, and no one takes over a fresh lock
        raise
    os.close(fd)
    return True


def _old_release_lock(path: Path, token: str) -> None:
    """Remove *path* if it still holds *token*.

    Between the read and the unlink the lock can only change hands if it is
    taken over as stale in that moment, i.e. if it was held for longer than
    it may be.
    """
    held = _old_lock_identity(path)
    if held is not None and held[0][3] == token:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _old_remove_stale(path: Path, stale: _OldLockId) -> bool:
    """Unlink *path* if it is still the lock judged stale as *stale*; whether
    it was.

    Nothing is renamed or put back: a lock other than *stale* is left where
    it is.  Between the check and the unlink *path* can only change if the
    holder of *stale* is alive after all, or another waiter removes *stale*
    meanwhile, which :func:`_take_over` rules out but for a waiter that
    stalled past :data:`_TAKEOVER_STALE_AFTER`.
    """
    seen = _old_lock_identity(path)
    if seen is None or seen[0] != stale:
        return False
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    return True


# -- harness -------------------------------------------------------------------


def _open_fds() -> list[str]:
    return sorted(os.listdir("/proc/self/fd"))


@pytest.fixture
def no_fd_leak() -> Iterator[None]:
    """Fail the test if it leaves a descriptor open."""
    before = _open_fds()
    yield
    assert _open_fds() == before


class _Calls:
    """``os.open``, ``os.write``, ``os.fstat``, ``os.close`` and ``os.unlink``
    recorded in order, with paths relative to *root* and descriptors
    numbered in the order they were opened; any of them can be made to
    raise at its n-th call."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.log: list[tuple[Any, ...]] = []
        self.fds: dict[int, int] = {}
        self.fail: dict[tuple[str, int], BaseException] = {}
        self.count: dict[str, int] = {}

    def _path(self, path: Any) -> str:
        return os.path.relpath(os.fspath(path), self.root)

    def _fd(self, fd: int) -> Any:
        return ("fd", self.fds.get(fd, f"other{fd}"))

    def _maybe_fail(self, name: str) -> None:
        n = self.count[name] = self.count.get(name, 0) + 1
        exc = self.fail.get((name, n))
        if exc is not None:
            self.log.append((name, "raised", type(exc).__name__))
            raise exc

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_open, real_write = os.open, os.write
        real_fstat, real_close, real_unlink = os.fstat, os.close, os.unlink

        def fake_open(path: Any, flags: int, mode: int = 0o777, **kw: Any) -> int:
            self.log.append(("open", self._path(path), flags, mode, kw))
            self._maybe_fail("open")
            fd = real_open(path, flags, mode, **kw)
            self.fds[fd] = len(self.fds)
            return fd

        def fake_write(fd: int, data: bytes) -> int:
            self.log.append(("write", self._fd(fd), bytes(data)))
            self._maybe_fail("write")
            return real_write(fd, data)

        def fake_fstat(fd: int) -> os.stat_result:
            self.log.append(("fstat", self._fd(fd)))
            self._maybe_fail("fstat")
            return real_fstat(fd)

        def fake_close(fd: int) -> None:
            self.log.append(("close", self._fd(fd)))
            self._maybe_fail("close")
            real_close(fd)

        def fake_unlink(path: Any, **kw: Any) -> None:
            self.log.append(("unlink", self._path(path), kw))
            self._maybe_fail("unlink")
            real_unlink(path, **kw)

        monkeypatch.setattr(os, "open", fake_open)
        monkeypatch.setattr(os, "write", fake_write)
        monkeypatch.setattr(os, "fstat", fake_fstat)
        monkeypatch.setattr(os, "close", fake_close)
        monkeypatch.setattr(os, "unlink", fake_unlink)


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "returned", call()
    except BaseException as exc:  # noqa: BLE001 -- KeyboardInterrupt too
        return "raised", (type(exc), exc.args)


def _files(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mode)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _compare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[Path], Any],
    old: Callable[[Path, Any], Any],
    new: Callable[[Path, Any], Any],
    fail: dict[tuple[str, int], BaseException] | None = None,
) -> tuple[str, Any]:
    """Run *old* and *new* each in a directory *setup* prepared the same way,
    and compare what they return or raise, the calls they make and the files
    they leave."""
    results = []
    for name, fn in (("old", old), ("new", new)):
        root = tmp_path / name
        root.mkdir()
        arg = setup(root)
        calls = _Calls(root)
        calls.fail = dict(fail or {})
        before = _open_fds()
        with monkeypatch.context() as m:
            calls.install(m)
            outcome = _outcome(partial(fn, root, arg))
        assert _open_fds() == before, f"{name} left a descriptor open"
        results.append((outcome, calls.log, _files(root)))
    assert results[0] == results[1]
    return results[0][0]


_ERRORS: list[BaseException] = [
    OSError(errno.EIO, "I/O error"),
    OSError(errno.ENOSPC, "No space left on device"),
    FileNotFoundError(errno.ENOENT, "gone"),
    PermissionError(errno.EACCES, "denied"),
    KeyboardInterrupt("signal SIGTERM"),
]


# -- constants -----------------------------------------------------------------


def test_the_flock_errnos_and_the_lock_id_are_as_they_were() -> None:
    assert isinstance(worker._NO_FLOCK_ERRNOS, frozenset)
    assert worker._NO_FLOCK_ERRNOS == _OLD_NO_FLOCK_ERRNOS
    # Linux's values, which the errno shim of the check declares.
    assert sorted(worker._NO_FLOCK_ERRNOS) == [22, 37, 38, 95]
    assert worker._LockId == _OldLockId


# -- _lock_identity ------------------------------------------------------------


def _contents() -> list[bytes]:
    rng = random.Random(2116)
    fixed = [
        b"",
        b"host:123:0123456789abcdef0123456789abcdef",
        b"\xff",
        b"\xff\xfe\x00t",
        b"\xef\xbb\xbftoken",
        b"a\r\nb\rc\nd",
        b"\r",
        b"abc\r",
        b"\r\n",
        b"\xe2\x82",
        b"\xe2\x82\xac",
        b"\xe2\x82\xactail\xe2",
        b"\x00\x00",
        b"\xed\xa0\x80",  # an encoded surrogate: invalid UTF-8
        b"\xf4\x90\x80\x80",  # past U+10FFFF
        "日本語のトークン".encode(),
        b"x" * 8191 + "€".encode(),  # straddling a read buffer
        b"x" * 8190 + b"\r\n" + b"y",
        b"x" * 8192 + b"\r",
    ]
    drawn = [
        bytes(rng.randrange(256) for _ in range(rng.choice([1, 7, 100, 9000, 20000])))
        for _ in range(40)
    ]
    mixed = [
        b"".join(
            rng.choice([b"\r", b"\n", b"\r\n", "é".encode(), b"\xc3", b"a", b"\x80"])
            for _ in range(rng.randrange(1, 3000))
        )
        for _ in range(40)
    ]
    return fixed + drawn + mixed


def test_a_lock_files_identity_is_read_as_it_was(
    tmp_path: Path, no_fd_leak: None
) -> None:
    for i, data in enumerate(_contents()):
        path = tmp_path / f"lock{i}"
        path.write_bytes(data)
        old = _old_lock_identity(path)
        new = worker._lock_identity(path)
        assert new == old, data[:40]
        assert new is not None
        assert new[0][3] == data.decode("utf-8", "replace")
        assert type(new[0][3]) is str and type(new[1]) is float


def test_a_missing_lock_has_no_identity_and_a_directory_raises(
    tmp_path: Path, no_fd_leak: None
) -> None:
    assert worker._lock_identity(tmp_path / "missing") is None
    assert _old_lock_identity(tmp_path / "missing") is None
    assert _outcome(lambda: worker._lock_identity(tmp_path)) == _outcome(
        lambda: _old_lock_identity(tmp_path)
    )
    assert _outcome(lambda: worker._lock_identity(tmp_path))[1][0] is (
        IsADirectoryError
    )
    # A parent that is a file: NotADirectoryError, not swallowed.
    (tmp_path / "file").write_text("x")
    assert _outcome(lambda: worker._lock_identity(tmp_path / "file" / "x")) == (
        _outcome(lambda: _old_lock_identity(tmp_path / "file" / "x"))
    )


@pytest.mark.parametrize("exc", _ERRORS, ids=repr)
def test_a_failing_fstat_of_the_identity_closes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    def setup(root: Path) -> Path:
        (root / "lock").write_bytes(b"tok")
        return root / "lock"

    _compare(
        tmp_path,
        monkeypatch,
        setup,
        lambda _r, p: _old_lock_identity(p),
        lambda _r, p: worker._lock_identity(p),
        fail={("fstat", 1): exc},
    )


# -- _create_lock --------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["host:1:abc", "", "日本:42:é", "a\r\nb", "\x00", "\ud800", "x" * 70000],
    ids=repr,
)
@pytest.mark.parametrize("existing", [None, b"other"])
def test_creating_a_lock_is_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    existing: bytes | None,
) -> None:
    def setup(root: Path) -> Path:
        if existing is not None:
            (root / "lock").write_bytes(existing)
        return root / "lock"

    outcome = _compare(
        tmp_path,
        monkeypatch,
        setup,
        lambda _r, p: _old_create_lock(p, token),
        lambda _r, p: worker._create_lock(p, token),
    )
    if token == "\ud800" and existing is None:  # encoded only once created
        assert outcome[0] == "raised" and outcome[1][0] is UnicodeEncodeError
    else:
        assert outcome == ("returned", existing is None)


@pytest.mark.parametrize("umask", [0o022, 0o077, 0o000])
def test_a_created_lock_has_the_mode_it_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, umask: int
) -> None:
    previous = os.umask(umask)
    try:
        _compare(
            tmp_path,
            monkeypatch,
            lambda root: root / "lock",
            lambda _r, p: _old_create_lock(p, "t"),
            lambda _r, p: worker._create_lock(p, "t"),
        )
    finally:
        os.umask(previous)
    assert (tmp_path / "new" / "lock").stat().st_mode & 0o777 == 0o644 & ~umask


@pytest.mark.parametrize("exc", _ERRORS, ids=repr)
@pytest.mark.parametrize("where", ["open", "write", "unlink"])
def test_a_failing_creation_cleans_up_as_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException, where: str
) -> None:
    fail = {(where, 1): exc}
    if where == "unlink":  # only reached when the write failed
        fail[("write", 1)] = OSError(errno.EIO, "I/O error")
    _compare(
        tmp_path,
        monkeypatch,
        lambda root: root / "lock",
        lambda _r, p: _old_create_lock(p, "tok"),
        lambda _r, p: worker._create_lock(p, "tok"),
        fail=fail,
    )


def test_a_failing_close_of_a_created_lock_is_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The close made to fail leaves its descriptor open in both (the fake
    raises before closing): what each returns, raises and calls is compared,
    and the descriptor is then closed here."""
    for write_fails in (True, False):
        results = []
        for name, fn in (("old", _old_create_lock), ("new", worker._create_lock)):
            root = tmp_path / f"{name}-{write_fails}"
            root.mkdir()
            calls = _Calls(root)
            calls.fail = {("close", 1): OSError(errno.EIO, "I/O error")}
            if write_fails:
                calls.fail[("write", 1)] = KeyboardInterrupt("signal SIGTERM")
            before = set(_open_fds())
            with monkeypatch.context() as m:
                calls.install(m)
                outcome = _outcome(partial(fn, root / "lock", "tok"))
            for fd in set(_open_fds()) - before:
                with contextlib.suppress(OSError):
                    os.close(int(fd))
            results.append((outcome, calls.log, _files(root)))
        assert results[0] == results[1]


# -- _release_lock and _remove_stale -------------------------------------------


@pytest.mark.parametrize("held", [None, b"mine", b"theirs", b"\xffmine"])
@pytest.mark.parametrize(
    "unlink_fails", [None, FileNotFoundError(errno.ENOENT, "x"), *_ERRORS[:2]]
)
def test_releasing_a_lock_is_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    held: bytes | None,
    unlink_fails: BaseException | None,
) -> None:
    def setup(root: Path) -> Path:
        if held is not None:
            (root / "lock").write_bytes(held)
        return root / "lock"

    fail = {} if unlink_fails is None else {("unlink", 1): unlink_fails}
    _compare(
        tmp_path,
        monkeypatch,
        setup,
        lambda _r, p: _old_release_lock(p, "mine"),
        lambda _r, p: worker._release_lock(p, "mine"),
        fail=fail,
    )


@pytest.mark.parametrize(
    "judged", ["same", "token", "inode", "mtime", "device", "missing"]
)
@pytest.mark.parametrize(
    "unlink_fails", [None, FileNotFoundError(errno.ENOENT, "x"), *_ERRORS[:2]]
)
def test_removing_a_stale_lock_is_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    judged: str,
    unlink_fails: BaseException | None,
) -> None:
    def setup(root: Path) -> tuple[Path, Any]:
        path = root / "lock"
        path.write_bytes(b"dead-task")
        old = time.time() - 3600
        os.utime(path, (old, old))
        st = path.stat()
        stale: Any = (st.st_dev, st.st_ino, st.st_mtime_ns, "dead-task")
        if judged == "token":
            stale = (*stale[:3], "other")
        elif judged == "inode":
            stale = (stale[0], stale[1] + 1, *stale[2:])
        elif judged == "mtime":
            stale = (*stale[:2], stale[2] + 1, stale[3])
        elif judged == "device":
            stale = (stale[0] + 1, *stale[1:])
        elif judged == "missing":
            path.unlink()
        return path, stale

    fail = {} if unlink_fails is None else {("unlink", 1): unlink_fails}
    outcome = _compare(
        tmp_path,
        monkeypatch,
        setup,
        lambda _r, a: _old_remove_stale(*a),
        lambda _r, a: worker._remove_stale(*a),
        fail=fail,
    )
    if judged != "same":
        assert outcome == ("returned", False)


# -- _lock_sibling and _server_now ---------------------------------------------


class _FixedUuids:
    """``uuid.uuid4`` handing out 1, 2, 3, ... and counting its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> uuid.UUID:
        self.calls += 1
        return uuid.UUID(int=self.calls, version=4)


def test_a_siblings_name_is_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, kind in [("start.lock.excl", "probe"), ("x", "takeover"), ("é", "")]:
        path = Path("/batch") / name
        olds, news = _FixedUuids(), _FixedUuids()
        monkeypatch.setattr(uuid, "uuid4", olds)
        old = [_old_lock_sibling(path, kind) for _ in range(3)]
        monkeypatch.setattr(uuid, "uuid4", news)
        new = [worker._lock_sibling(path, kind) for _ in range(3)]
        assert new == old and olds.calls == news.calls == 3
    monkeypatch.undo()
    sibling = worker._lock_sibling(Path("/b/start.lock.excl"), "probe")
    assert re.fullmatch(r"start\.lock\.excl\.probe\.[0-9a-f]{32}", sibling.name)
    assert sibling.parent == Path("/b")


@pytest.mark.parametrize(
    "fail",
    [
        {},
        {("open", 1): FileExistsError(errno.EEXIST, "exists")},
        *[{("open", 1): exc} for exc in _ERRORS],
        *[{("fstat", 1): exc} for exc in _ERRORS],
        *[{("unlink", 1): exc} for exc in _ERRORS],
        {("fstat", 1): OSError(errno.EIO, "x"), ("unlink", 1): KeyboardInterrupt()},
    ],
    ids=repr,
)
def test_the_server_clocks_probe_is_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: dict[Any, BaseException]
) -> None:
    def setup(root: Path) -> Path:
        monkeypatch.setattr(uuid, "uuid4", _FixedUuids())
        return root / "start.lock.excl"

    outcome = _compare(
        tmp_path,
        monkeypatch,
        setup,
        lambda _r, p: _old_server_now(p) > 0,
        lambda _r, p: worker._server_now(p) > 0,
        fail=fail,
    )
    if (
        not fail
        or list(fail) == [("unlink", 1)]
        and not isinstance(fail[("unlink", 1)], KeyboardInterrupt)
    ):
        assert outcome == ("returned", True)


def test_the_probe_returns_the_probes_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_fd_leak: None
) -> None:
    real_fstat = os.fstat
    seen: list[float] = []

    def fstat(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        seen.append(st.st_mtime)
        return st

    monkeypatch.setattr(os, "fstat", fstat)
    old = _old_server_now(tmp_path / "lock")
    new = worker._server_now(tmp_path / "lock")
    assert [old, new] == seen
    assert list(tmp_path.iterdir()) == []


# -- _ServerClock --------------------------------------------------------------


class _Script:
    """A fake ``time.monotonic()`` and a fake probe, each answering from its
    own list of values and recording its reads."""

    def __init__(self, monotonic: list[float], server: list[float]) -> None:
        self._monotonic = iter(monotonic)
        self._server = iter(server)
        self.log: list[tuple[str, Any]] = []

    def monotonic(self) -> float:
        value = next(self._monotonic)
        self.log.append(("monotonic", value))
        return value

    def server_now(self, path: Path) -> float:
        value = next(self._server)
        self.log.append(("probe", path))
        if isinstance(value, BaseException):
            raise value
        return value


def _special(rng: random.Random) -> float:
    return rng.choice([math.nan, math.inf, -math.inf, 0.0, -0.0, 1e308, -1e308])


def _clock_run(
    clock_cls: type, set_probe: Callable[[Any], None], script: _Script, ops: list[Any]
) -> list[Any]:
    set_probe(script.server_now)
    clock = clock_cls(Path("/batch/start.lock.excl"))
    answers = []
    for op in ops:
        if op[0] == "now":
            answers.append(_outcome(clock.now))
        else:
            answers.append(_outcome(partial(clock.stale, op[1], op[2])))
    return [answers, script.log]


def test_the_server_clock_answers_as_it_did(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(1360)
    new_globals = worker._ServerClock.now.__globals__
    for _ in range(400):
        n = rng.randrange(1, 12)
        ops: list[Any] = []
        for _ in range(n):
            if rng.random() < 0.2:
                ops.append(("now",))
            else:
                mtime = _special(rng) if rng.random() < 0.1 else rng.uniform(0, 2e3)
                after = _special(rng) if rng.random() < 0.1 else rng.uniform(0, 120)
                ops.append(("stale", mtime, after))
        monotonic = [
            _special(rng) if rng.random() < 0.05 else rng.uniform(0, 2e3)
            for _ in range(3 * n)
        ]
        server: list[Any] = [
            _special(rng) if rng.random() < 0.05 else rng.uniform(0, 4e3)
            for _ in range(3 * n)
        ]
        if rng.random() < 0.2:
            server[rng.randrange(len(server))] = rng.choice(_ERRORS)
        results = []
        for clock_cls, set_probe in (
            (
                _OldServerClock,
                lambda f: monkeypatch.setitem(globals(), "_old_server_now", f),
            ),
            (
                worker._ServerClock,
                lambda f: monkeypatch.setitem(new_globals, "_server_now", f),
            ),
        ):
            script = _Script(monotonic, server)
            with monkeypatch.context() as m:
                m.setattr(time, "monotonic", script.monotonic)
                results.append(_clock_run(clock_cls, set_probe, script, ops))
        assert repr(results[0]) == repr(results[1]), ops


def test_the_server_clock_probes_once_while_polling_a_fresh_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling a fresh lock reads the probe once, then the offset only; a
    lock that looks stale by the offset is probed again before it is judged
    stale."""
    script = _Script([10.0, 11.0, 12.0, 100.0], [1000.0, 1100.0])
    monkeypatch.setattr(time, "monotonic", script.monotonic)
    monkeypatch.setitem(
        worker._ServerClock.now.__globals__, "_server_now", script.server_now
    )
    clock = worker._ServerClock(Path("/b/lock"))
    assert clock.stale(995.0, 60.0) is False  # probe: offset 990
    assert clock.stale(995.0, 60.0) is False  # 11 + 990 - 995 <= 60
    # 12 + 990 - 940 = 62 by the offset: probed again (1100) before judged.
    assert clock.stale(940.0, 60.0) is True
    assert [kind for kind, _ in script.log] == [
        "probe",
        "monotonic",
        "monotonic",
        "monotonic",
        "probe",
        "monotonic",
    ]
