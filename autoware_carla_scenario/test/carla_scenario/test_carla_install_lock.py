"""How ``scenario-setup`` opens its lock file (``carla_install._open_lock``).

The ``_old_*`` oracles below are ``carla_install.py``'s ``_UNWRITABLE_ERRNOS``
and ``_open_lock()`` as they were before they moved into the library check
(#2144), copied verbatim but for their names (``_old_`` in front).  What
``carla_install`` has now must do the same: return a stream of the same kind
(type, mode, name, encoding, position) or ``None``, raise the same error
(type, args, errno, strerror, filename, str, cause and context), print the
same to stderr, open the same files in the same modes and leave the same files
behind, whether the directory exists, the lock file exists, is a directory,
a dangling symlink or sits under a file, and whether creating the directory or
either open fails, with every errno the code tells apart and with errors that
are not ``OSError`` (``KeyboardInterrupt``, ``ValueError``).

The failures are injected into what each opens the file with: the oracle's
``open`` is this module's (the builtin), and what ``carla_install`` has is
reached through ``Path.open`` and through an ``open`` set on the module that
defines it, whichever it calls.
"""

from __future__ import annotations

import builtins
import errno
import itertools
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Optional

import pytest

from autoware_carla_scenario import carla_install

# -- the oracles: carla_install.py as it was -----------------------------------


#: The errors of a lock file that cannot be created or written: a read-only
#: file system, or a directory of another user's.
_OLD_UNWRITABLE_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


def _old_open_lock(lock: Path) -> Optional[IO[str]]:
    """*lock* opened to ``flock``: for writing, else read-only, else nothing.

    Raises:
        OSError: *lock* cannot be created for a reason other than permissions
            or a read-only file system.
    """
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        return open(lock, "a")  # noqa: SIM115 -- the caller closes it
    except OSError as exc:
        if exc.errno not in _OLD_UNWRITABLE_ERRNOS:
            raise
        error = exc
    try:
        return open(lock)  # noqa: SIM115 -- for a shared lock, see _locked
    except OSError:
        print(
            f"scenario-setup: cannot create {lock} ({error}); going ahead "
            "unlocked, so run only one scenario-setup for it at a time",
            file=sys.stderr,
        )
        return None


# -- comparing ------------------------------------------------------------------


def _error(exc: Optional[BaseException]) -> Any:
    if exc is None:
        return None
    return (
        type(exc),
        exc.args,
        getattr(exc, "errno", None),
        getattr(exc, "strerror", None),
        getattr(exc, "filename", None),
        str(exc),
        _error(exc.__cause__),
        _error(exc.__context__),
        exc.__suppress_context__,
    )


def _stream(stream: Any) -> Any:
    """What a returned stream is, closing it."""
    if stream is None:
        return None
    try:
        return (
            type(stream),
            stream.mode,
            type(stream.name),
            str(stream.name),
            stream.encoding,
            stream.errors,
            stream.closed,
            stream.tell(),
            stream.writable(),
            stream.readable(),
        )
    finally:
        stream.close()


def _files(root: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            found[key] = ("link", os.readlink(path))
        elif path.is_dir():
            found[key] = ("dir", path.stat().st_mode)
        else:
            found[key] = ("file", path.read_bytes(), path.stat().st_mode)
    return found


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "returned", _stream(call())
    except BaseException as exc:  # noqa: BLE001 -- compared, KeyboardInterrupt too
        return "raised", _error(exc)


# An outcome of one call to open(): None opens the file, an exception is raised.
_Script = list[Optional[BaseException]]


class _Opener:
    """open() as a script says: each call raises the script's next exception,
    or opens the file for real once the script holds ``None`` or ends."""

    def __init__(self, script: _Script, real: Callable[..., Any]) -> None:
        self.script = list(script)
        self.real = real
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, file: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((str(file), args, kwargs))
        planned = self.script.pop(0) if self.script else None
        if planned is not None:
            raise planned
        return self.real(file, *args, **kwargs)


def _run_old(lock: Path, script: _Script, mkdir_error: Optional[BaseException]):
    opener = _Opener(script, builtins.open)
    patch = pytest.MonkeyPatch()
    try:
        patch.setitem(globals(), "open", opener)
        _fail_mkdir(patch, lock, mkdir_error)
        outcome = _outcome(lambda: _old_open_lock(lock))
    finally:
        patch.undo()
        globals().pop("open", None)
    return outcome, opener.calls


def _run_new(lock: Path, script: _Script, mkdir_error: Optional[BaseException]):
    real_path_open = Path.open
    opener = _Opener(script, builtins.open)
    path_opener = _Opener([], real_path_open)
    path_opener.script = opener.script  # one script, whichever is called
    path_opener.calls = opener.calls

    def path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self != lock:
            return real_path_open(self, *args, **kwargs)
        return path_opener(self, *args, **kwargs)

    module = sys.modules[carla_install._open_lock.__module__]
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(module, "open", opener, raising=False)
        patch.setattr(Path, "open", path_open)
        _fail_mkdir(patch, lock, mkdir_error)
        outcome = _outcome(lambda: carla_install._open_lock(lock))
    finally:
        patch.undo()
    # Path.open(mode) is open(path, mode): the same call, either way.
    return outcome, opener.calls


def _fail_mkdir(
    patch: pytest.MonkeyPatch, lock: Path, error: Optional[BaseException]
) -> None:
    if error is None:
        return
    real = Path.mkdir

    def mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == lock.parent:
            raise error
        real(self, *args, **kwargs)

    patch.setattr(Path, "mkdir", mkdir)


def _layouts() -> dict[str, Callable[[Path], None]]:
    def nothing(lock: Path) -> None:
        pass

    def parent(lock: Path) -> None:
        lock.parent.mkdir(parents=True)

    def present(lock: Path) -> None:
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"left\n")

    def directory(lock: Path) -> None:
        lock.mkdir(parents=True)

    def dangling(lock: Path) -> None:
        lock.parent.mkdir(parents=True)
        lock.symlink_to("elsewhere")

    def under_a_file(lock: Path) -> None:
        lock.parent.parent.mkdir(parents=True)
        lock.parent.write_text("a file")

    return {
        "nothing": nothing,
        "parent": parent,
        "present": present,
        "directory": directory,
        "dangling": dangling,
        "under_a_file": under_a_file,
    }


def _errors() -> list[Optional[BaseException]]:
    """What open() or mkdir() is made to fail with (None: it does not)."""
    return [
        None,
        PermissionError(errno.EACCES, "Permission denied", "/x/carla.lock"),
        PermissionError(errno.EPERM, "Operation not permitted"),
        OSError(errno.EROFS, "Read-only file system", "/x/carla.lock"),
        OSError(errno.EACCES, "Permission denied"),
        OSError(errno.ENOSPC, "No space left on device", "/x/carla.lock"),
        IsADirectoryError(errno.EISDIR, "Is a directory", "/x/carla.lock"),
        FileNotFoundError(errno.ENOENT, "No such file or directory"),
        FileExistsError(errno.EEXIST, "File exists", "/x"),
        OSError("no errno"),
        OSError(),
        KeyboardInterrupt(),
        ValueError("not an OSError"),
    ]


def _compare(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    layout: Callable[[Path], None],
    script: _Script,
    mkdir_error: Optional[BaseException],
) -> None:
    results = []
    for name, run in (("old", _run_old), ("new", _run_new)):
        root = tmp_path / name
        lock = root / "bin" / "carla.lock"
        layout(lock)
        capsys.readouterr()
        outcome, calls = run(lock, script, mkdir_error)
        captured = capsys.readouterr()
        result = (outcome, calls, captured.out, captured.err, _files(root))
        # Each run has a root of its own, which every path names.
        results.append(_rooted(result, str(root)))
    old, new = results
    assert old == new


def _rooted(value: Any, root: str) -> Any:
    """*value* with *root* in its strings written ``<root>``."""
    if isinstance(value, str):
        return value.replace(root, "<root>")
    if isinstance(value, tuple):
        return tuple(_rooted(v, root) for v in value)
    if isinstance(value, list):
        return [_rooted(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _rooted(v, root) for k, v in value.items()}
    return value


# -- tests ----------------------------------------------------------------------


def test_the_unwritable_errnos_are_as_they_were() -> None:
    assert carla_install._UNWRITABLE_ERRNOS == _OLD_UNWRITABLE_ERRNOS
    assert type(carla_install._UNWRITABLE_ERRNOS) is frozenset
    assert _OLD_UNWRITABLE_ERRNOS == {13, 1, 30}  # Linux's values


@pytest.mark.parametrize("layout", sorted(_layouts()))
def test_a_lock_opens_as_it_did_on_the_file_system(
    layout: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _compare(tmp_path, capsys, _layouts()[layout], [], None)


@pytest.mark.parametrize("layout", ["nothing", "present", "directory"])
@pytest.mark.parametrize(
    ("first", "second"),
    list(itertools.product(range(len(_errors())), range(len(_errors())))),
)
def test_a_failing_open_is_handled_as_it_was(
    layout: str,
    first: int,
    second: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    errors = _errors()
    _compare(
        tmp_path, capsys, _layouts()[layout], [errors[first], errors[second]], None
    )


@pytest.mark.parametrize("layout", ["nothing", "parent", "present"])
@pytest.mark.parametrize("mkdir_error", range(len(_errors())))
@pytest.mark.parametrize("second", [0, 1, 5, 11])
def test_a_failing_mkdir_is_handled_as_it_was(
    layout: str,
    mkdir_error: int,
    second: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    errors = _errors()
    _compare(
        tmp_path,
        capsys,
        _layouts()[layout],
        [errors[second]],
        errors[mkdir_error],
    )


def test_an_unwritable_lock_names_the_first_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The message quotes the error of the open for writing, not the second."""
    _compare(
        tmp_path,
        capsys,
        _layouts()["nothing"],
        [
            OSError(errno.EROFS, "Read-only file system", "first"),
            FileNotFoundError(errno.ENOENT, "No such file or directory", "second"),
        ],
        None,
    )
    lock = tmp_path / "x" / "carla.lock"
    patch = pytest.MonkeyPatch()
    try:
        real = Path.open

        def path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self == lock:
                raise OSError(errno.EROFS, "Read-only file system", "first")
            return real(self, *args, **kwargs)

        def module_open(*args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.EROFS, "Read-only file system", "first")

        module = sys.modules[carla_install._open_lock.__module__]
        patch.setattr(module, "open", module_open, raising=False)
        patch.setattr(Path, "open", path_open)
        capsys.readouterr()
        assert carla_install._open_lock(lock) is None
    finally:
        patch.undo()
    assert capsys.readouterr().err == (
        f"scenario-setup: cannot create {lock} ([Errno 30] Read-only file "
        "system: 'first'); going ahead unlocked, so run only one scenario-setup "
        "for it at a time\n"
    )


def test_the_stream_is_the_lock_opened_for_appending(tmp_path: Path) -> None:
    lock = tmp_path / "bin" / "carla.lock"
    stream = carla_install._open_lock(lock)
    assert stream is not None
    with stream:
        assert stream.mode == "a"
        assert stream.name == str(lock)
        assert stream.writable() and not stream.readable()
    assert lock.read_bytes() == b""


def test_carla_install_re_exports_the_checked_modules_own_objects() -> None:
    """``carla_install``'s names are ``carla_install_paths``' objects, which
    the library check compiles (#2144); ``_locked()`` stays."""
    from autoware_carla_scenario import carla_install_paths

    for name in ("_open_lock", "_UNWRITABLE_ERRNOS"):
        assert getattr(carla_install, name) is getattr(carla_install_paths, name)
    assert carla_install._open_lock.__module__ == carla_install_paths.__name__
    assert carla_install._locked.__module__ == carla_install.__name__
