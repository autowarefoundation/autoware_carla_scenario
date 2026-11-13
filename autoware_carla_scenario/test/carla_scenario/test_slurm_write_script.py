"""How the SLURM launcher writes a job script (``slurm.submit._write_script``).

``_old_write_script`` below is ``slurm/submit.py``'s ``_write_script()`` as it
was before it moved into the library check (#2124), copied verbatim but for
its name and its docstring.  What ``submit._write_script`` is now must do the
same: leave the same file (bytes and mode, under three umasks, over a file
already there with another mode or longer content, in a directory it has to
make), make the same ``os.open`` call (flags and mode) and ``chmod``, and
raise the same when a step fails.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import submit

# -- the oracle: slurm/submit.py as it was -------------------------------------


def _old_write_script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    path.chmod(0o700)


# -- helpers --------------------------------------------------------------------


def _tree(root: Path) -> dict[str, tuple[str, bytes, int]]:
    """Every entry under *root*: its kind, bytes (files) and permission bits."""
    out: dict[str, tuple[str, bytes, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in sorted(dirnames + filenames):
            path = Path(dirpath) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                out[str(path.relative_to(root))] = (
                    "link",
                    os.readlink(path).encode(),
                    0,
                )
            elif stat.S_ISDIR(mode):
                out[str(path.relative_to(root))] = ("dir", b"", stat.S_IMODE(mode))
            else:
                data = path.read_bytes() if mode & stat.S_IRUSR else b"?"
                out[str(path.relative_to(root))] = ("file", data, stat.S_IMODE(mode))
    return out


def _outcome(
    write: Callable[[Path, str], Any], base: Path, rel: str, text: str
) -> tuple[Any, ...]:
    try:
        result = write(base / rel, text)
    except BaseException as exc:  # noqa: BLE001 -- compared, old and new
        args = tuple(str(a).replace(str(base), "<base>") for a in exc.args)
        filename = getattr(exc, "filename", None)
        return (
            "raised",
            type(exc),
            args,
            None if filename is None else str(filename).replace(str(base), "<base>"),
        )
    return ("returned", result)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[Any, ...]]]:
    """The os.open and Path.chmod calls made, relative to the cwd's name."""
    made: list[tuple[Any, ...]] = []
    real_open = os.open
    real_chmod = Path.chmod

    def spy_open(path: Any, flags: int, mode: int = 0o777, **kw: Any) -> int:
        made.append(("open", Path(path).name, flags, mode, kw))
        return real_open(path, flags, mode, **kw)

    def spy_chmod(self: Path, mode: int, **kw: Any) -> None:
        made.append(("chmod", self.name, mode, kw))
        real_chmod(self, mode, **kw)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(Path, "chmod", spy_chmod)
    yield made


_TEXTS = [
    "",
    "#!/bin/bash\nexport A='b'\n",
    "no newline at the end",
    "crlf\r\nlines\r\n",
    "lone cr\rhere",
    "non-ASCII: é中\U0001f600\n",
    "x" * (1 << 17) + "\n",
]


def _prepare(base: Path, kind: str) -> str:
    """Prepare *base* as *kind* says; the script's path relative to it."""
    base.mkdir(parents=True)
    if kind == "new":
        return "s.sh"
    if kind == "missing-parents":
        return "a/b/c/s.sh"
    if kind == "longer":
        (base / "s.sh").write_bytes(b"old content " * 1000)
        return "s.sh"
    if kind in ("mode-644", "mode-777", "mode-200", "mode-400"):
        target = base / "s.sh"
        target.write_bytes(b"previous\n")
        target.chmod(int(kind[-3:], 8))
        return "s.sh"
    if kind == "symlink":
        (base / "real.sh").write_bytes(b"pointed at\n")
        (base / "real.sh").chmod(0o644)
        (base / "s.sh").symlink_to("real.sh")
        return "s.sh"
    if kind == "dangling":
        (base / "s.sh").symlink_to("nowhere.sh")
        return "s.sh"
    if kind == "directory":
        (base / "s.sh").mkdir()
        return "s.sh"
    if kind == "parent-is-file":
        (base / "p").write_bytes(b"")
        return "p/s.sh"
    raise AssertionError(kind)


_KINDS = [
    "new",
    "missing-parents",
    "longer",
    "mode-644",
    "mode-777",
    "mode-200",
    "mode-400",
    "symlink",
    "dangling",
    "directory",
    "parent-is-file",
]


def _compare(
    tmp_path: Path, calls: list[tuple[Any, ...]], kind: str, text: str
) -> None:
    results = []
    for name, write in (("old", _old_write_script), ("new", submit._write_script)):
        base = tmp_path / name
        rel = _prepare(base, kind)
        calls.clear()
        outcome = _outcome(write, base, rel, text)
        results.append((outcome, list(calls), _tree(base)))
    assert results[0] == results[1]


@pytest.mark.parametrize("umask", [0o022, 0o077, 0o000])
@pytest.mark.parametrize("kind", _KINDS)
def test_the_script_is_written_as_it_was(
    tmp_path: Path, calls: list[tuple[Any, ...]], umask: int, kind: str
) -> None:
    previous = os.umask(umask)
    try:
        for i, text in enumerate(_TEXTS):
            _compare(tmp_path / str(i), calls, kind, text)
    finally:
        os.umask(previous)


@pytest.mark.parametrize("umask", [0o022, 0o077, 0o000])
def test_a_new_script_is_the_users_only(tmp_path: Path, umask: int) -> None:
    previous = os.umask(umask)
    try:
        submit._write_script(tmp_path / "s.sh", "echo\n")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "s.sh").stat().st_mode) == 0o700
    assert (tmp_path / "s.sh").read_bytes() == b"echo\n"


def test_a_lone_surrogate_fails_as_it_did(
    tmp_path: Path, calls: list[tuple[Any, ...]]
) -> None:
    for kind in ("new", "longer", "mode-644"):
        _compare(tmp_path / kind, calls, kind, "a\ud800b\n")


@pytest.mark.parametrize("step", ["open", "chmod", "mkdir"])
@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EIO, "I/O error"),
        OSError(errno.ENOSPC, "no space"),
        PermissionError(errno.EACCES, "denied"),
        KeyboardInterrupt(),
    ],
)
def test_a_failing_step_fails_as_it_did(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[Any, ...]],
    step: str,
    error: BaseException,
) -> None:
    real_mkdir = Path.mkdir

    def failing_open(path: Any, flags: int, mode: int = 0o777, **kw: Any) -> int:
        calls.append(("failed open", Path(path).name, flags, mode, kw))
        raise error

    def failing_chmod(self: Path, *args: Any, **kw: Any) -> None:
        calls.append(("failed chmod", self.name))
        raise error

    def failing_mkdir(self: Path, *args: Any, **kw: Any) -> None:
        # The parent of the script, the base directory itself for "new".
        name = "<base>" if self.name in ("old", "new") else self.name
        calls.append(("failed mkdir", name, args, kw))
        raise error

    if step == "open":
        monkeypatch.setattr(os, "open", failing_open)
    elif step == "chmod":
        monkeypatch.setattr(Path, "chmod", failing_chmod)
    for kind in ("new", "missing-parents", "longer"):
        results = []
        for name, write in (("old", _old_write_script), ("new", submit._write_script)):
            base = tmp_path / kind / name
            rel = _prepare(base, kind)
            calls.clear()
            if step == "mkdir":
                monkeypatch.setattr(Path, "mkdir", failing_mkdir)
            try:
                outcome = _outcome(write, base, rel, "echo\n")
            finally:
                monkeypatch.setattr(Path, "mkdir", real_mkdir)
            results.append((outcome, list(calls), _tree(base)))
        assert results[0] == results[1]
