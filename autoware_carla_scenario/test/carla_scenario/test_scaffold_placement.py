"""How ``scenario-new`` puts a rendered package in place (``scaffold.generator``).

The ``_old_*`` oracles below are ``scaffold/generator.py``'s
``PackageExistsError``, ``_occupied()``, ``_RENAME_ONTO_OCCUPIED`` and
``_move_into_place()`` as they were before they moved into the library check
(#2144), copied verbatim but for their names (``_old_`` / ``_Old`` in front,
and in the references between them).  What the generator has now must do the
same: leave the same files, raise the same error (type name, bases, args,
errno, strerror, filename, str, ``root``, cause and context) and make the same
calls (``os.rename``, ``os.replace``, ``tempfile.mkdtemp``,
``shutil.rmtree``) in the same order with the same arguments, for a target
that is missing, an empty or a full directory, a file, a dangling symlink or a
symlink to a directory, with and without ``force``, and with any of those
calls failing (every errno the code tells apart, others, an ``OSError``
without one, ``KeyboardInterrupt``).
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import pytest

from autoware_carla_scenario.scaffold import generator

# -- the oracles: scaffold/generator.py as it was --------------------------------


class _OldPackageExistsError(FileExistsError):
    """The target directory of :func:`create_scenario_package` already exists.

    Its message names the Python API's ``force=True``; :attr:`root` lets a
    caller with another way to replace it (``scenario-new --force``) say so.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(
            f"Target directory already exists: {root} (pass force=True to replace it)."
        )
        #: The package directory that already exists.
        self.root = root


def _old_occupied(root: Path) -> bool:
    """Whether anything, a dangling symlink included, is at *root*."""
    return root.exists() or root.is_symlink()


#: What a rename onto an occupied *root* fails with: a non-empty directory
#: (``ENOTEMPTY``, or ``EEXIST`` on some systems) or a non-directory.
_OLD_RENAME_ONTO_OCCUPIED = frozenset({errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR})


def _old_move_into_place(staging: Path, root: Path, *, force: bool) -> None:
    """Move the complete package *staging* to *root*.

    Without *force*, whatever appeared at *root* while the package was being
    rendered is left alone and :class:`PackageExistsError` raised: *root* is
    checked again just before the move, and the move is a plain rename, which
    refuses a non-empty directory or a file that appears after that check.

    With *force*, an existing *root* is first renamed aside, beside it, and
    removed only once *staging* is in place, so a failed move restores it
    instead of leaving neither package behind.
    """
    if not force:
        if _old_occupied(root):
            raise _OldPackageExistsError(root)
        try:
            os.rename(staging, root)
        except OSError as exc:
            if exc.errno in _OLD_RENAME_ONTO_OCCUPIED:
                raise _OldPackageExistsError(root) from exc
            raise
        return
    if not _old_occupied(root):
        os.replace(staging, root)
        return
    aside = Path(tempfile.mkdtemp(prefix=f".{root.name}.old-", dir=root.parent))
    old = aside / root.name
    try:
        os.replace(root, old)
    except BaseException:
        # The old package was not moved: drop the empty holder and leave it.
        shutil.rmtree(aside, ignore_errors=True)
        raise
    try:
        os.replace(staging, root)
    except BaseException:
        os.replace(old, root)
        shutil.rmtree(aside, ignore_errors=True)
        raise
    shutil.rmtree(aside, ignore_errors=True)


# -- comparing ------------------------------------------------------------------


def _error(exc: Optional[BaseException]) -> Any:
    if exc is None:
        return None
    return (
        type(exc).__name__.replace("_Old", ""),
        [base.__name__ for base in type(exc).__mro__[1:]],
        exc.args,
        getattr(exc, "errno", None),
        getattr(exc, "strerror", None),
        getattr(exc, "filename", None),
        getattr(exc, "filename2", None),
        str(exc),
        repr(getattr(exc, "root", None)),
        _error(exc.__cause__),
        _error(exc.__context__),
        exc.__suppress_context__,
    )


def _files(root: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            found[key] = ("link", os.readlink(path))
        elif path.is_dir():
            found[key] = ("dir",)
        else:
            found[key] = ("file", path.read_bytes())
    return found


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


def _layouts() -> dict[str, Callable[[Path], None]]:
    def missing(root: Path) -> None:
        pass

    def empty(root: Path) -> None:
        root.mkdir()

    def full(root: Path) -> None:
        root.mkdir()
        (root / "theirs.txt").write_text("not ours")
        (root / "sub").mkdir()
        (root / "sub" / "deep.txt").write_text("deep")

    def file(root: Path) -> None:
        root.write_text("a file")

    def dangling(root: Path) -> None:
        root.symlink_to("nowhere")

    def link_to_dir(root: Path) -> None:
        (root.parent / "target").mkdir()
        (root.parent / "target" / "in.txt").write_text("in")
        root.symlink_to("target")

    return {
        "missing": missing,
        "empty": empty,
        "full": full,
        "file": file,
        "dangling": dangling,
        "link_to_dir": link_to_dir,
    }


def _errors() -> list[BaseException]:
    return [
        OSError(errno.EEXIST, "File exists"),
        OSError(errno.ENOTEMPTY, "Directory not empty", "/a", None, "/b"),
        NotADirectoryError(errno.ENOTDIR, "Not a directory", "/a"),
        OSError(errno.EXDEV, "Invalid cross-device link", "/a", None, "/b"),
        PermissionError(errno.EACCES, "Permission denied", "/a"),
        FileNotFoundError(errno.ENOENT, "No such file or directory"),
        OSError("no errno"),
        KeyboardInterrupt(),
    ]


class _Calls:
    """os.rename, os.replace, tempfile.mkdtemp and shutil.rmtree, recorded;
    the call numbered *fail_at* (of *which*, from 0) raises *error*."""

    def __init__(
        self, which: str = "", fail_at: int = -1, error: Optional[BaseException] = None
    ) -> None:
        self.which = which
        self.fail_at = fail_at
        self.error = error
        self.calls: list[tuple[Any, ...]] = []
        self.counts: dict[str, int] = {}

    def wrap(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(
                (
                    name,
                    tuple((type(a).__name__, str(a)) for a in args),
                    sorted((k, type(v).__name__, str(v)) for k, v in kwargs.items()),
                )
            )
            n = self.counts.get(name, 0)
            self.counts[name] = n + 1
            if name == self.which and n == self.fail_at:
                assert self.error is not None
                raise self.error
            return real(*args, **kwargs)

        return call

    def install(self, patch: pytest.MonkeyPatch) -> None:
        def mkdtemp(*args: Any, **kwargs: Any) -> str:
            # A fixed name: the same, old and new, for files and calls alike.
            made = Path(kwargs["dir"]) / (kwargs["prefix"] + "fixed")
            made.mkdir()
            return str(made)

        patch.setattr(os, "rename", self.wrap("rename", os.rename))
        patch.setattr(os, "replace", self.wrap("replace", os.replace))
        patch.setattr(tempfile, "mkdtemp", self.wrap("mkdtemp", mkdtemp))
        patch.setattr(shutil, "rmtree", self.wrap("rmtree", shutil.rmtree))


def _compare(
    tmp_path: Path,
    layout: str,
    force: bool,
    which: str = "",
    fail_at: int = -1,
    error: Optional[BaseException] = None,
) -> None:
    results = []
    for name, move in (
        ("old", _old_move_into_place),
        ("new", generator._move_into_place),
    ):
        base = tmp_path / name
        base.mkdir()
        staging_holder = base / ".pkg.new-x"
        staging_holder.mkdir()
        staging = staging_holder / "pkg"
        staging.mkdir()
        (staging / "pyproject.toml").write_text("ours")
        (staging / "src").mkdir()
        (staging / "src" / "mod.py").write_text("x = 1")
        root = base / "pkg"
        _layouts()[layout](root)
        calls = _Calls(which, fail_at, error)
        patch = pytest.MonkeyPatch()
        try:
            calls.install(patch)
            try:
                move(staging, root, force=force)
                outcome: Any = ("returned", None)
            except BaseException as exc:  # noqa: BLE001 -- compared
                outcome = ("raised", _error(exc))
        finally:
            patch.undo()
        results.append(_rooted((outcome, calls.calls, _files(base)), str(base)))
    old, new = results
    assert old == new


# -- tests ----------------------------------------------------------------------


def test_the_rename_errnos_are_as_they_were() -> None:
    assert generator._RENAME_ONTO_OCCUPIED == _OLD_RENAME_ONTO_OCCUPIED
    assert type(generator._RENAME_ONTO_OCCUPIED) is frozenset
    assert _OLD_RENAME_ONTO_OCCUPIED == {17, 39, 20}  # Linux's values


@pytest.mark.parametrize("layout", sorted(_layouts()))
def test_occupied_is_as_it_was(layout: str, tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    _layouts()[layout](root)
    assert generator._occupied(root) is _old_occupied(root)
    assert generator._occupied(root / "x") is _old_occupied(root / "x")


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("layout", sorted(_layouts()))
def test_a_package_is_put_in_place_as_it_was(
    layout: str, force: bool, tmp_path: Path
) -> None:
    _compare(tmp_path, layout, force)


@pytest.mark.parametrize("error", range(len(_errors())))
@pytest.mark.parametrize("layout", sorted(_layouts()))
def test_a_failing_rename_is_handled_as_it_was(
    layout: str, error: int, tmp_path: Path
) -> None:
    _compare(tmp_path, layout, False, "rename", 0, _errors()[error])


@pytest.mark.parametrize("error", range(len(_errors())))
@pytest.mark.parametrize(
    ("which", "fail_at"),
    [("replace", 0), ("replace", 1), ("replace", 2), ("mkdtemp", 0)],
)
@pytest.mark.parametrize("layout", sorted(_layouts()))
def test_a_failing_forced_move_is_handled_as_it_was(
    layout: str, which: str, fail_at: int, error: int, tmp_path: Path
) -> None:
    _compare(tmp_path, layout, True, which, fail_at, _errors()[error])


@pytest.mark.parametrize(
    "root", [Path("pkg"), Path("/abs/pkg"), Path("a b/ü"), Path(".")]
)
def test_the_package_exists_error_is_built_as_it_was(root: Path) -> None:
    new = generator.PackageExistsError(root)
    old = _OldPackageExistsError(root)
    assert _error(new) == _error(old)
    assert new.root is root
    assert isinstance(new, FileExistsError) and isinstance(new, OSError)
    assert generator.PackageExistsError.__doc__ == _OldPackageExistsError.__doc__
    assert generator.PackageExistsError.__name__ == "PackageExistsError"
    with pytest.raises(TypeError) as new_refusal:
        generator.PackageExistsError()  # type: ignore[call-arg]
    with pytest.raises(TypeError) as old_refusal:
        _OldPackageExistsError()  # type: ignore[call-arg]
    assert str(new_refusal.value) == str(old_refusal.value).replace("_Old", "")


def test_the_package_exists_error_pickles_as_it_did() -> None:
    import pickle

    outcomes = []
    for cls in (generator.PackageExistsError, _OldPackageExistsError):
        error = cls(Path("pkg"))
        try:
            back = pickle.loads(pickle.dumps(error))
            outcomes.append(("loaded", _error(back), back.__dict__))
        except Exception as exc:  # noqa: BLE001 -- compared
            outcomes.append(("raised", _error(exc), None))
    assert outcomes[0] == outcomes[1]


def test_the_generator_re_exports_the_placements_own_objects() -> None:
    """``scaffold.generator``'s names are ``scaffold.placement``'s objects,
    which the library check compiles (#2144), and the package's
    ``PackageExistsError`` is the same class."""
    from autoware_carla_scenario import scaffold
    from autoware_carla_scenario.scaffold import placement

    for name in (
        "PackageExistsError",
        "_occupied",
        "_RENAME_ONTO_OCCUPIED",
        "_move_into_place",
    ):
        assert getattr(generator, name) is getattr(placement, name), name
    assert scaffold.PackageExistsError is placement.PackageExistsError
    assert generator.create_scenario_package.__module__ == generator.__name__
