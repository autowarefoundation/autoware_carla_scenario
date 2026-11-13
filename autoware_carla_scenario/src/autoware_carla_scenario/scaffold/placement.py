"""How ``scenario-new`` puts a rendered package in place of its target directory.

The move of :mod:`autoware_carla_scenario.scaffold.generator`, which
re-exports every name of it: whether anything is at the target
(:func:`_occupied`), the plain rename that refuses an occupied one, or the
replacement that keeps the old package until the new one is in place
(:func:`_move_into_place`), and the error of a target that already exists
(:class:`PackageExistsError`). It needs nothing Codon lacks, so the library
check compiles it from its source (``docs/typecheck.md``); the rendering of the
templates (jinja2) and the command line stay in ``scaffold.generator``.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from pathlib import Path


class PackageExistsError(FileExistsError):
    """The target directory of :func:`create_scenario_package` already exists.

    Its message names the Python API's ``force=True``; :attr:`root` lets a
    caller with another way to replace it (``scenario-new --force``) say so.
    """

    #: The package directory that already exists.
    root: Path

    def __init__(self, root: Path) -> None:
        super().__init__(
            f"Target directory already exists: {root} (pass force=True to replace it)."
        )
        self.root = root


def _occupied(root: Path) -> bool:
    """Whether anything, a dangling symlink included, is at *root*."""
    return root.exists() or root.is_symlink()


#: What a rename onto an occupied *root* fails with: a non-empty directory
#: (``ENOTEMPTY``, or ``EEXIST`` on some systems) or a non-directory.
_RENAME_ONTO_OCCUPIED = frozenset({errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR})


def _move_into_place(staging: Path, root: Path, *, force: bool) -> None:
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
        if _occupied(root):
            raise PackageExistsError(root)
        try:
            os.rename(staging, root)
        except OSError as exc:
            if exc.errno in _RENAME_ONTO_OCCUPIED:
                raise PackageExistsError(root) from exc
            raise
        return
    if not _occupied(root):
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
