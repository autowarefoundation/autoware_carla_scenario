"""Work out how an exported scenario package should depend on this framework.

An exported package is only reproducible if the framework it runs on is pinned
to something immutable.  This module resolves exactly one of:

* an **exact release version** -- ``autoware-carla-scenario==X.Y.Z``; or
* an **exact commit** -- the repository URL plus a full commit SHA, for when
  the scenario was authored against an unreleased snapshot; or
* a **local path**, which is only ever produced when the caller explicitly asks
  for a development export.

A branch name is never emitted.  ``main``, ``master`` and ``HEAD`` all move, so
a package pinned to one stops being the package that was tested the moment
somebody pushes -- which is the failure mode this whole exercise exists to
prevent.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional

__all__ = [
    "CONVERTER_DISTRIBUTION",
    "DISTRIBUTION",
    "Pin",
    "PinResolutionError",
    "framework_source_root",
    "resolve_framework_pin",
]

#: Distribution name of the framework.
DISTRIBUTION = "autoware-carla-scenario"

#: The converter the framework imports at module scope
#: (``coordinate.road_lanelet_mapping``) without declaring it as a dependency:
#: inside the workspace it is always installed alongside, so the omission only
#: shows up in a package that depends on the framework alone.  An exported
#: Scenario Package therefore pins both, the same way.
CONVERTER_DISTRIBUTION = "autoware-lanelet2-to-opendrive"

#: Where each project sits inside the repository.
FRAMEWORK_SUBDIRECTORY = "autoware_carla_scenario"
CONVERTER_SUBDIRECTORY = "autoware_lanelet2_to_opendrive"

#: Overrides pin resolution with an exact released version.
VERSION_ENV = "SCENARIO_EXPORT_FRAMEWORK_VERSION"

#: Overrides the repository URL recorded for a commit pin (useful when the
#: checkout's ``origin`` is a fork or an SSH remote the consumer cannot reach).
REPOSITORY_ENV = "SCENARIO_EXPORT_FRAMEWORK_REPOSITORY"

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SCP_URL_PATTERN = re.compile(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?/?$")


class PinResolutionError(RuntimeError):
    """Raised when no immutable pin for the framework can be determined."""


@dataclass(frozen=True)
class Pin:
    """How an exported package depends on ``autoware-carla-scenario``.

    Attributes:
        kind: ``version``, ``git`` or ``path``.
        version: Exact version, for ``kind="version"``.
        repository: Repository URL, for ``kind="git"``.
        commit: Full 40-character commit SHA, for ``kind="git"``.
        subdirectory: Path of the framework project inside the repository.
        path: Absolute local path, for ``kind="path"`` (development only).
        warnings: Reproducibility caveats worth surfacing to the user.
    """

    distribution: str = DISTRIBUTION
    kind: Literal["version", "git", "path"] = "version"
    version: Optional[str] = None
    repository: Optional[str] = None
    commit: Optional[str] = None
    subdirectory: Optional[str] = None
    path: Optional[str] = None
    warnings: tuple[str, ...] = field(default=())

    # -- rendering ------------------------------------------------------

    def requirement(self) -> str:
        """Return the PEP 508 requirement for ``project.dependencies``."""
        if self.kind == "version":
            return f"{self.distribution}=={self.version}"
        # git and path pins carry their locator in [tool.uv.sources]; the
        # requirement itself stays a bare name so the two never disagree.
        return self.distribution

    def uv_source(self) -> Optional[dict[str, Any]]:
        """Return the ``[tool.uv.sources]`` entry, or ``None`` for a version pin."""
        if self.kind == "git":
            source: dict[str, Any] = {
                "git": self.repository,
                "rev": self.commit,
            }
            if self.subdirectory:
                source["subdirectory"] = self.subdirectory
            return source
        if self.kind == "path":
            return {"path": self.path, "editable": True}
        return None

    def manifest(self) -> dict[str, Any]:
        """Return the manifest section describing this pin.

        Only values that were actually determined are written -- a manifest
        that guesses is worse than one that says nothing.
        """
        entry: dict[str, Any] = {"source": self.kind}
        if self.version is not None:
            entry["version"] = self.version
        if self.repository is not None:
            entry["repository"] = self.repository
        if self.commit is not None:
            entry["commit"] = self.commit
        if self.subdirectory is not None:
            entry["subdirectory"] = self.subdirectory
        if self.path is not None:
            entry["path"] = self.path
        return entry

    @property
    def reproducible(self) -> bool:
        """Whether this pin survives being copied to another machine."""
        return self.kind in ("version", "git")

    def companion(self) -> "Pin":
        """Return the matching pin for :data:`CONVERTER_DISTRIBUTION`.

        Pinned the same way as the framework -- same release, same commit, or
        the sibling checkout -- so the two halves of the workspace can never
        drift apart in an exported package.
        """
        if self.kind == "git":
            return replace(
                self,
                distribution=CONVERTER_DISTRIBUTION,
                subdirectory=CONVERTER_SUBDIRECTORY,
                version=_installed_version(CONVERTER_DISTRIBUTION),
                warnings=(),
            )
        if self.kind == "path":
            sibling = (
                str(Path(self.path).parent / CONVERTER_SUBDIRECTORY)
                if self.path
                else None
            )
            return replace(
                self,
                distribution=CONVERTER_DISTRIBUTION,
                path=sibling,
                version=_installed_version(CONVERTER_DISTRIBUTION),
                warnings=(),
            )
        return Pin(
            distribution=CONVERTER_DISTRIBUTION,
            kind="version",
            version=_installed_version(CONVERTER_DISTRIBUTION),
        )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> Optional[str]:
    """Run ``git`` in *repo* and return stripped stdout, or ``None`` on failure."""
    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def normalize_repository_url(url: str) -> str:
    """Return *url* in a form ``uv`` (and anyone else) can clone.

    ``git@host:owner/repo.git`` becomes ``https://host/owner/repo``; anything
    already using a scheme is returned unchanged.
    """
    match = _SCP_URL_PATTERN.match(url.strip())
    if match:
        host, path = match.groups()
        return f"https://{host}/{path}"
    return url.strip().removesuffix(".git") if url.startswith("http") else url.strip()


def framework_source_root() -> Path:
    """Return the directory holding the framework's own ``pyproject.toml``.

    For a source checkout this is ``<repo>/autoware_carla_scenario``; for an
    installed wheel it is the ``site-packages`` directory, which has no
    ``pyproject.toml`` and therefore fails the git probe -- exactly as intended.
    """
    import autoware_carla_scenario  # noqa: PLC0415

    module_file = getattr(autoware_carla_scenario, "__file__", None)
    if module_file is None:  # pragma: no cover - namespace package edge case
        return Path.cwd()
    # <root>/src/autoware_carla_scenario/__init__.py -> <root>
    return Path(module_file).resolve().parents[2]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _installed_version(distribution: str = DISTRIBUTION) -> Optional[str]:
    """Return the installed version of *distribution*, or ``None``."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed
        return None


def _resolve_git_pin(source_root: Path) -> Optional[Pin]:
    """Return a commit pin for *source_root*, or ``None`` when it is not a checkout."""
    toplevel = _git(source_root, "rev-parse", "--show-toplevel")
    if not toplevel:
        return None
    commit = _git(source_root, "rev-parse", "HEAD")
    if not commit or not _SHA_PATTERN.match(commit):
        return None

    repository = os.environ.get(REPOSITORY_ENV) or _git(
        source_root, "config", "--get", "remote.origin.url"
    )
    if not repository:
        return None

    repo_root = Path(toplevel).resolve()
    try:
        subdirectory = source_root.resolve().relative_to(repo_root).as_posix()
    except ValueError:  # pragma: no cover - source_root is inside repo_root
        subdirectory = ""

    warnings: list[str] = []
    dirty = _git(source_root, "status", "--porcelain", "--", str(source_root))
    if dirty:
        warnings.append(
            "The framework checkout has uncommitted changes; the exported "
            f"package pins commit {commit[:12]}, which does not contain them."
        )
    contained = _git(source_root, "branch", "--remotes", "--contains", commit)
    if contained is not None and not contained:
        warnings.append(
            f"Commit {commit[:12]} is not on any remote branch yet. Push it "
            "before sharing this package, or dependency resolution will fail "
            "on another machine."
        )

    return Pin(
        kind="git",
        repository=normalize_repository_url(repository),
        commit=commit,
        subdirectory=subdirectory or None,
        version=_installed_version(),
        warnings=tuple(warnings),
    )


def resolve_framework_pin(*, dev_mode: bool = False) -> Pin:
    """Determine how the exported package should depend on the framework.

    *dev_mode* short-circuits everything and pins the local checkout, which is
    what you want while iterating on the framework and the scenario together --
    and never what you want for a package somebody else will run.  Otherwise:

    1. :data:`VERSION_ENV`, when the framework has a published release that the
       package should track exactly.
    2. The framework's own git checkout, pinned to ``HEAD``'s commit SHA.

    Args:
        dev_mode: Depend on the local framework checkout by path.  Packages
            exported this way are not portable and are marked as such in the
            manifest and the README.

    Returns:
        The resolved :class:`Pin`.

    Raises:
        PinResolutionError: If no immutable pin can be determined and
            *dev_mode* is not set.
    """
    source_root = framework_source_root()

    if dev_mode:
        return Pin(
            kind="path",
            path=str(source_root),
            version=_installed_version(),
            warnings=(
                "Development export: the package depends on the local path "
                f"{source_root}, so it will not resolve on another machine. "
                "Re-export without development mode before sharing it.",
            ),
        )

    version = os.environ.get(VERSION_ENV)
    if version:
        return Pin(kind="version", version=version.strip())

    git_pin = _resolve_git_pin(source_root)
    if git_pin is not None:
        return git_pin

    raise PinResolutionError(
        "Cannot pin autoware-carla-scenario to an exact version or commit. "
        f"Set {VERSION_ENV} to a released version, run the export from a git "
        "checkout of the framework, or export in development mode (which "
        "produces a non-portable package)."
    )
