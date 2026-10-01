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

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

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
#: (``coordinate.road_lanelet_mapping``).  It lives in its own repository, so an
#: exported Scenario Package pins it explicitly, to exactly the revision that is
#: installed alongside the framework.
CONVERTER_DISTRIBUTION = "autoware-lanelet2-to-opendrive"

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
        extras: Extras to request of this distribution, e.g. ``("carla",)``.
        warnings: Reproducibility caveats worth surfacing to the user.
    """

    distribution: str = DISTRIBUTION
    kind: Literal["version", "git", "path"] = "version"
    version: Optional[str] = None
    repository: Optional[str] = None
    commit: Optional[str] = None
    subdirectory: Optional[str] = None
    path: Optional[str] = None
    extras: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())

    # -- rendering ------------------------------------------------------

    def requirement(self) -> str:
        """Return the PEP 508 requirement for ``project.dependencies``."""
        name = self.distribution
        if self.extras:
            name += f"[{','.join(self.extras)}]"
        if self.kind == "version":
            return f"{name}=={self.version}"
        # git and path pins carry their locator in [tool.uv.sources]; the
        # requirement itself stays a bare name so the two never disagree.
        return name

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
        if self.extras:
            entry["extras"] = list(self.extras)
        return entry

    @property
    def reproducible(self) -> bool:
        """Whether this pin survives being copied to another machine."""
        return self.kind in ("version", "git")

    def companion(self) -> "Pin":
        """Return the matching pin for :data:`CONVERTER_DISTRIBUTION`.

        The converter lives in its own repository, so it cannot share the
        framework's commit.  It is pinned to exactly what is installed next to
        the framework -- the revision the lockfile resolved and the test suite
        ran against -- as recorded in its PEP 610 ``direct_url.json``:

        * a version pin of the framework pins the converter's installed
          version too;
        * a converter installed from a git repository is pinned to that
          repository and commit;
        * a converter installed from a local directory is pinned by path,
          which only a development export should ever end up with;
        * a converter installed from an index is pinned to its exact version.

        Extras are not carried across: they belong to the distribution that
        declares them.
        """
        version = _installed_version(CONVERTER_DISTRIBUTION)
        exact = Pin(
            distribution=CONVERTER_DISTRIBUTION, kind="version", version=version
        )
        if self.kind == "version":
            return exact

        origin = _installed_direct_url(CONVERTER_DISTRIBUTION)
        if origin is None:
            return exact

        url = origin.get("url", "")
        vcs_info = origin.get("vcs_info")
        if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git":
            commit = vcs_info.get("commit_id")
            if isinstance(commit, str) and _SHA_PATTERN.match(commit):
                return Pin(
                    distribution=CONVERTER_DISTRIBUTION,
                    kind="git",
                    repository=normalize_repository_url(url),
                    commit=commit,
                    subdirectory=origin.get("subdirectory") or None,
                    version=version,
                )
            return exact

        if url.startswith("file://"):
            path = url2pathname(urlparse(url).path)
            warnings: tuple[str, ...] = ()
            if self.kind != "path":
                warnings = (
                    f"{CONVERTER_DISTRIBUTION} is installed from the local "
                    f"directory {path}, so the exported package depends on it "
                    "by path and will not resolve on another machine. Install "
                    "it from its git repository before sharing the package.",
                )
            return Pin(
                distribution=CONVERTER_DISTRIBUTION,
                kind="path",
                path=path,
                version=version,
                warnings=warnings,
            )
        return exact


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


def _installed_direct_url(distribution: str) -> Optional[dict[str, Any]]:
    """Return the PEP 610 ``direct_url.json`` of *distribution*, or ``None``.

    ``None`` means it was installed from an index (or is not installed at all).
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        text = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


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
