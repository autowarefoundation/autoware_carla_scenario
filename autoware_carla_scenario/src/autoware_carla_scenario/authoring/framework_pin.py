"""Work out how an exported scenario package should depend on this framework.

An exported package is only reproducible if the framework it runs on is pinned
to something immutable.  This module resolves exactly one of:

* an **exact release version** -- ``autoware-carla-scenario==X.Y.Z``; or
* an **exact commit** -- the repository URL plus a full commit SHA, for when
  the scenario was authored against an unreleased snapshot; or
* a **local path**, which is only ever produced when the caller explicitly asks
  for a development export.

A commit or a path pin needs the framework's own source tree: a directory
whose ``pyproject.toml`` names ``autoware-carla-scenario``.  Being inside some
git work tree is not enough -- a framework installed from a wheel into a
project's ``.venv`` sits inside *that project's* repository (#1976).

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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from .requirements_pin import canonical_name

__all__ = [
    "DISTRIBUTION",
    "Pin",
    "PinResolutionError",
    "framework_source_root",
    "resolve_framework_pin",
]

#: Distribution name of the framework.
DISTRIBUTION = "autoware-carla-scenario"

#: Overrides pin resolution with an exact released version.
VERSION_ENV = "SCENARIO_EXPORT_FRAMEWORK_VERSION"

#: Overrides the repository URL recorded for a commit pin (useful when the
#: checkout's ``origin`` is a fork or an SSH remote the consumer cannot reach).
REPOSITORY_ENV = "SCENARIO_EXPORT_FRAMEWORK_REPOSITORY"

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
#: The scp-like form ``git@host:owner/repo.git``, which has no scheme.  A
#: ``scheme://`` URL is split by :func:`_split_url` instead.
_SCP_URL_PATTERN = re.compile(r"^git@([^:/]+)[:/](.+?)(?:\.git)?/?$")

#: Ports implied by an http(s) scheme, dropped when a URL spells them out.
_DEFAULT_PORTS = {"http": "80", "https": "443"}


class PinResolutionError(RuntimeError):
    """Raised when no immutable pin for the framework can be determined."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


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
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _split_url(url: str) -> tuple[str, str, str]:
    """Split ``scheme://netloc/path?query#fragment`` into scheme, netloc, path.

    By hand rather than with :func:`urllib.parse.urlsplit`, which raises
    ``ValueError`` on a malformed IPv6 host (``https://[::1/x``): a remote
    git accepted must not stop an export, nor reach its error message.
    """
    scheme, _sep, rest = url.partition("://")
    end = len(rest)
    for mark in "/?#":
        index = rest.find(mark)
        if 0 <= index < end:
            end = index
    netloc = rest[:end]
    path = rest[end:]
    for mark in "?#":
        index = path.find(mark)
        if index >= 0:
            path = path[:index]
    return scheme, netloc, path


def _split_host_port(host_port: str) -> tuple[str, str]:
    """Split ``host[:port]`` (or ``[v6]:port``) into its host and port."""
    if host_port.startswith("["):
        end = host_port.find("]")
        if end >= 0:
            rest = host_port[end + 1 :]
            port = rest[1:] if rest.startswith(":") else ""
            return host_port[: end + 1], port
    host, _sep, port = host_port.partition(":")
    return host, port


def _strip_repo_path(path: str) -> str:
    """Drop a trailing ``/`` and ``.git`` from a repository path."""
    return path.removesuffix("/").removesuffix(".git")


def normalize_repository_url(url: str) -> str:
    """Return *url* in a form ``uv`` (and anyone else) can clone.

    The result is written into an exported package that is meant to be shared,
    so credentials never survive:

    * ``git@host:owner/repo.git`` becomes ``https://host/owner/repo``;
    * ``ssh://user@host:2222/owner/repo.git`` becomes ``https://host/owner/repo``
      -- the user and the SSH port mean nothing over HTTPS;
    * ``http(s)://`` URLs lose their userinfo (a token or ``user:password``),
      query and fragment, keep a non-default port, and lose a ``.git`` suffix;
    * any other ``scheme://`` URL loses its userinfo and is otherwise kept.

    Anything else (a local path) is returned stripped.
    """
    stripped = url.strip()
    match = _SCP_URL_PATTERN.match(stripped)
    if match:
        host, path = match.groups()
        return f"https://{host}/{path}"
    if "://" not in stripped:
        return stripped

    raw_scheme, raw_netloc, raw_path = _split_url(stripped)
    scheme = raw_scheme.lower()
    host_port = raw_netloc.rpartition("@")[2]
    host, port = _split_host_port(host_port)
    if scheme in ("ssh", "git+ssh"):
        path = _strip_repo_path(raw_path).lstrip("/")
        return f"https://{host}/{path}"
    if scheme in ("http", "https"):
        netloc = host if port in ("", _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
        return f"{scheme}://{netloc}{raw_path.removesuffix('.git')}"
    if host_port == raw_netloc:
        return stripped
    return stripped.replace(raw_netloc, host_port, 1)


def framework_source_root() -> Path:
    """Return the directory holding the framework's own ``pyproject.toml``.

    For a source checkout this is ``<repo>/autoware_carla_scenario``.  For an
    installed wheel it is the directory above ``site-packages``
    (``.../lib/pythonX.Y``), which holds no framework ``pyproject.toml``, so
    :func:`_is_framework_project` refuses it and no commit or path pin is made
    from it -- even when it lies inside some other project's git work tree, as
    a project's ``.venv`` does (#1976).
    """
    # What getattr(autoware_carla_scenario, "__file__", None) reads: a
    # package without the attribute (ImportError here) or with it None (a
    # namespace package) has no source file to start from.
    try:
        from .. import __file__ as module_file  # noqa: PLC0415
    except ImportError:  # pragma: no cover - a loader that sets no __file__
        return Path.cwd()
    if module_file is None:  # pragma: no cover - namespace package edge case
        return Path.cwd()
    # <root>/src/autoware_carla_scenario/__init__.py -> <root>
    return Path(module_file).resolve().parents[2]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _installed_version() -> Optional[str]:
    """Return the installed version of the framework, or ``None``."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed
        return None


def _project_name(pyproject: Path) -> Optional[str]:
    """Return ``[project].name`` of *pyproject*, or ``None`` if it has none.

    A file that cannot be read or is not TOML has no name either.
    """
    # tomllib landed in 3.11, and 3.10 is still the floor of the supported
    # range; imported here because Codon has neither (typecheck/library.py:
    # UNCALLED).
    if sys.version_info >= (3, 11):
        import tomllib  # noqa: PLC0415
    else:  # pragma: no cover - only taken on 3.10
        import tomli as tomllib  # noqa: PLC0415

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    return name if isinstance(name, str) else None


def _is_framework_project(source_root: Path) -> bool:
    """Return whether *source_root* is the framework's own source tree.

    That is, whether it holds a ``pyproject.toml`` whose ``[project].name``
    normalises to :data:`DISTRIBUTION`.  Only such a directory may be pinned
    by commit or by path: the ``lib/pythonX.Y`` directory of a wheel install
    may well be inside a git work tree -- the user's own project, holding its
    ``.venv`` -- and pinning that would point the export at the wrong
    repository (#1976).
    """
    pyproject = source_root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    if TYPE_CHECKING:
        # tomllib and tomli have no model (typecheck/library.py: UNCALLED).
        name: Optional[str] = None
    else:
        name = _project_name(pyproject)
    if name is None:
        return False
    return canonical_name(name) == DISTRIBUTION


def _version_problem(version: str) -> Optional[str]:
    """Return why *version* is not a released version, or ``None``.

    It must be a valid PEP 440 version without a local label
    (``1.2.3+local``), which no package index serves.
    """
    # packaging has no model (typecheck/library.py: UNCALLED).
    from packaging.version import InvalidVersion, Version  # noqa: PLC0415

    try:
        parsed = Version(version)
    except InvalidVersion:
        return f"{version!r} is not a valid version (PEP 440)"
    if parsed.local is not None:
        return (
            f"{version!r} has a local version label (+{parsed.local}), which "
            "no package index serves"
        )
    return None


def _resolve_git_pin(source_root: Path) -> Optional[Pin]:
    """Return a commit pin for *source_root*, or ``None`` when it is not a checkout.

    Not a checkout of the framework, that is: *source_root* must be the
    framework's own source tree (:func:`_is_framework_project`), not merely a
    directory inside some git work tree.
    """
    if not _is_framework_project(source_root):
        return None
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
        subdirectory=subdirectory if subdirectory else None,
        version=_installed_version(),
        warnings=tuple(warnings),
    )


def resolve_framework_pin(*, dev_mode: bool = False) -> Pin:
    """Determine how the exported package should depend on the framework.

    *dev_mode* short-circuits everything and pins the local checkout, which is
    what you want while iterating on the framework and the scenario together --
    and never what you want for a package somebody else will run.  Otherwise:

    1. :data:`VERSION_ENV`, when the framework has a published release that the
       package should track exactly.  Surrounding whitespace is ignored, and a
       blank value counts as unset.
    2. The framework's own git checkout, pinned to ``HEAD``'s commit SHA.

    A commit or a path pin is only made from the framework's own source tree
    (a ``pyproject.toml`` naming ``autoware-carla-scenario``), never from a
    wheel install's directory, whichever git work tree that sits in.  A
    framework installed from a wheel is not pinned to its installed version
    either: a wheel built from an unreleased commit carries the version of
    the last release, so ``==<it>`` would silently resolve a different
    framework.  Such an install needs :data:`VERSION_ENV`.

    Args:
        dev_mode: Depend on the local framework checkout by path.  Packages
            exported this way are not portable and are marked as such in the
            manifest and the README.

    Returns:
        The resolved :class:`Pin`.

    Raises:
        PinResolutionError: If :data:`VERSION_ENV` is not a valid version, if
            no immutable pin can be determined and *dev_mode* is not set, or
            if *dev_mode* is set and the framework is not a source checkout.
    """
    source_root = framework_source_root()

    if dev_mode:
        if not _is_framework_project(source_root):
            raise PinResolutionError(
                "A development export depends on the framework's source "
                f"checkout by path, but {source_root} is not one (it has no "
                "pyproject.toml naming autoware-carla-scenario): the framework "
                "is installed from a wheel. Install it from a checkout "
                "(editable), or export without development mode."
            )
        return Pin(
            kind="path",
            path=str(source_root),
            version=_installed_version(),
            warnings=tuple(
                [
                    "Development export: the package depends on the local path "
                    f"{source_root}, so it will not resolve on another machine. "
                    "Re-export without development mode before sharing it."
                ]
            ),
        )

    version = (os.environ.get(VERSION_ENV) or "").strip()
    if version:
        if TYPE_CHECKING:
            # packaging has no model (typecheck/library.py: UNCALLED).
            problem: Optional[str] = None
        else:
            problem = _version_problem(version)
        if problem is not None:
            raise PinResolutionError(
                f"{VERSION_ENV} must be an exact released version of "
                f"autoware-carla-scenario (such as 1.2.3): {problem}."
            )
        return Pin(kind="version", version=version)

    git_pin = _resolve_git_pin(source_root)
    if git_pin is not None:
        return git_pin

    raise PinResolutionError(
        "Cannot pin autoware-carla-scenario to an exact version or commit. "
        f"Set {VERSION_ENV} to a released version (the way to export from a "
        "framework installed from a wheel), run the export from a git "
        "checkout of the framework, or export in development mode (which "
        "produces a non-portable package)."
    )
