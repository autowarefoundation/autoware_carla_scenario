"""A local cache of HD map repositories, filled by ``git``.

The cache is the point of the feature, not an optimisation: a map is edited,
previewed and swept over and over, and none of that should touch the network.
:meth:`GitMapCache.checkout` answers from disk whenever the files it was asked
for are already there; ``refresh=True`` is the one way to make it look again.

Two details make this workable against a real HD map repository. **LFS is
skipped** (``GIT_LFS_SKIP_SMUDGE=1``) so a checkout materialises pointer files,
and only the patterns asked for are pulled -- the repository this was written
against keeps a 22 MB point cloud and a 466 MB ``.usdz`` beside a 2 MB ``.osm``.
And **blobs are fetched lazily** (``--filter=blob:none``) into a **sparse**
checkout, which is also what makes :meth:`GitMapCache.list_files` cheap enough
to browse a repository with.

The root is ``~/autoware_data/maps``, where Autoware's own ``demo_artifacts``
role downloads map datasets to, unpacked at their repository-relative paths.
That is not a coincidence to be tidied away: a machine that has run the Autoware
setup already has the map, and :meth:`GitMapCache.provisioned` finds it there.
A map this cache cloned itself lands under ``.repos/`` in the same root.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .source import MapSource

logger = logging.getLogger(__name__)

__all__ = [
    "AUTOWARE_DATA_DIR_ENV",
    "CACHE_ROOT_ENV",
    "CachedRepo",
    "GitMapCache",
    "MapCacheError",
    "is_lfs_pointer",
    "map_root",
]

#: Environment variable overriding where maps are kept, cloned or downloaded.
CACHE_ROOT_ENV = "AUTOWARE_CARLA_SCENARIO_MAP_CACHE"

#: Environment variable naming Autoware's data directory, whose ``maps``
#: subdirectory is where its ``demo_artifacts`` role downloads map datasets.
AUTOWARE_DATA_DIR_ENV = "AUTOWARE_DATA_DIR"

#: Autoware's own default for that directory.
DEFAULT_AUTOWARE_DATA_DIR = "~/autoware_data"

#: Repositories this cache cloned, kept apart from the downloaded map trees
#: that share the root.  Dot-prefixed so it does not read as a map.
REPOS_SUBDIR = ".repos"

#: Artefacts generated from a map -- an OpenDRIVE fetched from CARLA -- kept
#: out of both the downloaded trees and the git checkouts.
DERIVED_SUBDIR = ".derived"

#: How long a git call may take before it is abandoned.  A clone of a map
#: repository is seconds; anything approaching this is a hung transport.
_GIT_TIMEOUT_SECONDS = 600.0

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class MapCacheError(RuntimeError):
    """A repository could not be cloned, updated or read."""


def map_root() -> Path:
    """Return the directory HD maps are kept in.

    ``$AUTOWARE_DATA_DIR/maps``, defaulting to ``~/autoware_data/maps`` exactly
    as Autoware's ``demo_artifacts`` role does, so that a map its setup already
    downloaded is a map this framework already has.
    :data:`CACHE_ROOT_ENV` overrides the whole path for a project that keeps its
    maps somewhere else.
    """
    configured = os.environ.get(CACHE_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_dir = (
        os.environ.get(AUTOWARE_DATA_DIR_ENV, "").strip() or DEFAULT_AUTOWARE_DATA_DIR
    )
    return (Path(data_dir).expanduser() / "maps").resolve()


@dataclass(frozen=True)
class CachedRepo:
    """One cloned repository in the cache.

    Attributes:
        source: What was asked for.  Its ``path`` is not part of the identity:
            one clone serves every map directory in the repository.
        root: The cache entry holding the checkout.
        worktree: The sparse checkout itself.
        commit: The commit the checkout is at, or ``""`` when not recorded.
        materialised: The repository-relative directory currently checked out,
            or ``None`` when nothing has been.
    """

    source: MapSource
    root: Path
    worktree: Path
    commit: str = ""
    materialised: Optional[str] = None


class GitMapCache:
    """Clones map repositories into a local directory, and reuses them."""

    def __init__(self, root: Optional[Path] = None) -> None:
        """Store maps under *root*, defaulting to :func:`map_root`."""
        self.root = Path(root) if root is not None else map_root()

    # -- addressing -----------------------------------------------------

    def entry_dir(self, source: MapSource) -> Path:
        """Return the cache entry for *source*'s repository and ref.

        The slug is readable so that a person can find a map in the cache by
        eye; the digest after it is what actually keeps two repositories apart,
        since sanitising a URL is not injective.
        """
        identity = f"{source.repo_url}\n{source.ref}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
        slug = _SLUG_UNSAFE.sub("-", f"{source.repo_name}-{source.ref or 'HEAD'}")
        return self.root / REPOS_SUBDIR / f"{slug.strip('-').lower()}-{digest}"

    def derived_dir(self, path: str) -> Path:
        """Return where artefacts generated from the map at *path* are kept."""
        return self.root / DERIVED_SUBDIR / (path or "root")

    def provisioned(self, source: MapSource) -> Optional[Path]:
        """Return *source*'s map directory if it is already on disk, unpacked.

        A downloader that keeps repository-relative paths -- ``hf download
        --local-dir``, which is what Autoware's setup runs -- leaves a map at
        exactly ``<root>/<path>``.  Finding one there means no git command runs
        and the ref is not consulted: what the machine has is what gets used.
        """
        if not source.path:
            return None
        directory = self.root / source.path
        if directory.is_file():
            directory = directory.parent
        if not directory.is_dir():
            return None
        return directory if any(directory.glob("*.osm")) else None

    # -- reading --------------------------------------------------------

    def peek(self, source: MapSource) -> Optional[CachedRepo]:
        """Return the cache entry for *source* if it has been cloned."""
        entry = self.entry_dir(source)
        worktree = entry / "repo"
        if not (worktree / ".git").exists():
            return None
        meta = self._read_meta(entry)
        return CachedRepo(
            source=source,
            root=entry,
            worktree=worktree,
            commit=str(meta.get("commit", "")),
            materialised=(
                str(meta["materialised"])
                if meta.get("materialised") is not None
                else None
            ),
        )

    def list_files(self, source: MapSource, *, refresh: bool = False) -> list[str]:
        """Return every file path in *source*'s repository, at its ref.

        Reads the tree rather than the working directory, so it sees the whole
        repository without materialising any of it -- which is what makes
        browsing a repository cost one clone of its commit and tree objects.
        """
        repo = self.checkout(source, include=(), materialise=False, refresh=refresh)
        out = self._git(
            repo.worktree, "ls-tree", "-r", "--name-only", "HEAD", capture=True
        )
        return [line for line in out.splitlines() if line]

    # -- writing --------------------------------------------------------

    def checkout(
        self,
        source: MapSource,
        *,
        include: Sequence[str] = (),
        materialise: bool = True,
        refresh: bool = False,
    ) -> CachedRepo:
        """Return *source*'s repository on disk, cloning it if needed.

        Args:
            source: The repository and ref to have locally.
            include: Gitignore-style patterns, relative to the repository root,
                whose LFS content is needed.  Everything else stays a pointer.
            materialise: Check the files out.  Left off, only the commit and
                tree objects are needed, which is enough to list what it holds.
            refresh: Fetch the ref again even when the cache already has it.

        Raises:
            MapCacheError: If git is unavailable, or the clone or fetch failed.
        """
        cached = self.peek(source)
        if cached is None:
            cached = self._clone(source)
        elif refresh:
            self._fetch(cached)
            # Re-read: the fetch moved the checkout, and `cached` still
            # describes the commit it was at before.
            cached = self.peek(source) or cached

        if materialise:
            self._materialise(cached, include=include, refresh=refresh)
        return cached

    # -- git ------------------------------------------------------------

    def _clone(self, source: MapSource) -> CachedRepo:
        """Clone *source* into a staging directory and move it into place."""
        entry = self.entry_dir(source)
        entry.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{entry.name}.", dir=str(entry.parent))
        )
        worktree = staging / "repo"
        try:
            if source.ref:
                self._fetch_ref(worktree, source.repo_url, source.ref)
            else:
                self._git(
                    None,
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--depth",
                    "1",
                    source.repo_url,
                    str(worktree),
                )

            commit = self._git(worktree, "rev-parse", "HEAD", capture=True).strip()
            self._write_meta(staging, source, commit)
            try:
                staging.replace(entry)
            except OSError:
                # Another process cloned the same repository while we did; its
                # copy is as good as ours, so keep whichever landed first.
                if not (entry / "repo" / ".git").exists():
                    raise
                logger.debug("Discarding a concurrent clone of %s", source.repo_url)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        cached = self.peek(source)
        if cached is None:
            raise MapCacheError(f"Could not clone {source.repo_url}.")
        logger.info(
            "Cloned %s@%s into %s", source.repo_url, source.ref or "HEAD", cached.root
        )
        return cached

    def _fetch_ref(self, worktree: Path, repo_url: str, ref: str) -> None:
        """Create a repository at *worktree* holding exactly *ref*.

        Fetching a named ref into an empty repository, rather than cloning with
        ``--branch``, is what lets a map be pinned: ``--branch`` takes a branch
        or a tag and refuses a commit hash, and a commit hash is the only ref
        that fixes what a scenario runs against.  One fetch of one ref also
        transfers less than a clone does, so the pinned path is the cheap one.
        """
        worktree.mkdir(parents=True, exist_ok=True)
        self._git(worktree, "init", "--quiet", ".")
        self._git(worktree, "remote", "add", "origin", repo_url)
        # What `clone --filter` would have configured for us.  Without it git
        # does not know the missing blobs can be fetched on demand.
        self._git(worktree, "config", "remote.origin.promisor", "true")
        self._git(worktree, "config", "remote.origin.partialclonefilter", "blob:none")
        self._git(
            worktree, "fetch", "--depth", "1", "--filter=blob:none", "origin", ref
        )
        self._git(worktree, "checkout", "--detach", "FETCH_HEAD")

    def _fetch(self, cached: CachedRepo) -> None:
        """Update an existing checkout to the tip of its ref.

        A pinned source has no tip to move to, so refreshing one re-checks that
        the checkout is the commit it claims rather than fetching again.
        """
        source = cached.source
        if source.pinned and cached.commit == source.ref:
            return
        ref = source.ref or "HEAD"
        self._git(cached.worktree, "fetch", "--depth", "1", "origin", ref)
        self._git(cached.worktree, "checkout", "--detach", "FETCH_HEAD")
        commit = self._git(cached.worktree, "rev-parse", "HEAD", capture=True).strip()
        self._write_meta(cached.root, source, commit)
        logger.info("Refreshed %s@%s to %s", source.repo_url, ref, commit[:10])

    def _materialise(
        self, cached: CachedRepo, *, include: Sequence[str], refresh: bool
    ) -> None:
        """Check out *source*'s directory and pull the LFS content asked for.

        Nothing runs when the checkout is already the one being asked for and
        its content is real rather than pointers.  That matters because this is
        on the path of every resolve -- every Download click, every sweep, every
        run -- and ``git checkout`` re-stats the whole index each time it is
        called, for a working tree that has not moved.
        """
        source = cached.source
        worktree = cached.worktree
        wanted = [pattern for pattern in include if pattern]

        if not refresh and cached.materialised == source.path:
            if not wanted or self._content_present(worktree, wanted):
                return

        if source.path:
            self._git(worktree, "sparse-checkout", "set", "--no-cone", source.path)
        self._git(worktree, "checkout")

        if wanted and (refresh or not self._content_present(worktree, wanted)):
            self._git(worktree, "lfs", "pull", f"--include={','.join(wanted)}")
        self._record(cached.root, materialised=source.path)

    @staticmethod
    def _content_present(worktree: Path, wanted: Sequence[str]) -> bool:
        """Whether every matched file holds real content rather than a pointer."""
        return all(
            not is_lfs_pointer(path)
            for pattern in wanted
            for path in worktree.glob(pattern)
        )

    def _git(self, cwd: Optional[Path], *args: str, capture: bool = False) -> str:
        """Run one git command, with LFS smudging off.

        Raises:
            MapCacheError: If git is missing, timed out, or exited non-zero.
        """
        env = dict(os.environ)
        # Every checkout in this cache is deliberately pointer-only; the content
        # that is actually wanted is pulled by pattern afterwards.
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            completed = subprocess.run(  # noqa: S603 -- fixed program, argv list
                ["git", *args],
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MapCacheError(
                "git is not installed, so a map repository cannot be cloned."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MapCacheError(f"git {args[0]} timed out.") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            lines = detail.splitlines()
            raise MapCacheError(
                f"git {args[0]} failed: {lines[-1] if lines else 'no output'}"
            )
        return completed.stdout if capture else ""

    # -- metadata -------------------------------------------------------

    @staticmethod
    def _read_meta(entry: Path) -> dict[str, Any]:
        """Return the entry's recorded metadata, or an empty mapping."""
        try:
            raw = (entry / "meta.json").read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @classmethod
    def _write_meta(cls, entry: Path, source: MapSource, commit: str) -> None:
        """Record what the entry holds, so a cache hit can describe itself."""
        cls._record(
            entry,
            repo_url=source.repo_url,
            ref=source.ref,
            commit=commit,
            fetched_at=time.time(),
            materialised=None,
        )

    @classmethod
    def _record(cls, entry: Path, **fields: Any) -> None:
        """Merge *fields* into the entry's metadata."""
        meta = cls._read_meta(entry)
        meta.update(fields)
        (entry / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


#: The first bytes of a git-lfs pointer file, which is what a checkout made with
#: ``GIT_LFS_SKIP_SMUDGE=1`` leaves in place of tracked content.
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """Whether *path* holds an LFS pointer rather than the real content."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False
