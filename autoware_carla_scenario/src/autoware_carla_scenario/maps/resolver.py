"""From a map source to the files on disk a scenario actually reads.

Everything upstream of this module talks about a repository; everything
downstream -- the Lanelet2 loader, the sweeper, the preview, the runtime --
takes paths.  This is the seam, and it is deliberately the only one: a document
that names a remote map and a document that names local files reach the same
loader by the same route.

What a map directory is expected to hold is what Autoware writes beside a map:

* ``lanelet2_map.osm`` -- the Lanelet2 map.  Any single ``.osm`` in the
  directory is accepted, since the name is a convention rather than a rule.
* ``map_projector_info.yaml`` -- optional, and read by
  :mod:`~autoware_carla_scenario.coordinate.projection` straight off disk.
* an OpenDRIVE ``.xodr`` -- optional.  A map published for Autoware usually has
  none, because the road network CARLA simulates comes from the CARLA asset
  itself; :mod:`~autoware_carla_scenario.maps.opendrive` fetches that one from a
  running server and leaves it in the map root's ``derived`` directory, which is
  where this module then finds it.

A map already unpacked under the map root -- which is what Autoware's setup
leaves behind for ``carla-ue5-maps`` -- is used as it stands, without cloning
anything.  See :meth:`~autoware_carla_scenario.maps.cache.GitMapCache.provisioned`.

That shortcut is off for a *pinned* source.  A directory on disk carries no
record of which revision it was downloaded at, so handing it back for a source
that named an exact commit would quietly answer a question about one revision
with the bytes of another -- which is the one thing pinning exists to prevent.
A pinned map is therefore always resolved through git, where the commit is
checked rather than assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .cache import CachedRepo, GitMapCache, is_lfs_pointer
from .source import MapSource

logger = logging.getLogger(__name__)

__all__ = [
    "MAP_FILE_PATTERNS",
    "MapResolutionError",
    "ResolvedMap",
    "cached_map",
    "pin_source",
    "resolve_map",
]

#: Which files in a map directory are worth pulling out of LFS.  Deliberately
#: not everything: the map directory this was written against also holds a 22 MB
#: point cloud that nothing in this framework reads.
MAP_FILE_PATTERNS = ("*.osm", "*.xodr", "*.yaml", "*.yml")

#: What Autoware names a Lanelet2 map, preferred when a directory holds several.
PREFERRED_LANELET2_NAME = "lanelet2_map.osm"


class MapResolutionError(RuntimeError):
    """A map source was cloned but does not hold a usable map."""


@dataclass(frozen=True)
class ResolvedMap:
    """One remote map, as local paths.

    Attributes:
        source: The source that was resolved.
        name: What to call the map -- its directory name, e.g. ``Town10HD_Opt``.
        directory: The map directory inside the checkout.
        lanelet2_path: The Lanelet2 ``.osm``.
        xodr_path: The OpenDRIVE file, if the repository ships one or one has
            been fetched from CARLA into :attr:`derived`.
        xodr_is_derived: Whether that file was fetched from CARLA rather than
            shipped by the repository.
        projector_info_path: ``map_projector_info.yaml``, if present.
        derived: Where artefacts generated from this map are written.
        commit: The commit the checkout is at, or ``""`` when the map was
            already unpacked under the map root rather than cloned.
        provisioned: Whether the map was found already on disk -- downloaded by
            Autoware's own setup -- rather than cloned by this framework.
    """

    source: MapSource
    name: str
    directory: Path
    lanelet2_path: Path
    xodr_path: Optional[Path]
    projector_info_path: Optional[Path]
    derived: Path
    xodr_is_derived: bool = False
    commit: str = ""
    provisioned: bool = False

    def derived_xodr(self, name: str = "") -> Path:
        """Return where an OpenDRIVE taken from CARLA for this map is written.

        The one place the derived file's name is decided.  Both the config
        layer, which tells a run where to put the OpenDRIVE it captures, and
        :func:`~autoware_carla_scenario.maps.opendrive.ensure_xodr`, which
        fetches one ahead of time, ask here -- if they disagreed, the editor
        would cache a file the runner then failed to find.
        """
        return self.derived / f"{name or self.name}.xodr"


def _include_patterns(path: str) -> tuple[str, ...]:
    """Return the LFS include patterns covering the map directory *path*."""
    prefix = f"{path}/" if path else ""
    return tuple(f"{prefix}{pattern}" for pattern in MAP_FILE_PATTERNS)


def _find_lanelet2(directory: Path, source: MapSource) -> Path:
    """Return the Lanelet2 map in *directory*.

    Raises:
        MapResolutionError: If there is no ``.osm``, or more than one and none
            of them is named the way Autoware names it.
    """
    if source.path.endswith(".osm"):
        named = directory / Path(source.path).name
        if named.is_file():
            return named

    candidates = sorted(p for p in directory.glob("*.osm") if p.is_file())
    if not candidates:
        raise MapResolutionError(
            f"{source.uri} has no Lanelet2 (.osm) file in {source.path or '/'}."
        )
    if len(candidates) == 1:
        return candidates[0]
    preferred = directory / PREFERRED_LANELET2_NAME
    if preferred.is_file():
        return preferred
    names = ", ".join(p.name for p in candidates)
    raise MapResolutionError(
        f"{source.uri} holds several Lanelet2 files ({names}) and none is named "
        f"{PREFERRED_LANELET2_NAME}. Point the source at one of them."
    )


def _find_xodr(directory: Path, derived: Path) -> tuple[Optional[Path], bool]:
    """Return the map's OpenDRIVE and whether it was derived rather than shipped.

    The difference decides what CARLA is asked to do with it.  An OpenDRIVE the
    repository ships describes roads CARLA does not have -- it is installed into
    the simulator, which is what
    :meth:`~...scenario_runner.ScenarioRunner.load_map_by_overwriting_xodr` is
    for.  One derived into the cache came *out* of CARLA in the first place, so
    installing it back would be a no-op that also demands an environment
    variable pointing into the CARLA installation.
    """
    for candidate in sorted(directory.glob("*.xodr")):
        if candidate.is_file():
            return candidate, False
    for candidate in sorted(derived.glob("*.xodr")):
        if candidate.is_file():
            return candidate, True
    return None, False


def resolve_map(
    source: "str | MapSource",
    *,
    cache: Optional[GitMapCache] = None,
    refresh: bool = False,
) -> ResolvedMap:
    """Return the local files for *source*, cloning it only if needed.

    Args:
        source: A map source URI, or an already-parsed :class:`MapSource`.
        cache: The cache to read and fill.  Defaults to the shared one.
        refresh: Fetch the repository again even when it is already cached.

    Returns:
        The resolved map.

    Raises:
        MapSourceError: If *source* is not a usable URI.
        MapCacheError: If the repository could not be cloned or updated.
        MapResolutionError: If the repository holds no Lanelet2 map there.
    """
    ask = _Ask.of(source, cache)

    directory = None if (refresh or ask.source.pinned) else ask.provisioned()
    if directory is not None:
        return _describe(ask, directory, provisioned=True)

    repo = ask.store.checkout(
        ask.at_directory,
        include=_include_patterns(ask.directory_path),
        refresh=refresh,
    )
    directory = ask.in_worktree(repo)
    if not directory.is_dir():
        raise MapResolutionError(
            f"{ask.source.uri} does not name a directory in the repository."
        )
    return _describe(ask, directory, commit=repo.commit)


def cached_map(
    source: "str | MapSource", *, cache: Optional[GitMapCache] = None
) -> Optional[ResolvedMap]:
    """Return *source*'s map if it is already on disk, without fetching it.

    The editor re-renders on every keystroke and cannot spend a clone on each
    one, so this is what it asks: describe the map if the machine already has
    it, and otherwise say nothing.  Fetching stays an explicit action.

    Returns:
        The resolved map, or ``None`` if it is not cached, not downloaded, or
        cached only as a git-lfs pointer.

    Raises:
        MapSourceError: If *source* is not a usable URI.
    """
    ask = _Ask.of(source, cache)

    directory = None if ask.source.pinned else ask.provisioned()
    provisioned = directory is not None
    commit = ""
    if directory is None:
        repo = ask.store.peek(ask.at_directory)
        if repo is None:
            return None
        if ask.source.pinned and repo.commit != ask.source.ref:
            # The cache holds this repository, but at another revision. Saying
            # "not cached" sends the caller to resolve_map, which checks it out.
            return None
        commit = repo.commit
        directory = ask.in_worktree(repo)
    if not directory.is_dir():
        return None

    try:
        lanelet2_path = _find_lanelet2(directory, ask.source)
    except MapResolutionError:
        return None
    if is_lfs_pointer(lanelet2_path):
        # Checked out but never pulled: the file is a pointer, and handing it to
        # a Lanelet2 parser reports a broken map rather than a missing fetch.
        return None
    return _describe(
        ask,
        directory,
        commit=commit,
        provisioned=provisioned,
        lanelet2_path=lanelet2_path,
    )


def pin_source(
    source: "str | MapSource",
    *,
    cache: Optional[GitMapCache] = None,
    refresh: bool = False,
) -> str:
    """Return *source* rewritten to name the exact commit it resolves to now.

    A scenario that says ``@splatsim`` runs against whatever that branch holds
    on the day it runs; the same scenario pinned to the commit behind that
    branch runs against the same map for as long as the repository exists.
    Pinning is therefore what makes a result comparable to an older one, and
    this is the operation the editor's "Pin" button and a release script both
    call.

    Args:
        source: A map source URI, or an already-parsed :class:`MapSource`.
        cache: The cache to read and fill.  Defaults to the shared one.
        refresh: Fetch the repository first, so the pin names the current tip
            rather than whatever revision happens to be cached.

    Returns:
        The canonical URI, with the ref replaced by a full commit hash.

    Raises:
        MapResolutionError: If the map resolved to a copy with no known commit
            -- one already unpacked on disk, which records no revision.
    """
    parsed = source if isinstance(source, MapSource) else MapSource.parse(source)
    if parsed.pinned and not refresh:
        return parsed.uri
    resolved = resolve_map(parsed, cache=cache, refresh=refresh)
    if not resolved.commit:
        raise MapResolutionError(
            f"{parsed.uri} resolved to a copy already on disk, which records no "
            "revision, so it cannot be pinned. Refresh it to clone the "
            "repository and read the commit from git."
        )
    return parsed.at(resolved.commit).uri


@dataclass(frozen=True)
class _Ask:
    """One request to resolve a map, addressed.

    Both entry points below start by working out the same four things, and they
    have to agree on all of them -- which directory in the repository, which
    cache entry, where derived files go -- or a map described by one would not
    be the map fetched by the other.
    """

    source: MapSource
    store: GitMapCache
    directory_path: str
    derived: Path

    @classmethod
    def of(cls, source: "str | MapSource", cache: Optional[GitMapCache]) -> "_Ask":
        """Return the addressed form of *source*."""
        parsed = source if isinstance(source, MapSource) else MapSource.parse(source)
        store = cache or GitMapCache()
        return cls(
            source=parsed,
            store=store,
            directory_path=parsed.directory,
            derived=store.derived_dir(parsed.directory),
        )

    @property
    def at_directory(self) -> MapSource:
        """The source, pointed at the map's directory rather than a file."""
        return self.source.with_path(self.directory_path)

    def provisioned(self) -> Optional[Path]:
        """The already-unpacked map directory, if there is one."""
        return self.store.provisioned(self.at_directory)

    def in_worktree(self, repo: CachedRepo) -> Path:
        """Where the map sits inside *repo*'s checkout."""
        return (
            repo.worktree / self.directory_path
            if self.directory_path
            else repo.worktree
        )


def _describe(
    ask: _Ask,
    directory: Path,
    *,
    commit: str = "",
    provisioned: bool = False,
    lanelet2_path: Optional[Path] = None,
) -> ResolvedMap:
    """Return the map held in *directory*.

    *lanelet2_path* lets a caller that has already found the ``.osm`` say so,
    rather than have the directory scanned for it twice.
    """
    source = ask.source
    derived = ask.derived
    projector_info = directory / "map_projector_info.yaml"
    xodr_path, xodr_is_derived = _find_xodr(directory, derived)
    return ResolvedMap(
        source=source,
        name=directory.name,
        directory=directory,
        lanelet2_path=lanelet2_path or _find_lanelet2(directory, source),
        xodr_path=xodr_path,
        projector_info_path=projector_info if projector_info.is_file() else None,
        derived=derived,
        xodr_is_derived=xodr_is_derived,
        commit=commit,
        provisioned=provisioned,
    )
