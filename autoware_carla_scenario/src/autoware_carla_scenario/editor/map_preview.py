"""Evaluate lanelet constraints against a real Lanelet2 map, and draw the result.

Two things make the constraint builder usable rather than theoretical: a match
count, and seeing *where* the matches are.  Both come from the framework's own
sweeper -- :func:`~autoware_carla_scenario.sweeper.constraints.parse_constraint`
and :func:`~autoware_carla_scenario.sweeper.constraints.find_matching_lanelets`
-- so the preview cannot disagree with what a sweep will actually do.

Map loading is deliberately opt-in and cached.  Lanelet2 is a heavy native
dependency and a map takes seconds to parse, so the editor never loads one
until asked, and an unreadable map degrades to "constraints only" rather than
breaking the page.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import ConstraintNode, LaneletSlot, ScenarioDocument
from ..authoring.validator import MAP_EXCLUSION_REF
from ..maps import MapCacheError, MapPaths, MapSource, MapSourceError, map_root
from ..maps.config import resolve_map_paths
from ..maps.resolver import MapResolutionError

logger = logging.getLogger(__name__)

__all__ = [
    "MAP_ROOTS_ENV",
    "PreviewResult",
    "clear_cache",
    "evaluate_slot",
    "is_map_loaded",
    "MapStatus",
    "lanelet2_source",
    "map_paths",
    "map_status",
    "materialize_constraints",
    "missing_map_reason",
]

#: Parsed maps kept in memory, newest last.  Parsing is measured in seconds, so
#: one map has to stay resident to make the preview usable; more than a couple
#: only pins native map objects for the life of the process, and a session edits
#: one map at a time.
_MAP_CACHE_SIZE = 2

_MAP_CACHE: "OrderedDict[tuple[str, str, str], _LoadedMap]" = OrderedDict()

#: Distinct constraint trees whose matches are remembered per map.  A session
#: writes a bounded number of them -- one per edit to a search -- and each is a
#: list of ids, so the cap is only there to stop a very long session growing
#: without end.
_MATCH_CACHE_SIZE = 64


@dataclass
class _LoadedMap:
    """A parsed map plus the derived structures the preview reuses."""

    lanelet_map: Any
    routing_graph: Any
    lanelet_count: int
    #: Matches by constraint tree.  Evaluating a search scans every lanelet in
    #: the map, and the places panel re-renders on *every* edit -- renaming an
    #: actor would otherwise re-scan a city to arrive at the same ids.  Kept
    #: here rather than in a cache of its own because that is what makes it
    #: correct: the answer belongs to this parse of this map, and dies with it.
    matches: "OrderedDict[str, list[int]]" = field(default_factory=OrderedDict)


@dataclass
class PreviewResult:
    """What a lanelet-search preview found.

    Attributes:
        searching: Whether the slot's lanelet is left to a constraint search.  A
            pinned one has nothing to evaluate: its lanelet is the one the
            picker is already outlining.
        matched_ids: Lanelet IDs satisfying the constraints.
        total: Lanelets in the map, for "37 of 812".
        constraint_count: How many top-level constraints were evaluated.
        map_loaded: Whether a map was available.
        error: Why the map or the constraints could not be evaluated.
    """

    searching: bool = True
    matched_ids: list[int] = field(default_factory=list)
    total: int = 0
    constraint_count: int = 0
    map_loaded: bool = False
    error: str = ""

    @property
    def highlight_ids(self) -> list[int]:
        """The lanelets the open map should outline: the matches.

        The viewer has a single highlight channel -- one outline colour, no
        second class -- so the set has to mean exactly one thing, and here it
        means "what the search found".  A pinned lanelet asks for no preview at
        all: the picker outlines it from the field itself.
        """
        return list(self.matched_ids)


# ---------------------------------------------------------------------------
# Constraint materialisation
# ---------------------------------------------------------------------------


def materialize_constraints(
    nodes: "list[ConstraintNode]", document: ScenarioDocument
) -> list[dict[str, Any]]:
    """Return the constraint tree as sweeper dicts with references resolved.

    ``${map.no_3d_model_lanelet_ids}`` is an OmegaConf interpolation that only
    resolves once Hydra composes the config.  The editor has no Hydra in the
    loop, so the one reference the constraint vocabulary uses is substituted
    from the document's own map here -- keeping the stored YAML identical to
    what an exported package ships.
    """
    exclusions = list(document.map.no_3d_model_lanelet_ids)

    def _resolve(raw: Any) -> Any:
        if isinstance(raw, dict):
            return {key: _resolve(value) for key, value in raw.items()}
        if isinstance(raw, list):
            return [_resolve(value) for value in raw]
        if raw == MAP_EXCLUSION_REF:
            return exclusions
        return raw

    return [_resolve(node.to_sweep_dict()) for node in nodes]


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------


#: Environment variable listing the directories a map may be read from,
#: separated by :data:`os.pathsep`.  Defaults to the working directory the
#: editor was started in, which is what document paths are written relative to.
MAP_ROOTS_ENV = "SCENARIO_EDITOR_MAP_ROOTS"


def map_roots() -> tuple[Path, ...]:
    """Return the directories a Lanelet2 map may be loaded from.

    The map cache is always one of them.  A map named by ``map.source`` is
    resolved to a file inside it, and the sandbox below is what decides whether
    that file may be served -- so leaving the cache out would make every remote
    map unservable, while the check still bounds a *typed* path to where the
    editor was started.
    """
    configured = os.environ.get(MAP_ROOTS_ENV, "")
    roots = [
        Path(part).expanduser().resolve()
        for part in configured.split(os.pathsep)
        if part.strip()
    ]
    return (*(roots or (Path.cwd().resolve(),)), map_root())


def map_paths(
    document: ScenarioDocument, *, allow_fetch: bool = False, refresh: bool = False
) -> MapPaths:
    """Return the document's map files, resolving ``map.source`` if it has one.

    Args:
        document: The scenario being edited.
        allow_fetch: Whether an uncached source may be cloned.  Off by default,
            because this is called on every render.
        refresh: Fetch the repository again even when it is already cached.

    Returns:
        The resolved paths.  A source that is unusable, unreachable or simply
        not cached yet contributes nothing rather than raising -- the editor
        stays usable without a map, and reports the reason where the map would
        have been.
    """
    try:
        return resolve_map_paths(document.map, allow_fetch=allow_fetch, refresh=refresh)
    except (MapSourceError, MapCacheError, MapResolutionError):
        if allow_fetch or refresh:
            raise
        logger.debug("Map source not resolvable yet", exc_info=True)
        return MapPaths(name=document.map.name)


@dataclass
class MapStatus:
    """What the Scenario inspector says about the map, without fetching it.

    Everything here is answerable from disk, because the inspector re-renders on
    every keystroke: downloading a map is a button, not a consequence of typing
    into the field above it.

    Attributes:
        source: The source URI as the document holds it.
        error: Why the source cannot be used, if it cannot be parsed.
        cached: Whether the Lanelet2 map is on this machine already.
        provisioned: Whether it was found unpacked under the map root -- that
            is, downloaded by Autoware's own setup rather than by this editor.
        pinned: Whether the source names an exact commit.
        commit: The commit the cached copy is at, when that is known.
        lanelet2_path: The Lanelet2 file, once cached.
        xodr_path: The OpenDRIVE file, once it exists.
    """

    source: str = ""
    error: str = ""
    cached: bool = False
    provisioned: bool = False
    pinned: bool = False
    commit: str = ""
    lanelet2_path: Optional[Path] = None
    xodr_path: Optional[Path] = None

    @property
    def has_source(self) -> bool:
        """Whether the scenario names a map repository at all."""
        return bool(self.source)

    @property
    def needs_opendrive(self) -> bool:
        """Whether the map has no OpenDRIVE yet.

        Not a blocker for authoring.  A scenario is written against Lanelet2 --
        lanelets are what a spawn, a goal and a search all name -- so the
        preview and a sweep work without one.  A *run* needs it, and takes it
        from CARLA when it loads the world.
        """
        return self.cached and self.xodr_path is None


def map_status(
    document: ScenarioDocument, paths: Optional[MapPaths] = None
) -> MapStatus:
    """Return what is known about *document*'s map, touching no network.

    Args:
        document: The scenario being edited.
        paths: Already-resolved paths, when the caller has them.  A render
            resolves the map once and hands the result to everything that asks
            about it, rather than each asking the filesystem again.
    """
    source = (document.map.source or "").strip()
    status = MapStatus(source=source)
    if source:
        try:
            status.pinned = MapSource.parse(source).pinned
        except MapSourceError as exc:
            status.error = str(exc)
            return status

    paths = map_paths(document) if paths is None else paths
    status.lanelet2_path = paths.lanelet2_path
    status.xodr_path = paths.xodr_path
    status.cached = paths.lanelet2_path is not None
    if paths.resolved is not None:
        status.provisioned = paths.resolved.provisioned
        status.commit = paths.resolved.commit
    return status


def lanelet2_source(
    document: ScenarioDocument, paths: Optional[MapPaths] = None
) -> Optional[Path]:
    """Return the document's Lanelet2 file, if it is configured and readable.

    The wasm viewer renders the map itself, so the editor only has to hand it
    the ``.osm``; this is the one place that decides which file that is -- and
    therefore the one place that has to refuse the wrong one.

    A *typed* path is the untrusted one.  It comes from a document field anyone
    using the editor can fill in, and the editor binds ``0.0.0.0`` by default.
    Handed straight to a ``FileResponse`` that made ``/draft/<id>/map.osm`` an
    arbitrary local file read for anyone who could reach the port: create a
    draft, point it at ``/etc/passwd``, download it.  So a typed path is
    accepted only when it resolves inside one of :func:`map_roots` and names a
    ``.osm``.

    A path the resolver produced is not typed: it names a file inside the map
    cache that this process just checked out, and it is accepted as such.  That
    keeps :func:`map_roots` meaning "where a typed path may point" rather than
    growing a new entry every time the framework learns another place to keep a
    map.
    """
    paths = map_paths(document) if paths is None else paths
    if paths.lanelet2_path is None:
        return None
    path = paths.lanelet2_path.expanduser()
    if paths.from_source:
        return path if path.is_file() else None
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve()
    except OSError:
        return None

    # ``resolve()`` first, so `..` cannot walk out of a root it started in.
    roots = map_roots()
    if not any(resolved == root or root in resolved.parents for root in roots):
        logger.warning(
            "Refusing map outside %s: %s",
            os.pathsep.join(str(r) for r in roots),
            resolved,
        )
        return None
    if resolved.suffix.lower() != ".osm":
        logger.warning("Refusing map that is not a .osm file: %s", resolved)
        return None
    return resolved if resolved.is_file() else None


def _cache_key(paths: MapPaths) -> Optional[tuple[str, str, str]]:
    """Return the cache key for a resolved map, or ``None`` when it has none.

    Every input :func:`~autoware_carla_scenario.sweeper.map_loader.load_map`
    reads is in the key.  The OpenDRIVE and the projection are not required --
    a map read without them lands in the same place, because the origin they
    would have supplied is the one its own descriptor states -- but a map that
    *gains* either has to be re-read rather than answered from the earlier
    parse.
    """
    if paths.lanelet2_path is None:
        return None
    return (
        str(paths.lanelet2_path),
        str(paths.xodr_path or ""),
        paths.projector_type or "",
    )


def is_map_loaded(document: ScenarioDocument, paths: Optional[MapPaths] = None) -> bool:
    """Whether the document's map is already parsed and cached."""
    key = _cache_key(map_paths(document) if paths is None else paths)
    return key is not None and key in _MAP_CACHE


def clear_cache() -> None:
    """Drop every cached map.  Used by tests and by an explicit reload."""
    _MAP_CACHE.clear()


def missing_map_reason(document: ScenarioDocument) -> str:
    """Say why the document has no readable Lanelet2 map, and what to do.

    Said in two places -- where the map would have been drawn, and where a
    preview would have been evaluated -- so it is worded once.  A scenario that
    names a map repository is not misconfigured, it is just not downloaded yet,
    and telling that person to type a path would be telling them to undo the
    thing they did.
    """
    if document.map.source:
        return (
            "This scenario's map has not been downloaded yet. Fetch it from "
            "the Map library, or clear the source and name a local file."
        )
    if document.map.lanelet2_path:
        return (
            "The Lanelet2 file this scenario names cannot be read: "
            f"{document.map.lanelet2_path}"
        )
    return (
        "The scenario has no map configured. Set a map source, or the Lanelet2 "
        "(.osm) path, in the Scenario inspector."
    )


def _load(document: ScenarioDocument, paths: MapPaths) -> _LoadedMap:
    """Parse the document's map, or return the cached parse.

    Raises:
        FileNotFoundError: If the map files are not configured or missing.
        RuntimeError: If Lanelet2 could not parse the map.
    """
    key = _cache_key(paths)
    if key is None:
        raise FileNotFoundError(missing_map_reason(document))
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        _MAP_CACHE.move_to_end(key)
        return cached

    from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415
    from ..sweeper.map_loader import load_map  # noqa: PLC0415

    try:
        lanelet_map = load_map(paths)
        routing_graph = create_routing_graph(lanelet_map)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 -- lanelet2 raises bare RuntimeErrors
        raise RuntimeError(f"Could not load the Lanelet2 map: {exc}") from exc

    loaded = _LoadedMap(
        lanelet_map=lanelet_map,
        routing_graph=routing_graph,
        lanelet_count=len(list(lanelet_map.laneletLayer)),
    )
    _MAP_CACHE[key] = loaded
    while len(_MAP_CACHE) > _MAP_CACHE_SIZE:
        _MAP_CACHE.popitem(last=False)
    return loaded


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_slot(
    document: ScenarioDocument,
    slot: LaneletSlot,
    *,
    load_map: bool = False,
    paths: Optional[MapPaths] = None,
) -> PreviewResult:
    """Evaluate one lanelet slot's constraints against the document's map.

    Every slot is previewed the same way -- a spawn, the ego's goal, the lanelet
    a condition watches -- because the question the preview answers ("which
    lanelets satisfy this?") does not depend on what the lanelet is for.

    Args:
        document: The scenario being edited.
        slot: The lanelet slot whose search is previewed.
        load_map: Parse the map when it is not cached yet.  Left off, an
            unloaded map returns a result that still describes the constraints,
            so editing them never waits on a map.
        paths: Already-resolved paths, when the caller has them.

    Returns:
        A :class:`PreviewResult`.  Failures are reported in
        :attr:`PreviewResult.error` rather than raised: the constraint builder
        has to keep working on a machine with no map files.
    """
    choice = slot.choice
    searching = choice.searching
    constraint_count = len(choice.constraints)
    result = PreviewResult(searching=searching, constraint_count=constraint_count)

    if searching and not constraint_count:
        result.error = "Add a constraint to see which lanelets match."
        return result

    resolved = map_paths(document) if paths is None else paths
    if not load_map and not is_map_loaded(document, resolved):
        return result

    try:
        loaded = _load(document, resolved)
    except (FileNotFoundError, RuntimeError) as exc:
        result.error = str(exc)
        return result

    result.map_loaded = True
    result.total = loaded.lanelet_count

    if not searching:
        return result

    materialized = materialize_constraints(choice.constraints, document)
    # The tree itself is the question being asked, so it is the key: two slots
    # searching for the same thing, or the same slot across an edit that did not
    # touch the search, are one answer.
    asked = json.dumps(materialized, sort_keys=True, default=str)
    remembered = loaded.matches.get(asked)
    if remembered is not None:
        loaded.matches.move_to_end(asked)
        result.matched_ids = list(remembered)
        return result

    from ..sweeper.constraints import (  # noqa: PLC0415
        find_matching_lanelets,
        parse_constraint,
    )

    try:
        parsed = [parse_constraint(cfg) for cfg in materialized]
        matched = find_matching_lanelets(
            parsed, loaded.lanelet_map, loaded.routing_graph
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not raised
        logger.info("Lanelet preview failed for %s: %s", slot.key, exc)
        result.error = f"Constraints could not be evaluated: {exc}"
        return result

    loaded.matches[asked] = matched
    while len(loaded.matches) > _MATCH_CACHE_SIZE:
        loaded.matches.popitem(last=False)
    result.matched_ids = list(matched)
    return result
