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
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import ConstraintNode, LaneletSlot, ScenarioDocument
from ..authoring.hydra_config import (
    MAP_EXCLUSION_REF,
    effective_no_3d_model_lanelet_ids,
)
from ..maps import MapCacheError, MapPaths, MapSource, MapSourceError, map_root
from ..maps.config import resolve_map_paths
from ..maps.resolver import PROJECTOR_INFO_FILENAME, MapResolutionError

logger = logging.getLogger(__name__)

__all__ = [
    "MAP_ROOTS_ENV",
    "AreaPreview",
    "PreviewResult",
    "SpawnOffset",
    "clear_cache",
    "evaluate_area",
    "evaluate_slot",
    "is_map_loaded",
    "MapStatus",
    "lanelet2_source",
    "map_paths",
    "map_status",
    "materialize_constraints",
    "missing_map_reason",
    "resolve_spawn_offset",
]

#: Parsed maps kept in memory, newest last.  Parsing is measured in seconds, so
#: one map has to stay resident to make the preview usable; more than a couple
#: only pins native map objects for the life of the process, and a session edits
#: one map at a time.
_MAP_CACHE_SIZE = 2

#: What a parse is cached under (:func:`_cache_key`): the Lanelet2 path, the
#: OpenDRIVE path and the projector override, then the modification time and
#: size of the ``.osm``, the ``.xodr`` and ``map_projector_info.yaml``.
_CacheKey = tuple[str, str, str, tuple[int, int], tuple[int, int], tuple[int, int]]

_MAP_CACHE: "OrderedDict[_CacheKey, _LoadedMap]" = OrderedDict()

#: Guards :data:`_MAP_CACHE` and the ``matches`` / ``derived`` caches of every
#: map in it.  The editor reaches them from worker threads at once -- the map
#: panel (a sync route, so run in Starlette's threadpool) and the lanelet
#: preview (offloaded so a parse does not stall the event loop) -- and an
#: ``OrderedDict``'s ``move_to_end`` / ``popitem`` are not safe to interleave.
#: Only ever held for a dictionary operation, never across a parse or a search.
_CACHE_LOCK = threading.Lock()

#: One lock per map being parsed, so two requests for the same unparsed map
#: parse it once (the second waits and is answered from the cache) while maps
#: with different keys still parse side by side.  An entry lives only while
#: its parse runs; guarded by :data:`_CACHE_LOCK`.
_LOADING: "dict[_CacheKey, threading.Lock]" = {}

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
    #: Lanelets derived from a pick, by binding and pick, for the same reason:
    #: the places panel works each one out again on every edit.
    derived: "OrderedDict[tuple[str, int], Optional[int]]" = field(
        default_factory=OrderedDict
    )
    #: Swept spawns' derived offsets, by binding and pick, for the same reason.
    offsets: "OrderedDict[tuple[str, int], SpawnOffset]" = field(
        default_factory=OrderedDict
    )
    #: What each area matches -- its lanelets and how many polygons -- by the
    #: area, for the same reason: the places panel outlines them on every edit.
    areas: "OrderedDict[Any, tuple[list[int], int]]" = field(
        default_factory=OrderedDict
    )


@dataclass(frozen=True)
class SpawnOffset:
    """Where a swept spawn's derived ``s`` puts it for one pick.

    Attributes:
        lanelet_id: The lanelet the spawn is on: the pick, or the predecessor
            a ``stop_line_offset`` walks back to when the pick is too short.
        s: The offset along :attr:`lanelet_id`, or ``None`` when there is
            none to show.
        walked_back: Whether :attr:`lanelet_id` is not the pick.
        error: Why the binding failed, which is also whether the sweep drops
            the pick.
    """

    lanelet_id: int
    s: Optional[float] = None
    walked_back: bool = False
    error: str = ""

    @property
    def dropped(self) -> bool:
        """Whether the sweep skips this pick because the binding failed."""
        return bool(self.error)


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
    here -- keeping the stored YAML identical to what an exported package
    ships -- with the list a run of the export resolves it to
    (:func:`~..authoring.hydra_config.effective_no_3d_model_lanelet_ids`): the
    document's own when it lists any, else the built-in map group's, so the
    preview matches what a sweep enumerates (#1523).
    """
    exclusions = effective_no_3d_model_lanelet_ids(document)

    def _resolve(raw: Any) -> Any:
        if isinstance(raw, dict):
            return {key: _resolve(value) for key, value in raw.items()}
        if isinstance(raw, list):
            return [_resolve(value) for value in raw]
        if isinstance(raw, str) and raw.strip() == MAP_EXCLUSION_REF:
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

    Every path is checked the same way: it must resolve inside one of
    :func:`map_roots` and name a ``.osm``.  The editor has no authentication
    and may bind ``0.0.0.0`` (``--host``), so without that, ``/draft/<id>/map.osm`` is an arbitrary
    local file read for anyone who can reach the port.

    A *typed* path is the obvious way in -- create a draft, point it at
    ``/etc/passwd``, download it -- but it is not the only one.  A path the
    resolver produced used to be trusted on the grounds that it names a file
    inside the map cache this process just checked out.  What that missed is
    that the *contents* of a checkout are the repository's, and git carries
    symlinks: a repository whose ``lanelet2_map.osm`` is a symlink to any
    readable file on the host resolves, passes ``is_file()`` -- which follows
    it -- and is served.  Resolving before the containment check is what closes
    that, because a link out of the cache lands outside every root.
    """
    paths = map_paths(document) if paths is None else paths
    if paths.lanelet2_path is None:
        return None
    resolved = _inside_map_roots(paths.lanelet2_path)
    if resolved is None:
        return None
    if resolved.suffix.lower() != ".osm":
        logger.warning("Refusing map that is not a .osm file: %s", resolved)
        return None
    return resolved if resolved.is_file() else None


def _inside_map_roots(path: Path) -> Optional[Path]:
    """Return *path* resolved when it lies inside one of :func:`map_roots`,
    else ``None`` (logged).

    The containment check every map file the editor reads goes through: the
    drawn map (:func:`lanelet2_source`) and every parse (:func:`_load`, and
    the OpenDRIVE an area reads its road surface from), so a map refused for
    drawing is not parsed for a preview either (#2412).  A relative path is
    the working directory's, as the document's paths are written.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        # ValueError: a NUL byte in a typed path, which is no map either
        # (fetch_opendrive answered it with a 500, #2414).
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
    return resolved


def _outside_map_roots_reason(path: Path) -> str:
    """Say that the map file *path* is outside the editor's map roots."""
    return (
        f"The map file {path} is outside the editor's map roots: add its "
        f"directory to {MAP_ROOTS_ENV} (the working directory unless set) to "
        "preview it."
    )


def _cache_key(paths: MapPaths) -> Optional[_CacheKey]:
    """Return the cache key for a resolved map, or ``None`` when it has none.

    Every input :func:`~autoware_carla_scenario.sweeper.map_loader.load_map`
    reads is in the key.  The OpenDRIVE and the projection are not required --
    a map read without them lands in the same place, because the origin they
    would have supplied is the one its own descriptor states -- but a map that
    *gains* either has to be re-read rather than answered from the earlier
    parse.

    So is what those files hold, by their modification time and size (``(0,
    -1)`` for one that is missing): a ``map.directory`` map edited while the
    editor runs -- its ``.osm``, OpenDRIVE or ``map_projector_info.yaml`` -- is
    parsed again rather than previewed, picked from and matched against as it
    was (#2446).
    """
    lanelet2 = paths.lanelet2_path
    if lanelet2 is None:
        return None
    xodr = paths.xodr_path
    return (
        str(lanelet2),
        str(xodr or ""),
        paths.projector_type or "",
        _file_stamp(lanelet2),
        (0, -1) if xodr is None else _file_stamp(xodr),
        _file_stamp(lanelet2.parent / PROJECTOR_INFO_FILENAME),
    )


def _file_stamp(path: Path) -> tuple[int, int]:
    """*path*'s modification time (ns) and size, ``(0, -1)`` when missing.

    The viewer's ``ui.case_paths._stat``, which the editor may not import
    (``test_editor_viewer_isolation.py``).
    """
    try:
        info = path.stat()
    except OSError:
        return (0, -1)
    return (info.st_mtime_ns, info.st_size)


def is_map_loaded(document: ScenarioDocument, paths: Optional[MapPaths] = None) -> bool:
    """Whether the document's map is already parsed and cached."""
    key = _cache_key(map_paths(document) if paths is None else paths)
    if key is None:
        return False
    with _CACHE_LOCK:
        return key in _MAP_CACHE


def clear_cache() -> None:
    """Drop every cached map.  Used by tests and by an explicit reload."""
    with _CACHE_LOCK:
        _MAP_CACHE.clear()


def _cached(key: _CacheKey) -> Optional[_LoadedMap]:
    """Return the cached parse for *key*, marking it most recently used."""
    with _CACHE_LOCK:
        cached = _MAP_CACHE.get(key)
        if cached is not None:
            _MAP_CACHE.move_to_end(key)
        return cached


def missing_map_reason(document: ScenarioDocument) -> str:
    """Say why the document has no readable Lanelet2 map, and what to do.

    Said in two places -- where the map would have been drawn, and where a
    preview would have been evaluated -- so it is worded once.  A scenario that
    names a map repository is not misconfigured, it is just not downloaded yet,
    and telling that person to type a path would be telling them to undo the
    thing they did.
    """
    if document.map.directory:
        return (
            "The map directory this scenario names holds no readable Lanelet2 "
            f"map, or is outside the editor's map roots: {document.map.directory}"
        )
    if document.map.source:
        return (
            "This scenario's map has not been downloaded yet. Fetch it from "
            "the Map library, or clear the source and name a local file."
        )
    if document.map.lanelet2_path:
        return (
            "The Lanelet2 file this scenario names cannot be read, or is "
            f"outside the editor's map roots: {document.map.lanelet2_path}"
        )
    return (
        "The scenario has no map configured. Set a map source, or the Lanelet2 "
        "(.osm) path, in the Scenario inspector."
    )


def _loaded(
    document: ScenarioDocument, paths: Optional[MapPaths], *, load_map: bool
) -> Optional[_LoadedMap]:
    """Return the document's parsed map, or ``None`` when it is not parsed yet.

    *load_map* parses it on the spot instead, which is the expensive part:
    left off, nothing here waits on a map nobody has loaded.

    Raises:
        FileNotFoundError: If the map files are not configured or missing.
        RuntimeError: If Lanelet2 could not parse the map.
    """
    resolved = map_paths(document) if paths is None else paths
    if not load_map and not is_map_loaded(document, resolved):
        return None
    return _load(document, resolved)


def _load(document: ScenarioDocument, paths: MapPaths) -> _LoadedMap:
    """Parse the document's map, or return the cached parse.

    Raises:
        FileNotFoundError: If the map files are not configured, missing or
            outside :func:`map_roots`.
        RuntimeError: If Lanelet2 could not parse the map.
    """
    key = _cache_key(paths)
    if key is None:
        raise FileNotFoundError(missing_map_reason(document))
    # Checked on every call, a cached parse too: the files load_map() reads
    # are held to the roots lanelet2_source() holds the drawn map to.
    for path in (paths.lanelet2_path, paths.xodr_path):
        if path is not None and _inside_map_roots(path) is None:
            raise FileNotFoundError(_outside_map_roots_reason(path))
    cached = _cached(key)
    if cached is not None:
        return cached

    # Parsing takes seconds, so it runs outside _CACHE_LOCK: holding that would
    # stall every cache lookup -- and so every editor render -- for the length
    # of the parse.  The per-map lock dedupes concurrent parses of one map.
    with _CACHE_LOCK:
        loading = _LOADING.setdefault(key, threading.Lock())
    with loading:
        try:
            cached = _cached(key)
            if cached is not None:
                return cached
            loaded = _parse(paths)
            with _CACHE_LOCK:
                # A parse of the same files as they were before an edit is
                # never looked up again: it goes rather than hold a slot.
                for stale in [k for k in _MAP_CACHE if k[:3] == key[:3]]:
                    del _MAP_CACHE[stale]
                _MAP_CACHE[key] = loaded
                while len(_MAP_CACHE) > _MAP_CACHE_SIZE:
                    _MAP_CACHE.popitem(last=False)
            return loaded
        finally:
            with _CACHE_LOCK:
                if _LOADING.get(key) is loading:
                    del _LOADING[key]


def _parse(paths: MapPaths) -> _LoadedMap:
    """Parse the map at *paths* and build its routing graph.  Not cached.

    Raises:
        FileNotFoundError: If the map files are missing.
        RuntimeError: If Lanelet2 could not parse the map.
    """
    from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415
    from ..sweeper.map_loader import load_map  # noqa: PLC0415

    try:
        lanelet_map = load_map(paths)
        routing_graph = create_routing_graph(lanelet_map)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 -- lanelet2 raises bare RuntimeErrors
        raise RuntimeError(f"Could not load the Lanelet2 map: {exc}") from exc

    return _LoadedMap(
        lanelet_map=lanelet_map,
        routing_graph=routing_graph,
        lanelet_count=len(list(lanelet_map.laneletLayer)),
    )


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

    try:
        loaded = _loaded(document, paths, load_map=load_map)
    except (FileNotFoundError, RuntimeError) as exc:
        result.error = str(exc)
        return result
    if loaded is None:
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
    with _CACHE_LOCK:
        remembered = loaded.matches.get(asked)
        if remembered is not None:
            loaded.matches.move_to_end(asked)
    if remembered is not None:
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

    with _CACHE_LOCK:
        loaded.matches[asked] = matched
        while len(loaded.matches) > _MATCH_CACHE_SIZE:
            loaded.matches.popitem(last=False)
    result.matched_ids = list(matched)
    return result


def derive_lanelet(
    document: ScenarioDocument,
    slot: LaneletSlot,
    picked_id: int,
    *,
    paths: Optional[MapPaths] = None,
) -> Optional[int]:
    """Return the lanelet a derived *slot* works out from the pick *picked_id*.

    The sweeper's own binding does the working out, so the map draws the
    lanelet a run would get.  ``None`` when the map is not loaded yet, the slot
    has no binding, or the pick has no such lanelet -- a run would drop that
    case.
    """
    binding = slot.choice.binding
    if binding is None:
        return None
    try:
        loaded = _loaded(document, paths, load_map=False)
    except (FileNotFoundError, RuntimeError):
        return None
    if loaded is None:
        return None
    asked = (json.dumps(binding.to_sweep_dict(), sort_keys=True), picked_id)
    with _CACHE_LOCK:
        if asked in loaded.derived:
            loaded.derived.move_to_end(asked)
            return loaded.derived[asked]

    from ..sweeper.bindings import parse_binding  # noqa: PLC0415

    try:
        value = (
            parse_binding(slot.key, binding.to_sweep_dict())
            .resolve(picked_id, loaded.lanelet_map, loaded.routing_graph)
            .value
        )
    except Exception as exc:  # noqa: BLE001 -- shown as "not derived", not raised
        logger.info("Binding for %s did not resolve: %s", slot.key, exc)
        value = None
    derived = value if isinstance(value, int) else None
    with _CACHE_LOCK:
        loaded.derived[asked] = derived
        while len(loaded.derived) > _MATCH_CACHE_SIZE:
            loaded.derived.popitem(last=False)
    return derived


def resolve_spawn_offset(
    document: ScenarioDocument,
    s_key: str,
    binding: dict[str, Any],
    picked_id: int,
    *,
    paths: Optional[MapPaths] = None,
) -> Optional[SpawnOffset]:
    """Return where the swept spawn lands for the pick *picked_id*.

    *s_key* and *binding* are the spawn's offset binding as ``sweep.bindings``
    carries it (:func:`~autoware_carla_scenario.authoring.hydra_config.spawn_s_binding`),
    and the sweeper's own binding resolves it, exactly as
    :func:`~autoware_carla_scenario.sweeper.expand.expand_sweep` does: it may
    walk back to a predecessor, which moves the spawn there, and a binding that
    fails drops the pick from the sweep.  ``None`` when the map is not loaded.
    """
    try:
        loaded = _loaded(document, paths, load_map=False)
    except (FileNotFoundError, RuntimeError):
        return None
    if loaded is None:
        return None
    asked = (json.dumps([s_key, binding], sort_keys=True), picked_id)
    with _CACHE_LOCK:
        if asked in loaded.offsets:
            loaded.offsets.move_to_end(asked)
            return loaded.offsets[asked]

    from ..sweeper.bindings import parse_binding  # noqa: PLC0415

    try:
        result = parse_binding(s_key, binding).resolve(
            picked_id, loaded.lanelet_map, loaded.routing_graph
        )
    except Exception as exc:  # noqa: BLE001 -- shown as "dropped", not raised
        logger.info("Binding for %s did not resolve: %s", s_key, exc)
        offset = SpawnOffset(lanelet_id=picked_id, error=str(exc) or type(exc).__name__)
    else:
        moved = result.lanelet_id_override
        value = result.value
        offset = SpawnOffset(
            lanelet_id=picked_id if moved is None else moved,
            s=float(value) if isinstance(value, (int, float)) else None,
            walked_back=moved is not None and moved != picked_id,
        )
    with _CACHE_LOCK:
        loaded.offsets[asked] = offset
        while len(loaded.offsets) > _MATCH_CACHE_SIZE:
            loaded.offsets.popitem(last=False)
    return offset


@dataclass
class AreaPreview:
    """What an area of the map matches on the document's map.

    Attributes:
        lanelet_ids: The lanelets in it, which the map outlines.
        polygon_count: How many polygons of the polygon layer are in it; the
            viewer outlines lanelets only, so these are counted, not drawn.
        surface_count: How many raised or grooved parts of the roads' surface
            are in it, read from the OpenDRIVE map's OpenCRG grids; counted,
            not drawn, as polygons are.
        skipped: What of those grids could not be read, each said in words.
        description: The area in words ("crosswalks (lanelet 587)"), or
            ``""`` when its fields cannot be read (:attr:`error` says why).
        map_loaded: Whether the map was parsed, so the match is known.
        error: Why the area could not be matched -- one no map could hold, a
            picked lanelet of another ground -- or ``""``.
    """

    lanelet_ids: list[int] = field(default_factory=list)
    polygon_count: int = 0
    surface_count: int = 0
    skipped: list[str] = field(default_factory=list)
    description: str = ""
    map_loaded: bool = False
    error: str = ""

    @property
    def empty(self) -> bool:
        """Whether the map was matched and nothing on it is in the area: the
        run would say so, and the card never hold."""
        return (
            self.map_loaded
            and not self.error
            and not (self.lanelet_ids or self.polygon_count or self.surface_count)
        )


def evaluate_area(
    document: ScenarioDocument,
    params: "dict[str, Any]",
    *,
    load_map: bool = False,
    paths: Optional[MapPaths] = None,
) -> AreaPreview:
    """Match an area card's fields (``ground``, ``lanelet_ids``,
    ``attributes``) against the document's map, with the framework's own
    :func:`~autoware_carla_scenario.conditions.map_area.area_shapes`, so the
    outline is the ground the run measures to.

    Failures are reported in :attr:`AreaPreview.error`, never raised; an
    unloaded map (with *load_map* off) returns an empty, unloaded result.
    """
    from ..conditions.map_area import (  # noqa: PLC0415
        area_shapes,
        map_area_from_fields,
        surface_regions,
    )

    result = AreaPreview()
    try:
        area = map_area_from_fields(
            str(params.get("ground") or "CROSSWALK"),
            params.get("lanelet_ids") or None,
            str(params.get("attributes") or ""),
            params.get("relief_m"),
        )
    except (TypeError, ValueError) as exc:
        result.error = str(exc)
        return result
    result.description = area.describe()
    if area.ground.on_surface:
        # On the OpenDRIVE map, not the Lanelet2 one: its grids are read (and
        # remembered) without parsing a Lanelet2 map, so on every render.
        resolved = map_paths(document) if paths is None else paths
        xodr = resolved.xodr_path
        # The roots first, so a path outside them is not even looked up.
        if xodr is not None and _inside_map_roots(xodr) is None:
            result.error = _outside_map_roots_reason(xodr)
            return result
        if xodr is None or not xodr.expanduser().exists():
            result.error = (
                "The road surface is read from the OpenDRIVE map, and there is "
                "none yet."
            )
            return result
        try:
            regions, skipped = surface_regions(xodr.expanduser(), area)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not raised
            result.error = f"The OpenDRIVE map could not be read: {exc}"
            return result
        result.map_loaded = True
        result.surface_count = len(regions)
        result.skipped = list(skipped)
        return result
    try:
        loaded = _loaded(document, paths, load_map=load_map)
    except (FileNotFoundError, RuntimeError) as exc:
        result.error = str(exc)
        return result
    if loaded is None:
        return result
    result.map_loaded = True
    with _CACHE_LOCK:
        remembered = loaded.areas.get(area)
        if remembered is not None:
            loaded.areas.move_to_end(area)
    if remembered is None:
        try:
            shapes = area_shapes(loaded.lanelet_map, area)
        except ValueError as exc:
            result.error = str(exc)
            return result
        lanelets = [s.element_id for s in shapes if s.layer == "lanelet"]
        remembered = (lanelets, len(shapes) - len(lanelets))
        with _CACHE_LOCK:
            loaded.areas[area] = remembered
            while len(loaded.areas) > _MATCH_CACHE_SIZE:
                loaded.areas.popitem(last=False)
    result.lanelet_ids = list(remembered[0])
    result.polygon_count = remembered[1]
    return result
