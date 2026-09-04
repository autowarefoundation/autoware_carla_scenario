"""Evaluate spawn constraints against a real Lanelet2 map, and draw the result.

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

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import ConstraintNode, Entity, ScenarioDocument
from ..authoring.validator import MAP_EXCLUSION_REF

logger = logging.getLogger(__name__)

__all__ = [
    "MAP_ROOTS_ENV",
    "PreviewResult",
    "clear_cache",
    "evaluate_spawn",
    "is_map_loaded",
    "lanelet2_source",
    "materialize_constraints",
]

#: Parsed maps kept in memory, newest last.  Parsing is measured in seconds, so
#: one map has to stay resident to make the preview usable; more than a couple
#: only pins native map objects for the life of the process, and a session edits
#: one map at a time.
_MAP_CACHE_SIZE = 2

_MAP_CACHE: "OrderedDict[tuple[str, str], _LoadedMap]" = OrderedDict()


@dataclass
class _LoadedMap:
    """A parsed map plus the derived structures the preview reuses."""

    lanelet_map: Any
    routing_graph: Any
    lanelet_count: int


@dataclass
class PreviewResult:
    """What a spawn preview found.

    Attributes:
        searching: Whether the entity spawns by constraint search.  A fixed
            spawn still gets a map, just no match list.
        matched_ids: Lanelet IDs satisfying the constraints.
        total: Lanelets in the map, for "37 of 812".
        constraint_count: How many top-level constraints were evaluated.
        selected_id: The entity's current spawn lanelet.
        map_loaded: Whether a map was available.
        error: Why the map or the constraints could not be evaluated.
    """

    searching: bool = True
    matched_ids: list[int] = field(default_factory=list)
    total: int = 0
    constraint_count: int = 0
    selected_id: Optional[int] = None
    map_loaded: bool = False
    error: str = ""

    @property
    def highlight_ids(self) -> list[int]:
        """The lanelets the viewer should outline, which depends on the mode.

        The viewer has a single highlight channel -- one outline colour, no
        second class -- so the set has to mean exactly one thing:

        * a constraint search outlines **the matches**, which is what the
          search is for;
        * a fixed spawn outlines **the pinned lanelet**, because there are no
          matches to show.

        Mixing the current spawn into a search's matches would paint it the
        same colour as them, which says it is one of the matches whether or not
        it is.
        """
        if self.searching:
            return list(self.matched_ids)
        return [self.selected_id] if self.selected_id else []


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
    """Return the directories a Lanelet2 map may be loaded from."""
    configured = os.environ.get(MAP_ROOTS_ENV, "")
    roots = [
        Path(part).expanduser().resolve()
        for part in configured.split(os.pathsep)
        if part.strip()
    ]
    return tuple(roots) or (Path.cwd().resolve(),)


def lanelet2_source(document: ScenarioDocument) -> Optional[Path]:
    """Return the document's Lanelet2 file, if it is configured and readable.

    The wasm viewer renders the map itself, so the editor only has to hand it
    the ``.osm``; this is the one place that decides which file that is -- and
    therefore the one place that has to refuse the wrong one.

    The path comes from a document field anyone using the editor can type, and
    the editor binds ``0.0.0.0`` by default.  Handed straight to a
    ``FileResponse`` that made ``/draft/<id>/map.osm`` an arbitrary local file
    read for anyone who could reach the port: create a draft, point it at
    ``/etc/passwd``, download it.  A path is accepted only when it resolves
    inside one of :func:`map_roots` and names a ``.osm``, so what the route can
    serve is bounded by where the editor was started rather than by what the
    process can read.
    """
    configured = document.map.lanelet2_path
    if not configured:
        return None
    path = Path(configured).expanduser()
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


def _cache_key(document: ScenarioDocument) -> Optional[tuple[str, str]]:
    """Return the cache key for the document's map, or ``None`` when unset."""
    lanelet2_path = document.map.lanelet2_path
    xodr_path = document.map.xodr_path
    if not lanelet2_path or not xodr_path:
        return None
    return (str(Path(lanelet2_path).expanduser()), str(Path(xodr_path).expanduser()))


def is_map_loaded(document: ScenarioDocument) -> bool:
    """Whether the document's map is already parsed and cached."""
    key = _cache_key(document)
    return key is not None and key in _MAP_CACHE


def clear_cache() -> None:
    """Drop every cached map.  Used by tests and by an explicit reload."""
    _MAP_CACHE.clear()


def _load(document: ScenarioDocument) -> _LoadedMap:
    """Parse the document's map, or return the cached parse.

    Raises:
        FileNotFoundError: If the map files are not configured or missing.
        RuntimeError: If Lanelet2 could not parse the map.
    """
    key = _cache_key(document)
    if key is None:
        raise FileNotFoundError(
            "The scenario has no map files configured. Set the Lanelet2 (.osm) "
            "and OpenDRIVE (.xodr) paths in the Scenario inspector."
        )
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        _MAP_CACHE.move_to_end(key)
        return cached

    from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415
    from ..sweeper.map_loader import load_lanelet2_map  # noqa: PLC0415

    lanelet2_path, xodr_path = key
    try:
        lanelet_map = load_lanelet2_map(lanelet2_path, xodr_path)
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


def evaluate_spawn(
    document: ScenarioDocument, entity: Entity, *, load_map: bool = False
) -> PreviewResult:
    """Evaluate *entity*'s spawn constraints against the document's map.

    A fixed spawn is previewed too, without a match list: seeing where the
    lanelet actually is -- and being able to click a different one -- is just as
    useful when the ID was typed by hand.

    Args:
        document: The scenario being edited.
        entity: The entity whose spawn is previewed.
        load_map: Parse the map when it is not cached yet.  Left off, an
            unloaded map returns a result that still describes the constraints,
            so editing them never waits on a map.

    Returns:
        A :class:`PreviewResult`.  Failures are reported in
        :attr:`PreviewResult.error` rather than raised: the constraint builder
        has to keep working on a machine with no map files.
    """
    searching = entity.spawn.mode == "constraint_search"
    constraint_count = len(entity.spawn.constraints)
    result = PreviewResult(
        searching=searching,
        constraint_count=constraint_count,
        selected_id=entity.spawn.lanelet_id or None,
    )

    if searching and not constraint_count:
        result.error = "Add a constraint to see which lanelets match."
        return result

    if not load_map and not is_map_loaded(document):
        return result

    try:
        loaded = _load(document)
    except (FileNotFoundError, RuntimeError) as exc:
        result.error = str(exc)
        return result

    result.map_loaded = True
    result.total = loaded.lanelet_count

    if not searching:
        return result

    from ..sweeper.constraints import (  # noqa: PLC0415
        find_matching_lanelets,
        parse_constraint,
    )

    try:
        parsed = [
            parse_constraint(cfg)
            for cfg in materialize_constraints(entity.spawn.constraints, document)
        ]
        result.matched_ids = find_matching_lanelets(
            parsed, loaded.lanelet_map, loaded.routing_graph
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not raised
        logger.info("Spawn preview failed for %s: %s", entity.id, exc)
        result.error = f"Constraints could not be evaluated: {exc}"
    return result
