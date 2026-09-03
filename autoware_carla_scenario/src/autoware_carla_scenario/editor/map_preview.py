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
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..authoring.models import ConstraintNode, Entity, ScenarioDocument
from ..authoring.validator import MAP_EXCLUSION_REF

logger = logging.getLogger(__name__)

__all__ = [
    "LaneletPolyline",
    "MapGeometry",
    "PreviewResult",
    "clear_cache",
    "evaluate_spawn",
    "is_map_loaded",
    "lanelet2_source",
    "materialize_constraints",
]

#: Points kept per lanelet centerline.  The preview is a locator, not a
#: rendering: more points cost DOM nodes and buy nothing at this zoom.
_MAX_POINTS_PER_LANELET = 8

#: Side of the square SVG user-space the map is fitted into.
_VIEWBOX = 1000.0

#: Parsed maps kept in memory, newest last.  Parsing is measured in seconds, so
#: one map has to stay resident to make the preview usable; more than a couple
#: only pins native map objects for the life of the process, and a session edits
#: one map at a time.
_MAP_CACHE_SIZE = 2

_MAP_CACHE: "OrderedDict[tuple[str, str], _LoadedMap]" = OrderedDict()


@dataclass
class LaneletPolyline:
    """One lanelet centerline, already in SVG user-space coordinates."""

    lanelet_id: int
    points: str


@dataclass
class MapGeometry:
    """Every lanelet centerline, fitted to a square viewBox."""

    size: float = _VIEWBOX
    lanelets: list[LaneletPolyline] = field(default_factory=list)


@dataclass
class _LoadedMap:
    """A parsed map plus the derived structures the preview reuses."""

    lanelet_map: Any
    routing_graph: Any
    geometry: MapGeometry
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
        geometry: Map drawing, when a map was loaded.
        selected_id: The entity's current spawn lanelet, highlighted.
        map_loaded: Whether a map was available.
        error: Why the map or the constraints could not be evaluated.
    """

    searching: bool = True
    matched_ids: list[int] = field(default_factory=list)
    total: int = 0
    constraint_count: int = 0
    geometry: Optional[MapGeometry] = None
    selected_id: Optional[int] = None
    map_loaded: bool = False
    error: str = ""

    @property
    def matched_set(self) -> set[int]:
        """Matched IDs as a set, for template membership tests."""
        return set(self.matched_ids)


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


def lanelet2_source(document: ScenarioDocument) -> Optional[Path]:
    """Return the document's Lanelet2 file, if it is configured and readable.

    The wasm viewer renders the map itself, so the editor only has to hand it
    the ``.osm``; this is the one place that decides which file that is.
    """
    configured = document.map.lanelet2_path
    if not configured:
        return None
    path = Path(configured).expanduser()
    return path if path.is_file() else None


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


def _build_geometry(lanelet_map: Any) -> MapGeometry:
    """Project every centerline into a square SVG viewBox."""
    raw: list[tuple[int, list[tuple[float, float]]]] = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for lanelet in lanelet_map.laneletLayer:
        points = [(float(p.x), float(p.y)) for p in lanelet.centerline]
        if len(points) < 2:
            continue
        if len(points) > _MAX_POINTS_PER_LANELET:
            step = (len(points) - 1) / (_MAX_POINTS_PER_LANELET - 1)
            points = [
                points[min(len(points) - 1, round(i * step))]
                for i in range(_MAX_POINTS_PER_LANELET)
            ]
        raw.append((int(lanelet.id), points))
        for x, y in points:
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

    geometry = MapGeometry()
    if not raw:
        return geometry

    span = max(max_x - min_x, max_y - min_y) or 1.0
    scale = _VIEWBOX / span
    # Centre the map in the square, and flip y: Lanelet2 is y-up, SVG is y-down.
    offset_x = (_VIEWBOX - (max_x - min_x) * scale) / 2.0
    offset_y = (_VIEWBOX - (max_y - min_y) * scale) / 2.0

    for lanelet_id, points in raw:
        projected = " ".join(
            f"{(x - min_x) * scale + offset_x:.1f},"
            f"{_VIEWBOX - ((y - min_y) * scale + offset_y):.1f}"
            for x, y in points
        )
        geometry.lanelets.append(
            LaneletPolyline(lanelet_id=lanelet_id, points=projected)
        )
    return geometry


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
        geometry=_build_geometry(lanelet_map),
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
    result.geometry = loaded.geometry
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
