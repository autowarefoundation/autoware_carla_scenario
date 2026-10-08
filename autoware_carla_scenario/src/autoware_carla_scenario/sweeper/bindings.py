"""Binding resolution for auto-deriving parameter values per lanelet.

Bindings are parsed from the ``sweep.bindings`` section of a scenario YAML.
Each binding computes a parameter value (e.g. ``ego.spawn_s``) as a function
of the matched lanelet (e.g. "15 m before the stop line").
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

import lanelet2.core
import lanelet2.geometry

from ..utils.stop_line import _collect_stop_lines_from_reg_elems

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binding protocol & implementations
# ---------------------------------------------------------------------------


@dataclass
class BindingResult:
    """Result of resolving a binding.

    Attributes:
        value: The computed parameter value (e.g. ``spawn_s``).  A number for
            a scalar parameter, or a list for one that takes a sequence
            (``scenario.expected_route_lanelet_ids``, say).
        lanelet_id_override: If set, the spawn lanelet should be changed
            to this lanelet because the original lanelet did not have
            enough distance to satisfy the offset.
    """

    value: float | int | list[int]
    lanelet_id_override: int | None = None


class Binding(Protocol):
    """Interface that all parameter bindings must satisfy."""

    @property
    def target_key(self) -> str:
        """Hydra override key this binding writes to (e.g. ``ego.spawn_s``)."""
        ...

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Compute the parameter value for the given lanelet."""
        ...


@dataclass
class StopLineOffsetBinding:
    """Compute ``spawn_s = stop_line_arc_length - offset``.

    The stop line arc-length is determined by projecting the stop line
    centroid onto the lanelet centerline, following the same pattern as
    :func:`~autoware_carla_scenario.coordinate.stop_line._linestring_to_pose`.
    """

    target_key: str
    offset: float = 15.0

    @staticmethod
    def _project_stop_line(ls: Any, lanelet: Any) -> float:
        """Project stop line centroid onto lanelet centerline, return arc-length."""
        points = list(ls)
        if not points:
            raise ValueError("Stop line linestring has no points.")
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        pt = lanelet2.core.BasicPoint2d(cx, cy)
        centerline_2d = lanelet2.geometry.to2D(lanelet.centerline)
        arc = lanelet2.geometry.toArcCoordinates(centerline_2d, pt)
        return arc.length

    def _walk_back_to_predecessor(
        self,
        shortfall: float,
        lanelet: Any,
        lanelet_map: Any,
        routing_graph: Any,
        original_lanelet_id: int,
    ) -> BindingResult:
        """Walk backwards through predecessor lanelets to satisfy the offset.

        When the distance from the spawn lanelet start to the stop line is
        less than the requested offset, we need to place the vehicle on an
        earlier (predecessor) lanelet.

        Args:
            shortfall: Remaining distance that could not be satisfied on the
                original spawn lanelet (always > 0).
            lanelet: The original spawn lanelet object.
            lanelet_map: The lanelet2 map.
            routing_graph: The routing graph for predecessor lookups.
            original_lanelet_id: The ID of the original spawn lanelet.

        Returns:
            A :class:`BindingResult` with the spawn position on a
            predecessor lanelet and the predecessor's ID as override.
        """
        current = lanelet
        remaining = shortfall

        while remaining > 0:
            predecessors = routing_graph.previous(current)
            if not predecessors:
                logger.warning(
                    "[%s] No more predecessors for lanelet %d; "
                    "%.2f m of offset could not be satisfied. "
                    "Placing at start of lanelet %d.",
                    self.target_key,
                    current.id,
                    remaining,
                    current.id,
                )
                result_lanelet = current
                result_s = 0.0
                break

            prev = predecessors[0]
            prev_length = lanelet2.geometry.length2d(prev)

            if prev_length >= remaining:
                result_s = prev_length - remaining
                result_lanelet = prev
                break

            remaining -= prev_length
            current = prev
        else:
            result_lanelet = current
            result_s = 0.0

        logger.info(
            "[%s] Offset not satisfiable on spawn lanelet %d "
            "(shortfall=%.2f m). Walked back to predecessor lanelet %d "
            "-> spawn_s=%.2f m (lanelet length=%.2f m).",
            self.target_key,
            original_lanelet_id,
            shortfall,
            result_lanelet.id,
            result_s,
            lanelet2.geometry.length2d(result_lanelet),
        )
        return BindingResult(
            value=result_s,
            lanelet_id_override=result_lanelet.id,
        )

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the arc-length position *offset* metres before the stop line.

        Walks the spawn lanelet and its following lanelets (BFS via routing
        graph) searching for the nearest stop line.  The result is
        ``accumulated + stop_s - offset``.

        If the offset cannot be satisfied within the spawn lanelet (i.e. the
        result would be negative), the method walks backwards through
        predecessor lanelets to find a position that satisfies the full
        offset distance, returning a :class:`BindingResult` with the
        predecessor lanelet ID as ``lanelet_id_override``.

        The search terminates when the accumulated distance from the spawn
        lanelet end exceeds the offset (further stop lines would only clamp
        the result to ``spawn_lanelet_length``).
        """
        lanelet = lanelet_map.laneletLayer[lanelet_id]
        spawn_lanelet_length = lanelet2.geometry.length2d(lanelet)

        if routing_graph is None:
            from .constraints import create_routing_graph

            routing_graph = create_routing_graph(lanelet_map)

        # BFS over the lanelet chain: spawn -> following -> following -> ...
        # Each entry is (lanelet_object, arc_length_from_spawn_lanelet_start).
        candidates: list[tuple[Any, float]] = [(lanelet, 0.0)]
        visited: set[int] = {lanelet_id}
        seen_ids: set[int] = set()

        while candidates:
            next_candidates: list[tuple[Any, float]] = []
            for current, accumulated in candidates:
                stop_lines = _collect_stop_lines_from_reg_elems(
                    current.regulatoryElements, seen_ids
                )
                if stop_lines:
                    stop_s = self._project_stop_line(stop_lines[0], current)
                    total_arc = accumulated + stop_s
                    raw_result = total_arc - self.offset

                    if raw_result < 0:
                        logger.info(
                            "[%s] Stop line at %.2f m on lanelet %d "
                            "(accumulated=%.2f m from spawn lanelet %d). "
                            "%.2f m before stop line requires %.2f m "
                            "beyond spawn lanelet start; "
                            "walking back to predecessors.",
                            self.target_key,
                            stop_s,
                            current.id,
                            accumulated,
                            lanelet_id,
                            self.offset,
                            -raw_result,
                        )
                        return self._walk_back_to_predecessor(
                            shortfall=-raw_result,
                            lanelet=lanelet,
                            lanelet_map=lanelet_map,
                            routing_graph=routing_graph,
                            original_lanelet_id=lanelet_id,
                        )

                    result = min(raw_result, spawn_lanelet_length)
                    logger.info(
                        "[%s] Stop line at %.2f m on lanelet %d "
                        "(accumulated=%.2f m from spawn lanelet %d). "
                        "%.2f m before stop line -> spawn_s=%.2f m "
                        "on spawn lanelet.",
                        self.target_key,
                        stop_s,
                        current.id,
                        accumulated,
                        lanelet_id,
                        self.offset,
                        result,
                    )
                    return BindingResult(value=result)

                new_accumulated = accumulated + lanelet2.geometry.length2d(current)

                # Stop expanding beyond offset from spawn lanelet end.
                if new_accumulated - spawn_lanelet_length > self.offset:
                    continue

                for fll in routing_graph.following(current):
                    if fll.id not in visited:
                        visited.add(fll.id)
                        next_candidates.append((fll, new_accumulated))

            candidates = next_candidates

        raise ValueError(
            f"Lanelet {lanelet_id} and its following lanelets have no stop line; "
            "cannot resolve stop_line_offset binding."
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@dataclass
class RouteThroughBinding:
    """The lanelets a case drives, starting from the one the sweep picked.

    The picked lanelet is what the case is *about* -- the right-turn lanelet of
    a junction, say -- and the route is that lanelet plus the ``depth``
    lanelets that follow it.  A scenario asserts on those, and for an ego that
    plans for itself the last of them is where it is sent::

        bindings:
          scenario.expected_route_lanelet_ids:
            type: route_through
            depth: 1

    With ``last_only: true`` only the last lanelet is returned, as an ID, which
    is what ``ego.goal_lanelet_id`` takes.

    Where the graph forks, the lowest lanelet ID is taken, so the same map
    always expands to the same cases.  A pick the route cannot be walked from
    raises, and the caller (:func:`~autoware_carla_scenario.sweeper.expand.expand_sweep`)
    drops that case.
    """

    target_key: str
    depth: int = 1
    #: Only the route's last lanelet, as an ID rather than a list: a goal.
    last_only: bool = False

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"route_through depth must be >= 1, got {self.depth}")

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the picked lanelet and the ``depth`` lanelets after it."""
        from .constraints import create_routing_graph

        if routing_graph is None:
            routing_graph = create_routing_graph(lanelet_map)
        route = [lanelet_id]
        current = lanelet_map.laneletLayer[lanelet_id]
        for _ in range(self.depth):
            following = sorted(routing_graph.following(current), key=lambda ll: ll.id)
            if not following:
                raise ValueError(
                    f"[{self.target_key}] lanelet {current.id} has no following "
                    f"lanelet; cannot walk a route of depth {self.depth} from "
                    f"lanelet {lanelet_id}."
                )
            current = following[0]
            route.append(current.id)
        if self.last_only:
            return BindingResult(value=route[-1])
        return BindingResult(value=route)


@dataclass
class AdjacentBinding:
    """The lanelet beside the pick that a vehicle may change into from it, by
    ID: where another vehicle drives alongside the ego::

        bindings:
          scenario.npc_lanelet_id:
            type: adjacent
            side: left

    The lane ``has_adjacent`` asks about, from the routing graph's ``left`` /
    ``right``. A pick with no such lane on that side raises, and the case is
    dropped.
    """

    target_key: str
    side: str = "left"

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(
                f"adjacent side must be 'left' or 'right', got {self.side!r}"
            )

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the ID of the lanelet beside the pick."""
        from .constraints import create_routing_graph

        if routing_graph is None:
            routing_graph = create_routing_graph(lanelet_map)
        lanelet = lanelet_map.laneletLayer[lanelet_id]
        beside = (
            routing_graph.left(lanelet)
            if self.side == "left"
            else routing_graph.right(lanelet)
        )
        if beside is None:
            raise ValueError(
                f"[{self.target_key}] lanelet {lanelet_id} has no lane to change "
                f"into on its {self.side}."
            )
        return BindingResult(value=beside.id)


@dataclass
class MatchedBinding:
    """The lanelet the sweep picked, by ID: another place on the same lane::

        bindings:
          scenario.spawn_overrides.npc1.lanelet_id:
            type: matched

    What a case means by "the ego's lane" when the ego's own spawn has been
    walked back off it by ``stop_line_offset``, and how a second slot -- a
    pedestrian's spawn, the ego's goal -- follows the pick.
    """

    target_key: str

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the pick itself."""
        del lanelet_map, routing_graph
        return BindingResult(value=lanelet_id)


@dataclass
class StopLineApproachBinding:
    """The lanelet ``stop_line_offset`` with the same offset spawns on, by ID::

        bindings:
          scenario.param_overrides.p_stop.lanelet_id:
            type: stop_line_approach
            offset: 15.0

    The pick when the offset fits on it, else the predecessor it walks back
    to: the lanelet a spawn *offset* metres before the stop line is on, which
    is where a search for that stop line has to start from.
    """

    target_key: str
    offset: float = 15.0

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the lanelet the offset lands on."""
        landed = StopLineOffsetBinding(self.target_key, self.offset).resolve(
            lanelet_id, lanelet_map, routing_graph
        )
        return BindingResult(value=landed.lanelet_id_override or lanelet_id)


#: The lanes :class:`RouteOffsetBinding` can land beside the pick's own.
ROUTE_OFFSET_SIDES = ("same", "left", "right", "opposite")
#: Which side of the road traffic keeps to: ``right`` (CARLA's towns, most of
#: the world) or ``left`` (Japan, the UK).  It decides where the oncoming lane
#: is -- across the centre line, which is on the driver's left when traffic
#: keeps right and on the driver's right when it keeps left.
TRAFFIC_SIDES = ("right", "left")


#: How far to the left of a lane's middle an oncoming lane's middle may lie.
_OPPOSITE_SEARCH_RADIUS_M = 15.0
#: How many lanelets nearest a lane's middle are considered for its oncoming one.
_OPPOSITE_CANDIDATES = 24


def _unit_direction(points: list[Any], index: int) -> tuple[float, float]:
    """The unit direction of a polyline at its *index*-th point."""
    a = points[max(index - 1, 0)]
    b = points[min(index + 1, len(points) - 1)]
    dx, dy = b.x - a.x, b.y - a.y
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def _turns(lanelet: Any) -> bool:
    """Whether *lanelet* turns through a junction (left or right, not straight)."""
    if "turn_direction" not in lanelet.attributes:
        return False
    return lanelet.attributes["turn_direction"] in ("left", "right")


def _opposite_lanelet(
    lanelet: Any, lanelet_map: Any, traffic_side: str = "right"
) -> Any | None:
    """The nearest lanelet running the other way across the centre, if any.

    The routing graph only knows lanes a vehicle may change into, and maps do
    not agree on whether the two directions share a centre line (Nishi-Shinjuku
    has a median), so it is answered geometrically: the closest lanelet whose
    middle lies on the centre-line side within
    :data:`_OPPOSITE_SEARCH_RADIUS_M` and points the other way -- the left
    where traffic keeps right, the right where it keeps left.  Turning junction
    lanelets are skipped -- they cross, not oppose -- but one going straight
    through is the oncoming lane inside the junction.
    """
    toward = 1.0 if traffic_side == "right" else -1.0
    centre = list(lanelet2.geometry.to2D(lanelet.centerline))
    if len(centre) < 2:
        return None
    middle = centre[len(centre) // 2]
    hx, hy = _unit_direction(centre, len(centre) // 2)
    best: tuple[float, Any] | None = None
    nearby = lanelet_map.laneletLayer.nearest(
        lanelet2.core.BasicPoint2d(middle.x, middle.y), _OPPOSITE_CANDIDATES
    )
    for other in nearby:
        if other.id == lanelet.id or _turns(other):
            continue
        points = list(lanelet2.geometry.to2D(other.centerline))
        if len(points) < 2:
            continue
        index = min(
            range(len(points)),
            key=lambda i: (points[i].x - middle.x) ** 2 + (points[i].y - middle.y) ** 2,
        )
        dx, dy = points[index].x - middle.x, points[index].y - middle.y
        gap = math.hypot(dx, dy)
        # On the centre-line side of the lane (the cross product is positive
        # to its left) and within reach.
        if gap > _OPPOSITE_SEARCH_RADIUS_M or toward * (hx * dy - hy * dx) <= 0:
            continue
        ox, oy = _unit_direction(points, index)
        if hx * ox + hy * oy > -0.9:
            continue
        if best is None or gap < best[0]:
            best = (gap, other)
    return None if best is None else best[1]


def route_offset_pose(
    lanelet_id: int,
    lanelet_map: Any,
    routing_graph: Any,
    *,
    distance: float,
    side: str = "same",
    traffic_side: str = "right",
) -> tuple[int, float]:
    """Where ``distance`` metres along the road from the pick's start lands.

    The walk follows the pick's lane -- forwards through the lowest-id
    following lanelet, as :class:`RouteThroughBinding` does, and backwards
    through the first predecessor, as :class:`StopLineOffsetBinding` does, so a
    point measured from an ego that offset walked back lands on the ego's own
    approach.  The lane it ends on is then swapped for the one on ``side``,
    with the point projected across onto it; ``opposite`` is the oncoming lane,
    found across the centre line on the side ``traffic_side`` puts it.

    Returns:
        ``(lanelet_id, s)`` of the landed point.

    Raises:
        ValueError: If the road ends before ``distance`` or the landed lanelet
            has no lane on ``side``: the case is dropped.
    """
    if side not in ROUTE_OFFSET_SIDES:
        raise ValueError(
            f"route offset side must be one of {ROUTE_OFFSET_SIDES}, got {side!r}"
        )
    if traffic_side not in TRAFFIC_SIDES:
        raise ValueError(
            f"traffic side must be one of {TRAFFIC_SIDES}, got {traffic_side!r}"
        )
    current = lanelet_map.laneletLayer[lanelet_id]
    s = float(distance)
    length = lanelet2.geometry.length2d(current)
    while s > length:
        following = sorted(routing_graph.following(current), key=lambda ll: ll.id)
        if not following:
            raise ValueError(
                f"lanelet {current.id} has no following lanelet; "
                f"{distance} m from lanelet {lanelet_id} runs off the road."
            )
        s -= length
        current = following[0]
        length = lanelet2.geometry.length2d(current)
    while s < 0:
        previous = routing_graph.previous(current)
        if not previous:
            raise ValueError(
                f"lanelet {current.id} has no previous lanelet; "
                f"{distance} m from lanelet {lanelet_id} runs off the road."
            )
        current = previous[0]
        length = lanelet2.geometry.length2d(current)
        s += length
    if side == "same":
        return current.id, s
    if side == "opposite":
        # Measured from the innermost lane of this direction -- the one next to
        # the centre line -- so a pick in the middle of a wide road still finds
        # the lane across it.
        inward = routing_graph.left if traffic_side == "right" else routing_graph.right
        innermost = current
        while (further := inward(innermost)) is not None:
            innermost = further
        beside = _opposite_lanelet(innermost, lanelet_map, traffic_side)
    else:
        beside = (
            routing_graph.left(current)
            if side == "left"
            else routing_graph.right(current)
        )
    if beside is None:
        raise ValueError(f"lanelet {current.id} has no {side} lane.")
    # The same spot, projected across onto the other lane.
    point = lanelet2.geometry.interpolatedPointAtDistance(
        lanelet2.geometry.to2D(current.centerline), s
    )
    across = lanelet2.geometry.toArcCoordinates(
        lanelet2.geometry.to2D(beside.centerline), point
    ).length
    return beside.id, min(max(across, 0.0), lanelet2.geometry.length2d(beside))


@dataclass
class RouteOffsetBinding:
    """The lanelet ``distance`` metres along the road from the pick, by ID::

        bindings:
          scenario.spawn_overrides.npc1.lanelet_id:
            type: route_offset
            distance: 40.0
            side: left

    Where another vehicle is *relative to* the case: 40 m ahead in the lane to
    the left.  Pair it with :class:`RouteOffsetSBinding` on the same entity's
    ``s`` -- this names the lanelet, that the offset along it.

    ``traffic_side`` is the map's, not the scenario's: the exported config
    fills it from ``${map.traffic_side}``.
    """

    target_key: str
    distance: float = 0.0
    side: str = "same"
    traffic_side: str = "right"

    def __post_init__(self) -> None:
        if self.side not in ROUTE_OFFSET_SIDES:
            raise ValueError(
                f"route_offset side must be one of {ROUTE_OFFSET_SIDES}, "
                f"got {self.side!r}"
            )
        if self.traffic_side not in TRAFFIC_SIDES:
            raise ValueError(
                f"route_offset traffic_side must be one of {TRAFFIC_SIDES}, "
                f"got {self.traffic_side!r}"
            )

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the ID of the lanelet the offset lands on."""
        from .constraints import create_routing_graph

        if routing_graph is None:
            routing_graph = create_routing_graph(lanelet_map)
        landed, _ = route_offset_pose(
            lanelet_id,
            lanelet_map,
            routing_graph,
            distance=self.distance,
            side=self.side,
            traffic_side=self.traffic_side,
        )
        return BindingResult(value=landed)


@dataclass
class RouteOffsetSBinding(RouteOffsetBinding):
    """The ``s`` on the lanelet :class:`RouteOffsetBinding` lands on.

    Unlike :class:`StopLineOffsetBinding` it never moves the *searched* slot:
    it describes a point relative to the pick, so it may sit on any entity's
    spawn, not only the swept one's.
    """

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the offset along the landed lanelet."""
        from .constraints import create_routing_graph

        if routing_graph is None:
            routing_graph = create_routing_graph(lanelet_map)
        _, s = route_offset_pose(
            lanelet_id,
            lanelet_map,
            routing_graph,
            distance=self.distance,
            side=self.side,
            traffic_side=self.traffic_side,
        )
        return BindingResult(value=round(s, 3))


def _graph(lanelet_map: Any, routing_graph: Any | None) -> Any:
    from .constraints import create_routing_graph

    return (
        routing_graph
        if routing_graph is not None
        else create_routing_graph(lanelet_map)
    )


@dataclass
class CrossingBinding:
    """Where a vehicle crossing the case's path starts, by lanelet ID::

        bindings:
          scenario.spawn_overrides.npc1.lanelet_id:
            type: crossing
            side: left
            approach: 15.0

    The lane approaching the junction lanelet that crosses the ego's path from
    ``side`` (straight on preferred), ``approach`` metres before the junction.
    Pair it with :class:`CrossingSBinding` on the same spawn's ``s``, and with
    ``has_crossing`` on the search.
    """

    target_key: str
    side: str = "left"
    approach: float = 15.0

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(
                f"crossing side must be 'left' or 'right', got {self.side!r}"
            )

    def _pose(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any
    ) -> tuple[int, float]:
        from .topology import crossing_approach  # noqa: PLC0415

        return crossing_approach(
            lanelet_map.laneletLayer[lanelet_id],
            lanelet_map,
            _graph(lanelet_map, routing_graph),
            self.side,
            self.approach,
        )

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the ID of the lane the crossing vehicle starts on."""
        return BindingResult(
            value=self._pose(lanelet_id, lanelet_map, routing_graph)[0]
        )


@dataclass
class CrossingSBinding(CrossingBinding):
    """The ``s`` on the lane :class:`CrossingBinding` names."""

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the offset along the crossing vehicle's lane."""
        s = self._pose(lanelet_id, lanelet_map, routing_graph)[1]
        return BindingResult(value=round(s, 3))


@dataclass
class CrosswalkBinding:
    """The crosswalk across the case's lane, by lanelet ID::

        bindings:
          scenario.spawn_overrides.ped1.lanelet_id:
            type: crosswalk
            side: left

    The first crosswalk across the lane within ``search_distance`` metres of the
    pick's start.  :class:`CrosswalkSBinding` puts a pedestrian at its
    ``side`` kerb and :class:`CrosswalkHeadingBinding` faces it across; pair
    all three, and ``has_crosswalk_ahead`` on the search.
    """

    target_key: str
    side: str = "left"
    search_distance: float = 60.0

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(
                f"crosswalk side must be 'left' or 'right', got {self.side!r}"
            )

    def _start(self, lanelet_id: int, lanelet_map: Any, routing_graph: Any) -> Any:
        from .topology import crosswalk_start  # noqa: PLC0415

        return crosswalk_start(
            lanelet_map.laneletLayer[lanelet_id],
            lanelet_map,
            _graph(lanelet_map, routing_graph),
            self.side,
            self.search_distance,
        )

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the crosswalk's lanelet ID."""
        return BindingResult(
            value=self._start(lanelet_id, lanelet_map, routing_graph).lanelet_id
        )


@dataclass
class CrosswalkSBinding(CrosswalkBinding):
    """The ``s`` at the crosswalk's ``side`` kerb."""

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the offset along the crosswalk."""
        return BindingResult(
            value=round(self._start(lanelet_id, lanelet_map, routing_graph).s, 3)
        )


@dataclass
class CrosswalkHeadingBinding(CrosswalkBinding):
    """The heading that walks the crosswalk from its ``side`` kerb: 0 or pi."""

    def resolve(
        self, lanelet_id: int, lanelet_map: Any, routing_graph: Any | None = None
    ) -> BindingResult:
        """Return the heading relative to the crosswalk's direction."""
        return BindingResult(
            value=round(self._start(lanelet_id, lanelet_map, routing_graph).heading, 6)
        )


_BINDING_REGISTRY: dict[str, type] = {
    "stop_line_offset": StopLineOffsetBinding,
    "route_through": RouteThroughBinding,
    "adjacent": AdjacentBinding,
    "matched": MatchedBinding,
    "stop_line_approach": StopLineApproachBinding,
    "route_offset": RouteOffsetBinding,
    "route_offset_s": RouteOffsetSBinding,
    "crossing": CrossingBinding,
    "crossing_s": CrossingSBinding,
    "crosswalk": CrosswalkBinding,
    "crosswalk_s": CrosswalkSBinding,
    "crosswalk_heading": CrosswalkHeadingBinding,
}


def register_binding(type_id: str, cls: type) -> None:
    """Make a binding available to ``sweep.bindings`` as *type_id*.

    *cls* is built as ``cls(target_key=..., **params)`` and satisfies
    :class:`Binding`.
    """
    _BINDING_REGISTRY[type_id] = cls


def parse_binding(target_key: str, cfg: dict[str, Any]) -> Binding:
    """Instantiate a :class:`Binding` from a YAML mapping.

    Args:
        target_key: The Hydra override key (e.g. ``"ego.spawn_s"``).
        cfg: A dict with at least a ``type`` key and optional parameters.

    Returns:
        The corresponding binding instance.

    Raises:
        ValueError: If the ``type`` value is not recognised.
    """
    binding_type = cfg.get("type")
    if binding_type is None:
        raise ValueError(f"Binding config is missing 'type': {cfg}")
    cls = _BINDING_REGISTRY.get(binding_type)
    if cls is None:
        from ..extensions import load_extensions  # noqa: PLC0415

        load_extensions()
        cls = _BINDING_REGISTRY.get(binding_type)
    if cls is None:
        raise ValueError(
            f"Unknown binding type: {binding_type!r}. "
            f"Available: {list(_BINDING_REGISTRY)}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return cls(target_key=target_key, **kwargs)
