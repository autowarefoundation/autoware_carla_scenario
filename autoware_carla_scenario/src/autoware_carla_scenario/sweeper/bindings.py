"""Binding resolution for auto-deriving parameter values per lanelet.

Bindings are parsed from the ``sweep.bindings`` section of a scenario YAML.
Each binding computes a parameter value (e.g. ``ego.spawn_s``) as a function
of the matched lanelet (e.g. "15 m before the stop line").
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

import lanelet2.core
import lanelet2.geometry

from ..lanelet_junction import is_junction_lanelet
from ..utils.stop_line import _collect_stop_lines_from_reg_elems
from ..value_text import shown

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet, ConstLineString3d, LaneletMap
    from lanelet2.routing import RoutingGraph

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
        lanelet_id_override: If set, the spawn whose ``s`` this binding
            writes is on this lanelet, not the pick, because the pick did not
            have enough distance to satisfy the offset.  The sweep moves that
            spawn's own lanelet key (``ego.spawn_s`` ->
            ``ego.spawn_lanelet_id``, ``<entity>.s`` -> ``<entity>.lanelet_id``),
            not the sweep target's.
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
        self,
        lanelet_id: int,
        lanelet_map: LaneletMap,
        routing_graph: Optional[RoutingGraph] = None,
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

    def __post_init__(self) -> None:
        _check_offset("stop_line_offset", self.offset)

    @staticmethod
    def _project_stop_line(ls: ConstLineString3d, lanelet: ConstLanelet) -> float:
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
        lanelet: ConstLanelet,
        lanelet_map: LaneletMap,
        routing_graph: RoutingGraph,
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
        current: ConstLanelet = lanelet
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
        self,
        lanelet_id: int,
        lanelet_map: LaneletMap,
        routing_graph: Optional[RoutingGraph] = None,
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
        # A Lanelet, then the routing graph's ConstLanelets.
        lanelet: ConstLanelet = lanelet_map.laneletLayer[lanelet_id]
        spawn_lanelet_length = lanelet2.geometry.length2d(lanelet)

        if routing_graph is None:
            from .constraints import create_routing_graph

            routing_graph = create_routing_graph(lanelet_map)

        # BFS over the lanelet chain: spawn -> following -> following -> ...
        # Each entry is (lanelet_object, arc_length_from_spawn_lanelet_start).
        candidates: list[tuple[ConstLanelet, float]] = [(lanelet, 0.0)]
        visited: set[int] = {lanelet_id}
        seen_ids: set[int] = set()

        while candidates:
            next_candidates: list[tuple[ConstLanelet, float]] = []
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


def _check_offset(kind: str, value: Any) -> None:
    """Refuse a stop-line *offset* that is no finite number of metres >= 0.

    An int or a float, or another real number (numpy's ``int64`` or
    ``float32``, say); never a bool, which would be 1 m or 0 m (#1951).  An
    int too large for a float is no finite offset either.
    """
    number = math.nan
    if isinstance(value, bool):
        pass
    elif isinstance(value, int) or isinstance(value, float):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
    elif TYPE_CHECKING:
        pass
    else:
        import numbers  # noqa: PLC0415 -- run time only

        # numpy registers its integers and floats as numbers.Real, and its
        # bool_ as none of them.
        if isinstance(value, numbers.Real):
            try:
                number = float(value)
            except OverflowError:
                number = math.inf
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"{kind} offset: a finite number of metres >= 0, got {shown(value)}"
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

    With ``through_junction: true`` the route goes on past ``depth`` while its
    last lanelet is still in a junction (it has a ``turn_direction`` tag), to
    the first lanelet out of it.  A turn's junction can be several lanelets
    long, and a route that ends inside it never says which way out the turn
    leads::

        bindings:
          scenario.expected_route_lanelet_ids:
            type: route_through
            depth: 1
            through_junction: true

    Where the graph forks, the lowest lanelet ID is taken, so the same map
    always expands to the same cases.  A pick the route cannot be walked from
    raises, and the caller (:func:`~autoware_carla_scenario.sweeper.expand.expand_sweep`)
    drops that case.
    """

    target_key: str
    depth: int = 1
    #: Only the route's last lanelet, as an ID rather than a list: a goal.
    last_only: bool = False
    #: Walk on past ``depth`` until the route's last lanelet is out of a
    #: junction.
    through_junction: bool = False

    def __post_init__(self) -> None:
        # Read by their truth, so the text "off" would turn one on (#1946).
        if not isinstance(self.last_only, bool):
            raise ValueError(
                f"route_through last_only must be true or false, got {self.last_only!r}"
            )
        if not isinstance(self.through_junction, bool):
            raise ValueError(
                "route_through through_junction must be true or false, "
                f"got {self.through_junction!r}"
            )
        # A whole number: "2" would fail the comparison with a TypeError, and
        # true would be read as 1 (#1951).
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise ValueError(
                f"route_through depth must be a whole number >= 1, got {shown(self.depth)}"
            )
        if self.depth < 1:
            raise ValueError(f"route_through depth must be >= 1, got {self.depth}")

    def resolve(
        self,
        lanelet_id: int,
        lanelet_map: LaneletMap,
        routing_graph: Optional[RoutingGraph] = None,
    ) -> BindingResult:
        """Return the picked lanelet and the ``depth`` lanelets after it, and
        with ``through_junction`` the rest of the junction the last is in."""
        from .constraints import create_routing_graph

        if routing_graph is None:
            routing_graph = create_routing_graph(lanelet_map)
        route = [lanelet_id]
        # A Lanelet, then the routing graph's ConstLanelets.
        current: ConstLanelet = lanelet_map.laneletLayer[lanelet_id]
        steps = 0
        while steps < self.depth or (
            self.through_junction and is_junction_lanelet(current)
        ):
            steps += 1
            following = sorted(routing_graph.following(current), key=lambda ll: ll.id)
            if not following:
                raise ValueError(
                    f"[{self.target_key}] lanelet {current.id} has no following "
                    f"lanelet; cannot walk a route of depth {self.depth} from "
                    f"lanelet {lanelet_id}."
                )
            current = following[0]
            if current.id in route:
                raise ValueError(
                    f"[{self.target_key}] the route from lanelet {lanelet_id} "
                    f"comes back to lanelet {current.id} without leaving the "
                    "junction."
                )
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
        self,
        lanelet_id: int,
        lanelet_map: LaneletMap,
        routing_graph: Optional[RoutingGraph] = None,
    ) -> BindingResult:
        """Return the ID of the lanelet beside the pick."""
        from .constraints import create_routing_graph

        if routing_graph is None:
            routing_graph = create_routing_graph(lanelet_map)
        lanelet = lanelet_map.laneletLayer[lanelet_id]
        beside: Optional[ConstLanelet] = None
        if self.side == "left":
            beside = routing_graph.left(lanelet)
        else:
            beside = routing_graph.right(lanelet)
        if beside is None:
            raise ValueError(
                f"[{self.target_key}] lanelet {lanelet_id} has no lane to change "
                f"into on its {self.side}."
            )
        # Through an annotated local: Codon 0.19 hands the id of an Optional
        # lanelet to the union field unconverted (invalid LLVM IR).
        found: ConstLanelet = beside
        return BindingResult(value=found.id)


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
        self,
        lanelet_id: int,
        lanelet_map: LaneletMap,
        routing_graph: Optional[RoutingGraph] = None,
    ) -> BindingResult:
        """Return the pick itself."""
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

    def __post_init__(self) -> None:
        _check_offset("stop_line_approach", self.offset)

    def resolve(
        self,
        lanelet_id: int,
        lanelet_map: LaneletMap,
        routing_graph: Optional[RoutingGraph] = None,
    ) -> BindingResult:
        """Return the lanelet the offset lands on."""
        landed = StopLineOffsetBinding(self.target_key, self.offset).resolve(
            lanelet_id, lanelet_map, routing_graph
        )
        return BindingResult(value=landed.lanelet_id_override or lanelet_id)


#: The bindings keyed by their ``type`` string.
#: Declared here, assigned only at run time: Codon holds no class as a value,
#: and the function that reads it (parse_binding) is not compiled (UNCALLED).
_BINDING_REGISTRY: dict[str, Any]
if TYPE_CHECKING:
    pass
else:
    _BINDING_REGISTRY = {
        "stop_line_offset": StopLineOffsetBinding,
        "route_through": RouteThroughBinding,
        "adjacent": AdjacentBinding,
        "matched": MatchedBinding,
        "stop_line_approach": StopLineApproachBinding,
    }


def parse_binding(target_key: str, cfg: dict[str, Any]) -> Binding:
    """Instantiate a :class:`Binding` from a YAML mapping.

    Args:
        target_key: The Hydra override key (e.g. ``"ego.spawn_s"``).
        cfg: A dict with at least a ``type`` key and optional parameters.

    Returns:
        The corresponding binding instance.

    Raises:
        ValueError: If *cfg* is not a mapping, its ``type`` is not recognised,
            or it is given a key the binding does not take or a value it
            refuses (#1951).
    """
    from collections.abc import Mapping  # noqa: PLC0415 -- run time only

    if not isinstance(cfg, Mapping):
        raise ValueError(
            f"Binding {target_key} must be a mapping with a 'type' "
            f"({{type: ...}}), got {shown(cfg)}"
        )
    binding_type = cfg.get("type")
    if binding_type is None:
        raise ValueError(f"Binding config is missing 'type': {cfg}")
    cls = _BINDING_REGISTRY.get(binding_type) if isinstance(binding_type, str) else None
    if cls is None:
        raise ValueError(
            f"Unknown binding type: {shown(binding_type)}. "
            f"Available: {list(_BINDING_REGISTRY)}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    # A key the binding does not take is named here rather than as the
    # constructor's TypeError (#1951).
    import dataclasses  # noqa: PLC0415 -- run time only

    names = [
        f.name for f in dataclasses.fields(cls) if f.init and f.name != "target_key"
    ]
    unknown = [k for k in kwargs if k not in names]
    if unknown:
        raise ValueError(
            f"{binding_type} binding {target_key}: unknown key(s) {unknown}; it "
            f"takes {names if names else 'no keys besides type'}"
        )
    try:
        return cls(target_key=target_key, **kwargs)
    except TypeError as exc:
        raise ValueError(f"{binding_type} binding {target_key}: {exc}") from exc


def _walk_back_lanelet_key(s_key: str, lanelet_id: int) -> str:
    """The lanelet key of the spawn whose ``s`` a walk-back moved, by *s_key*.

    A ``stop_line_offset`` binding that walks back off the pick puts its spawn
    on a predecessor: that spawn's lanelet key has to say so, and no other.
    The spawn keys come in pairs, so the lanelet key is the ``s`` key's
    sibling: ``<prefix>.s`` -> ``<prefix>.lanelet_id`` (an entity's
    ``scenario.spawn_overrides.<entity>``), ``<prefix>.spawn_s`` ->
    ``<prefix>.spawn_lanelet_id`` and ``<prefix>.goal_s`` ->
    ``<prefix>.goal_lanelet_id`` (the ego's).

    Raises:
        ValueError: If *s_key* is none of those, so there is no lanelet key
            to move to *lanelet_id*.
    """
    prefix, _, leaf = s_key.rpartition(".")
    sibling = {
        "s": "lanelet_id",
        "spawn_s": "spawn_lanelet_id",
        "goal_s": "goal_lanelet_id",
    }.get(leaf)
    if sibling is None:
        raise ValueError(
            f"sweep.bindings: {s_key} walks back to lanelet {lanelet_id}, but "
            "is not a spawn's s (<prefix>.s, <prefix>.spawn_s or "
            "<prefix>.goal_s), so there is no lanelet key to move with it. "
            "Bind stop_line_offset to a spawn's s, or use an offset that fits "
            "on the pick."
        )
    return f"{prefix}.{sibling}" if prefix else sibling
