"""Check a scenario's planned route against an ODD before it runs.

Before a run, the ego's route can be planned on the Lanelet2 map alone: from
its spawn pose, through its waypoints, to its goal, as the shortest path in
the map's routing graph (the path Autoware plans too, give or take its own
costs).  The ODD's attributes that the map decides are then read lanelet by
lanelet along it, and the ODD's modules judge each lanelet:

* **outside**: the ODD fails whatever the attributes the map cannot decide
  turn out to be at run time;
* **inside**: the ODD holds, whatever they turn out to be;
* **undecided**: it depends on them (weather, traffic, speed) or on values
  the map does not have (a lanelet with no ``location`` tag).  At run time a
  missing value counts as inside, so this is "inside only by assumption".

The map decides an attribute when its probe has an ``on_lanelet`` form
(``probe.on_lanelet(lanelet, lanelet_map, routing_graph)``): the built-in
Lanelet2 probes, ``in_junction`` and ``lane_count`` do (see
:mod:`~autoware_carla_scenario.odd.probes`), and a custom probe opts in by
having one.  Every other attribute is open: it could take any value.  An
attribute derived from others (OpenODD) is worked out from their values on
the lanelet, and is open when one of them is.

The metres the route spends in each bucket of each attribute the map decides
are the run's *expected* coverage, and summed over a batch of scenarios they
say which buckets no planned route reaches at all.

Two simplifications, both in the length rather than in the verdict: a lane
change splits the length of the lanelets on either side of it between them,
and a goal behind the start on the same lanelet is a route of no length (the
routing graph is not asked to go round the block).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

from ..coverage.items import value_label
from .model import _OPEN, OddAttribute, OddDefinition, _derived, read_probe
from .registry import resolve_odd

if TYPE_CHECKING:
    from ..coordinate.poses import Lanelet2Pose
    from ..scenario_base import BaseScenario

__all__ = [
    "MISSING_BUCKET",
    "UNDECIDED_BUCKET",
    "PlannedRoute",
    "RouteCoverage",
    "RouteCoverageSummary",
    "RouteError",
    "RouteLanelet",
    "combine_route_coverage",
    "plan_route",
    "plan_route_coverage",
]

logger = logging.getLogger(__name__)

#: Pseudo-bucket for metres on lanelets where the map says nothing (``None``).
MISSING_BUCKET = "<missing>"
#: Pseudo-bucket for metres on lanelets where only the run can tell the value.
UNDECIDED_BUCKET = "<undecided>"

INSIDE = "inside"
OUTSIDE = "outside"
UNDECIDED = "undecided"


class RouteError(ValueError):
    """A route that cannot be planned: no goal, an unknown lanelet, no path."""


@dataclass(frozen=True)
class PlannedRoute:
    """Where a route starts, the lanelets it passes through, and where it ends.

    Args:
        start: The ego's initial pose.
        goal: Where it is routed to.  ``None`` cannot be planned.
        via: Poses the route passes through, in order (their lanelets).
        name: What the report calls it.
    """

    start: "Lanelet2Pose"
    goal: Optional["Lanelet2Pose"] = None
    via: Sequence["Lanelet2Pose"] = ()
    name: str = ""

    @classmethod
    def of_scenario(cls, scenario: "BaseScenario", name: str = "") -> "PlannedRoute":
        """The route a scenario's ego will be given: spawn, waypoints, goal.

        Read before ``setup()``, so a goal or waypoints the scenario only
        derives there are not known yet.

        Raises:
            RouteError: if the scenario has no spawn pose.
        """
        start = getattr(scenario, "_spawn_pose", None)
        if start is None:
            raise RouteError(
                f"{type(scenario).__name__} names no spawn pose to start a route at"
            )
        return cls(
            start=start,
            goal=scenario.goal_pose,
            via=tuple(scenario.waypoint_poses),
            name=name or type(scenario).__name__,
        )


@dataclass
class RouteLanelet:
    """One lanelet of a planned route, and the ODD's verdict on it."""

    lanelet_id: int
    #: Metres of the route on this lanelet.
    length_m: float
    #: ``"inside"``, ``"outside"`` or ``"undecided"``.
    verdict: str
    #: Every attribute's value on it, by name: ``None`` when the map says
    #: nothing, :data:`~autoware_carla_scenario.odd.model.UNDECIDED` when
    #: only the run can tell.
    values: dict[str, Any] = field(default_factory=dict)
    #: The modules that do not hold on it (an outside lanelet's reasons).
    failing_modules: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "lanelet_id": self.lanelet_id,
            "length_m": round(self.length_m, 3),
            "verdict": self.verdict,
            "values": {k: _describe_value(v) for k, v in self.values.items()},
            "failing_modules": list(self.failing_modules),
        }


@dataclass
class RouteCoverage:
    """An ODD's verdict on one planned route, and the coverage it is expected to give."""

    odd: str
    name: str
    lanelets: list[RouteLanelet]
    #: Attribute name -> bucket label -> metres, for the attributes with
    #: buckets.  :data:`UNDECIDED_BUCKET` and :data:`MISSING_BUCKET` hold the
    #: metres where the map does not decide the value or has none.
    expected_m: dict[str, dict[str, float]]
    #: The attributes the map decides (their probe reads a lanelet).
    map_attributes: list[str]

    def _metres(self, verdict: str) -> float:
        return sum(ll.length_m for ll in self.lanelets if ll.verdict == verdict)

    @property
    def length_m(self) -> float:
        return sum(ll.length_m for ll in self.lanelets)

    @property
    def inside_m(self) -> float:
        return self._metres(INSIDE)

    @property
    def outside_m(self) -> float:
        return self._metres(OUTSIDE)

    @property
    def undecided_m(self) -> float:
        return self._metres(UNDECIDED)

    @property
    def leaves_odd(self) -> bool:
        """Whether the route drives on a lanelet outside the ODD whatever the run does."""
        return bool(self.outside())

    @property
    def lanelet_ids(self) -> list[int]:
        return [ll.lanelet_id for ll in self.lanelets]

    def outside(self) -> list[RouteLanelet]:
        """The lanelets the route drives on that are outside the ODD whatever the run does.

        A lanelet the route only touches (a goal at its very start) is not
        driven on, so it does not take the route outside.
        """
        return [
            ll for ll in self.lanelets if ll.verdict == OUTSIDE and ll.length_m > 0.0
        ]

    def undecided(self) -> list[RouteLanelet]:
        """The lanelets inside the ODD only if the run's values allow it."""
        return [ll for ll in self.lanelets if ll.verdict == UNDECIDED]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "odd": self.odd,
            "lanelet_ids": self.lanelet_ids,
            "length_m": round(self.length_m, 3),
            "inside_m": round(self.inside_m, 3),
            "undecided_m": round(self.undecided_m, 3),
            "outside_m": round(self.outside_m, 3),
            "leaves_odd": self.leaves_odd,
            "map_attributes": list(self.map_attributes),
            "expected_m": _round_buckets(self.expected_m),
            "lanelets": [ll.describe() for ll in self.lanelets],
        }


@dataclass
class RouteCoverageSummary:
    """The routes of a batch of scenarios together."""

    odd: str
    routes: list[RouteCoverage]
    #: Attribute name -> bucket label -> metres, summed over the routes.
    expected_m: dict[str, dict[str, float]]
    #: Attribute name -> the labels of its buckets inside the ODD (not ruled
    #: out by it) that no route reaches, for the attributes the map decides.
    unreached: dict[str, list[str]]
    #: Attributes the map decides in general but not on every lanelet of
    #: these routes (metres in :data:`UNDECIDED_BUCKET`): the run may reach
    #: any of their buckets there, so none is called unreached.
    undetermined: list[str] = field(default_factory=list)

    @property
    def leaving(self) -> list[str]:
        """The routes that leave the ODD."""
        return [r.name for r in self.routes if r.leaves_odd]

    def describe(self) -> dict[str, Any]:
        return {
            "odd": self.odd,
            "routes": [r.name for r in self.routes],
            "leaving_odd": self.leaving,
            "expected_m": _round_buckets(self.expected_m),
            "unreached": {k: list(v) for k, v in self.unreached.items()},
            "undetermined": list(self.undetermined),
        }


def _describe_value(value: Any) -> Any:
    if value is _OPEN:
        return UNDECIDED
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return value_label(value)


def _round_buckets(buckets: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {a: {k: round(v, 3) for k, v in b.items()} for a, b in buckets.items()}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _lanelet(lanelet_map: Any, lanelet_id: int, what: str) -> Any:
    layer = lanelet_map.laneletLayer
    if not layer.exists(int(lanelet_id)):
        raise RouteError(f"the {what} names lanelet {lanelet_id}, not in the map")
    return layer[int(lanelet_id)]


def plan_route(
    route: PlannedRoute, lanelet_map: Any, routing_graph: Any
) -> list[tuple[int, float]]:
    """The lanelets of *route*, in order, with the metres driven on each.

    Each leg (start to the first waypoint, ... to the goal) is the shortest
    path in *routing_graph*, lane changes allowed.

    Raises:
        RouteError: if the route has no goal, names a lanelet the map does
            not have, or a leg has no path.
    """
    if route.goal is None:
        raise RouteError("no goal: the route cannot be planned")
    import lanelet2.geometry  # noqa: PLC0415

    stops = [
        (_lanelet(lanelet_map, route.start.lanelet_id, "start"), "start"),
        *(
            (_lanelet(lanelet_map, pose.lanelet_id, f"waypoint {i}"), f"waypoint {i}")
            for i, pose in enumerate(route.via, 1)
        ),
        (_lanelet(lanelet_map, route.goal.lanelet_id, "goal"), "goal"),
    ]
    path: list[Any] = [stops[0][0]]
    for there, what in stops[1:]:
        if there.id == path[-1].id:
            continue  # already on it
        leg = routing_graph.shortestPath(path[-1], there)
        if leg is None:
            raise RouteError(
                f"no route from lanelet {path[-1].id} to lanelet {there.id} ({what})"
            )
        path.extend(list(leg)[1:])

    # Metres on each: the start lanelet from the start pose on, the goal
    # lanelet up to the goal pose.
    last = len(path) - 1
    driven: list[float] = []
    for i, lanelet in enumerate(path):
        length = float(lanelet2.geometry.length2d(lanelet))
        begin = _clamp(route.start.s, length) if i == 0 else 0.0
        end = _clamp(route.goal.s, length) if i == last else length
        driven.append(max(0.0, end - begin))

    # A lane change: the vehicle drives along the two lanelets side by side,
    # so their length is split between them rather than counted twice.
    lateral = [0] * len(path)
    for i in range(last):
        following = {ll.id for ll in routing_graph.following(path[i])}
        if path[i + 1].id not in following:
            lateral[i] += 1
            lateral[i + 1] += 1
    return [
        (int(lanelet.id), metres / (1 + sides))
        for lanelet, metres, sides in zip(path, driven, lateral)
    ]


def _clamp(s: float, length: float) -> float:
    return min(max(float(s), 0.0), length)


# ---------------------------------------------------------------------------
# Evaluating the route
# ---------------------------------------------------------------------------


def _on_lanelet(attribute: OddAttribute) -> Any:
    return getattr(attribute.probe, "on_lanelet", None)


def _map_attributes(odd: OddDefinition) -> list[str]:
    """The attributes the map decides: their probe reads a lanelet.

    A derived attribute is one when every attribute it derives from is.
    """
    decided = {a.name for a in odd.attributes if callable(_on_lanelet(a))}
    for attribute in odd.attributes:
        sources = getattr(attribute.probe, "sources", None)
        if _derived(attribute.probe) and sources:
            if all(getattr(s, "name", None) in decided for s in sources):
                decided.add(attribute.name)
    return [a.name for a in odd.attributes if a.name in decided]


def _read_on_lanelet(read: Any, lanelet: Any, lanelet_map: Any, graph: Any) -> Any:
    try:
        return read(lanelet, lanelet_map, graph)
    except Exception:  # noqa: BLE001 - a probe that raises reads nothing
        logger.debug("on_lanelet failed on lanelet %s", lanelet.id, exc_info=True)
        return None


def _lanelet_values(
    odd: OddDefinition, lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> dict[str, Any]:
    """Every attribute's value on *lanelet*: open where the map cannot tell."""
    values: dict[str, Any] = {}
    derived: list[OddAttribute] = []
    for attribute in odd.attributes:
        if _derived(attribute.probe):
            derived.append(attribute)
            continue
        read = _on_lanelet(attribute)
        values[attribute.name] = (
            _read_on_lanelet(read, lanelet, lanelet_map, routing_graph)
            if callable(read)
            else _OPEN
        )
    for attribute in derived:
        sources = getattr(attribute.probe, "sources", None)
        names = (
            [s.name for s in sources if hasattr(s, "name")] if sources else list(values)
        )
        value = read_probe(attribute.probe.from_values, values)  # type: ignore[attr-defined]
        if value is None and any(values.get(n, _OPEN) is _OPEN for n in names):
            value = _OPEN  # missing because a source is open: open too
        values[attribute.name] = value
    return values


def _bucket(attribute: OddAttribute, value: Any) -> Optional[str]:
    item = attribute.item
    if item is None:
        return None
    if value is _OPEN:
        return UNDECIDED_BUCKET
    if value is None:
        return MISSING_BUCKET
    try:
        return item.bucket_of(value)
    except (TypeError, ValueError):
        return value_label(value)


def _route_of(scenario_or_route: Any) -> Union[PlannedRoute, list[int]]:
    if isinstance(scenario_or_route, PlannedRoute):
        return scenario_or_route
    if hasattr(scenario_or_route, "ego_config") and hasattr(
        scenario_or_route, "waypoint_poses"
    ):
        return PlannedRoute.of_scenario(scenario_or_route)
    if isinstance(scenario_or_route, (str, bytes)):
        raise TypeError("a route is a PlannedRoute, a scenario or lanelet ids")
    try:
        ids = [int(x) for x in scenario_or_route]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "a route is a PlannedRoute, a scenario or lanelet ids, not "
            f"{type(scenario_or_route).__name__}"
        ) from exc
    if not ids:
        raise RouteError("an empty route")
    return ids


def _loaded_map(routing_graph: Any) -> tuple[Any, Any]:
    """The map the ``MapManager`` has loaded, and its routing graph."""
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415

    manager = MapManager.get_instance()
    try:
        lanelet_map = manager.lanelet_map
    except RuntimeError as exc:
        raise RouteError(
            "no Lanelet2 map loaded: pass lanelet_map (and routing_graph)"
        ) from exc
    return (
        lanelet_map,
        manager.routing_graph if routing_graph is None else routing_graph,
    )


def plan_route_coverage(
    odd: Union[OddDefinition, str, None],
    scenario_or_route: Union["BaseScenario", PlannedRoute, Iterable[int]],
    *,
    lanelet_map: Any = None,
    routing_graph: Any = None,
    name: str = "",
) -> RouteCoverage:
    """The ODD's verdict on a planned route, lanelet by lanelet, and its expected coverage.

    Args:
        odd: The ODD, or what names it (as the ``odd`` config key does).
        scenario_or_route: A scenario (its spawn pose, waypoints and goal), a
            :class:`PlannedRoute`, or the route's lanelet ids in order (taken
            whole, as they are).
        lanelet_map: The Lanelet2 map.  ``None`` takes the one the
            ``MapManager`` has loaded.
        routing_graph: Its routing graph; built from *lanelet_map* (with the
            framework's traffic rules) when ``None``.
        name: What the report calls the route.

    Raises:
        RouteError: if the route cannot be planned, or there is no map.
    """
    odd = resolve_odd(odd)
    route = _route_of(scenario_or_route)
    if lanelet_map is None:
        lanelet_map, routing_graph = _loaded_map(routing_graph)
    if routing_graph is None:
        from ..sweeper.constraints import create_routing_graph  # noqa: PLC0415

        routing_graph = create_routing_graph(lanelet_map)

    if isinstance(route, PlannedRoute):
        name = name or route.name
        planned = plan_route(route, lanelet_map, routing_graph)
    else:
        import lanelet2.geometry  # noqa: PLC0415

        planned = [
            (i, float(lanelet2.geometry.length2d(_lanelet(lanelet_map, i, "route"))))
            for i in route
        ]

    map_attributes = _map_attributes(odd)
    expected: dict[str, dict[str, float]] = {
        a.name: {} for a in odd.attributes if a.item is not None
    }
    lanelets: list[RouteLanelet] = []
    for lanelet_id, metres in planned:
        lanelet = lanelet_map.laneletLayer[lanelet_id]
        values = _lanelet_values(odd, lanelet, lanelet_map, routing_graph)
        verdict = odd.evaluate(values)
        lanelets.append(
            RouteLanelet(
                lanelet_id=lanelet_id,
                length_m=metres,
                verdict=(
                    OUTSIDE
                    if not verdict.inside
                    else UNDECIDED
                    if verdict.assumed
                    else INSIDE
                ),
                values=values,
                failing_modules=[m for m, v in verdict.modules.items() if v is False],
            )
        )
        if metres <= 0.0:
            continue  # touched, not driven on: no coverage
        for attribute in odd.attributes:
            label = _bucket(attribute, values[attribute.name])
            if label is not None:
                buckets = expected[attribute.name]
                buckets[label] = buckets.get(label, 0.0) + metres
    return RouteCoverage(
        odd=odd.name,
        name=name or "route",
        lanelets=lanelets,
        expected_m=_in_bucket_order(odd, expected),
        map_attributes=map_attributes,
    )


def _in_bucket_order(
    odd: OddDefinition, expected: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """*expected* with each attribute's buckets in its cover item's order, then the rest."""
    out: dict[str, dict[str, float]] = {}
    for attribute in odd.attributes:
        if attribute.item is None or attribute.name not in expected:
            continue
        buckets = expected[attribute.name]
        order = {label: i for i, label in enumerate(attribute.item.labels)}
        out[attribute.name] = dict(
            sorted(
                buckets.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])
            )
        )
    for name, buckets in expected.items():
        out.setdefault(name, buckets)
    return out


def combine_route_coverage(
    odd: Union[OddDefinition, str, None], routes: Sequence[RouteCoverage]
) -> RouteCoverageSummary:
    """The expected coverage of *routes* together, and the buckets none reaches.

    Only the attributes the map decides are said to leave buckets unreached:
    the others are read at run time, which no route says anything about.  A
    bucket the ODD rules out is not a target, so it is not unreached either.
    An attribute the map leaves undecided on some lanelet of a route (a
    ``speed_limit_kph`` lanelet with no tag) could reach any bucket there: it
    is reported as undetermined rather than with unreached buckets.
    """
    odd = resolve_odd(odd)
    expected: dict[str, dict[str, float]] = {
        a.name: {} for a in odd.attributes if a.item is not None
    }
    for route in routes:
        for attribute_name, buckets in route.expected_m.items():
            total = expected.setdefault(attribute_name, {})
            for label, metres in buckets.items():
                total[label] = total.get(label, 0.0) + metres
    expected = _in_bucket_order(odd, expected)
    map_attributes = set(_map_attributes(odd))
    unreached: dict[str, list[str]] = {}
    undetermined: list[str] = []
    for attribute in odd.attributes:
        item = attribute.item
        if item is None or attribute.name not in map_attributes:
            continue
        if expected.get(attribute.name, {}).get(UNDECIDED_BUCKET, 0.0) > 0.0:
            undetermined.append(attribute.name)
            continue
        outside = set(odd.outside_buckets(attribute))
        reached = expected.get(attribute.name, {})
        missing = [
            label
            for label in item.labels
            if label not in outside and reached.get(label, 0.0) <= 0.0
        ]
        if missing:
            unreached[attribute.name] = missing
    return RouteCoverageSummary(
        odd=odd.name,
        routes=list(routes),
        expected_m=expected,
        unreached=unreached,
        undetermined=undetermined,
    )
