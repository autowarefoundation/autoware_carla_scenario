"""Scenario measures: what a run sets up, read off the world under fixed keys.

A scenario stages road users -- a vehicle ahead in the next lane, a pedestrian
running out -- and its parameters say how: how far ahead, how fast.  A
**measure** reads such a quantity off the world while the run drives, under a
key the framework fixes (:data:`VEHICLE_AHEAD_GAP_M`, ...), so every scenario
speaks the same vocabulary of what it sets up, whoever wrote it.

Every :class:`~autoware_carla_scenario.BaseScenario` has the built-in measures
(:data:`BUILT_IN_MEASURES`), which read the world generically, in the ego's
frame.  Besides what a run sets up, they include how the ego drives it -- the
driving-quality measures of :mod:`autoware_carla_scenario.driving_measures`
(collisions, off the road, distances to obstacles and lane edges).  A scenario that knows better -- which actor *is* its cut-in vehicle --
replaces one with :meth:`~autoware_carla_scenario.BaseScenario.register_measure`,
or adds one of its own.

Two things are keyed by measures, and neither knows the ODD:

* a scenario's ``controls`` (its config): which of its parameters sets which
  measure, and what it can stage;
* an ODD's mapping of its taxonomy onto them
  (:func:`~autoware_carla_scenario.odd.scenario_measure`, ``measure:`` in a
  binding file), which is how coverage is measured and how a sweep draws the
  scenario's parameters from the ODD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .conditions.base import find_actor_in_list
from .constants import EGO_ROLE_NAME
from .driving_measures import DRIVING_MEASURES, reset_driving_measures

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from .scenario_base import BaseScenario

__all__ = [
    "AHEAD_RANGE_M",
    "BUILT_IN_MEASURES",
    "CROSSING_PEDESTRIAN_GAP_M",
    "CROSSING_PEDESTRIAN_SPEED_MS",
    "LANE_HALF_WIDTH_M",
    "Measure",
    "PEDESTRIAN_LATERAL_M",
    "PEDESTRIAN_MOVING_MS",
    "VEHICLE_AHEAD_GAP_M",
    "VEHICLE_AHEAD_RELATIVE_SPEED_KPH",
    "in_ego_frame",
    "measured_scenario",
    "read_measure",
    "set_measured_scenario",
    "vehicle_ahead_gap_between",
]

#: Metres to the nearest vehicle ahead in the ego's lane or a lane beside it.
VEHICLE_AHEAD_GAP_M = "vehicle_ahead_gap_m"
#: That vehicle's speed along the ego's heading less the ego's, in km/h.
VEHICLE_AHEAD_RELATIVE_SPEED_KPH = "vehicle_ahead_relative_speed_kph"
#: Metres to the nearest pedestrian ahead that is moving.
CROSSING_PEDESTRIAN_GAP_M = "crossing_pedestrian_gap_m"
#: That pedestrian's speed, in m/s.
CROSSING_PEDESTRIAN_SPEED_MS = "crossing_pedestrian_speed_ms"

#: How far ahead a road user counts as ahead of the ego, in metres.
AHEAD_RANGE_M = 100.0
#: Half a lane's width, in metres: a vehicle within this of the ego's centre
#: line is in its lane, and within three times this, in a lane beside it.
LANE_HALF_WIDTH_M = 1.75
#: How far to either side of the ego's centre line a pedestrian ahead counts.
PEDESTRIAN_LATERAL_M = 10.0
#: The speed above which a pedestrian is moving rather than standing, in m/s.
PEDESTRIAN_MOVING_MS = 0.5

_KMH_PER_MS = 3.6


@dataclass(frozen=True)
class Measure:
    """A quantity a run sets up, read off the world.

    Attributes:
        key: Its key, e.g. :data:`VEHICLE_AHEAD_GAP_M`.
        read: Reads it from the world; ``None`` when there is nothing to read.
        unit: The unit it is read in.
        text: A description.
    """

    key: str
    read: Callable[[carla.World], Optional[float]]
    unit: str = ""
    text: str = ""


# ---------------------------------------------------------------------------
# The ego's frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Around:
    """A road user, seen from the ego: metres ahead and to the side, speeds."""

    actor_id: int
    vehicle: bool
    ahead: float
    lateral: float
    #: Its velocity along the ego's heading less the ego's, in m/s.
    closing: float
    #: Its own speed, in m/s.
    speed: float


class _Cache:
    """What was read for which world at which frame: (id(world), frame), and
    the road users around the ego then."""

    at: Optional[tuple[int, int]] = None
    around: Optional[list[_Around]] = None


_CACHE = _Cache()
#: Pedestrian actor id -> how far ahead of the ego it was when it set off.
_SET_OFF: dict[int, float] = {}


def _frame_of(world: "carla.World") -> Optional[int]:
    try:
        return int(world.get_snapshot().frame)
    except Exception:
        return None


def _around(world: "carla.World") -> Optional[list[_Around]]:
    """Every other vehicle and pedestrian, in the ego's frame; once per frame."""
    frame = _frame_of(world)
    at: Optional[tuple[int, int]] = None
    if frame is not None:
        at = (id(world), int(frame))
    if at is not None and _CACHE.at is not None and at == _CACHE.at:
        return _CACHE.around
    found: Optional[list[_Around]]
    try:
        actors = list(world.get_actors())
        ego = find_actor_in_list(actors, EGO_ROLE_NAME)
        found = None if ego is None else _relative_to(ego, actors)
    except Exception:
        found = None
    _CACHE.at, _CACHE.around = at, found
    return found


def in_ego_frame(
    ego: "carla.Transform", location: "carla.Location"
) -> tuple[float, float]:
    """*location* seen from an ego at *ego*: ``(ahead, lateral)`` in metres.

    ``ahead`` is along the ego's heading, ``lateral`` across it (positive to
    the ego's right, CARLA's left-handed frame), both from the ego's origin.
    It is the frame every built-in measure reads in, so code that places a
    road user for a measure (:mod:`autoware_carla_scenario.placement`) uses it
    too rather than a definition of its own.
    """
    yaw = math.radians(ego.rotation.yaw)
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    dx = location.x - ego.location.x
    dy = location.y - ego.location.y
    return dx * forward_x + dy * forward_y, -dx * forward_y + dy * forward_x


def _counts_as_vehicle_ahead(ahead: float, lateral: float) -> bool:
    """Whether a vehicle that far ahead and aside is a vehicle ahead."""
    return 0.0 < ahead <= AHEAD_RANGE_M and abs(lateral) <= 3 * LANE_HALF_WIDTH_M


def vehicle_ahead_gap_between(
    ego: "carla.Transform", location: "carla.Location"
) -> Optional[float]:
    """What :data:`VEHICLE_AHEAD_GAP_M` reads for a vehicle at *location*, seen
    from an ego at *ego*: centre to centre along the ego's heading.

    ``None`` when the measure would not count it as ahead (behind the ego,
    beyond :data:`AHEAD_RANGE_M`, or more than a lane beside the ego's).  The
    built-in measure reads the nearest such vehicle with this same
    definition, so a vehicle placed where this reads *v*, alone ahead of the
    ego, is measured at *v*.
    """
    ahead, lateral = in_ego_frame(ego, location)
    if not _counts_as_vehicle_ahead(ahead, lateral):
        return None
    return ahead


def _relative_to(ego: "carla.Actor", actors: "list[carla.Actor]") -> list[_Around]:
    transform = ego.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    forward = (math.cos(yaw), math.sin(yaw))
    ego_velocity = ego.get_velocity()
    ego_along = ego_velocity.x * forward[0] + ego_velocity.y * forward[1]
    found: list[_Around] = []
    for actor in actors:
        kind = actor.type_id
        if actor.id == ego.id or not (
            kind.startswith("vehicle.") or kind.startswith("walker.")
        ):
            continue
        ahead, lateral = in_ego_frame(transform, actor.get_location())
        velocity = actor.get_velocity()
        found.append(
            _Around(
                actor_id=actor.id,
                vehicle=kind.startswith("vehicle."),
                ahead=ahead,
                lateral=lateral,
                closing=velocity.x * forward[0] + velocity.y * forward[1] - ego_along,
                speed=math.hypot(velocity.x, velocity.y),
            )
        )
    return found


def _vehicle_ahead(world: "carla.World") -> Optional[_Around]:
    around = _around(world)
    if around is None:
        return None
    candidates = [
        a for a in around if a.vehicle and _counts_as_vehicle_ahead(a.ahead, a.lateral)
    ]
    return min(candidates, key=lambda a: a.ahead, default=None)


def _crossing_pedestrian(world: "carla.World") -> Optional[_Around]:
    around = _around(world)
    if around is None:
        return None
    candidates = [
        a
        for a in around
        if not a.vehicle
        and 0.0 < a.ahead <= AHEAD_RANGE_M
        and abs(a.lateral) <= PEDESTRIAN_LATERAL_M
        and a.speed > PEDESTRIAN_MOVING_MS
    ]
    return min(candidates, key=lambda a: a.ahead, default=None)


# ---------------------------------------------------------------------------
# The built-in measures
# ---------------------------------------------------------------------------


def _vehicle_ahead_gap_m(world: "carla.World") -> Optional[float]:
    ahead = _vehicle_ahead(world)
    if ahead is None:
        return None
    return ahead.ahead


def _vehicle_ahead_relative_speed_kph(world: "carla.World") -> Optional[float]:
    ahead = _vehicle_ahead(world)
    if ahead is None:
        return None
    return ahead.closing * _KMH_PER_MS


def _crossing_pedestrian_gap_m(world: "carla.World") -> Optional[float]:
    pedestrian = _crossing_pedestrian(world)
    if pedestrian is None:
        return None
    # The gap it set off at, held while it crosses: that is what a scenario
    # stages, not every gap the ego then closes through.
    return _SET_OFF.setdefault(pedestrian.actor_id, pedestrian.ahead)


def _crossing_pedestrian_speed_ms(world: "carla.World") -> Optional[float]:
    pedestrian = _crossing_pedestrian(world)
    if pedestrian is None:
        return None
    return pedestrian.speed


#: The measures every scenario has, by key.
BUILT_IN_MEASURES: dict[str, Measure] = {
    m.key: m
    for m in (
        Measure(
            VEHICLE_AHEAD_GAP_M,
            _vehicle_ahead_gap_m,
            "m",
            "Centre to centre, along the ego's heading, to the nearest vehicle "
            "ahead in its lane or a lane beside it (a vehicle about to cut in "
            "is measured from before it changes lanes)",
        ),
        Measure(
            VEHICLE_AHEAD_RELATIVE_SPEED_KPH,
            _vehicle_ahead_relative_speed_kph,
            "km/h",
            "That vehicle's speed along the ego's heading less the ego's; "
            "negative when the ego closes in",
        ),
        Measure(
            CROSSING_PEDESTRIAN_GAP_M,
            _crossing_pedestrian_gap_m,
            "m",
            "How far ahead the nearest pedestrian ahead that is moving was "
            "when it set off, held while it crosses: one standing at the kerb "
            "is not measured until it sets off",
        ),
        Measure(
            CROSSING_PEDESTRIAN_SPEED_MS,
            _crossing_pedestrian_speed_ms,
            "m/s",
            "That pedestrian's speed",
        ),
    )
}


def _add_driving_measures(measures: dict[str, Measure]) -> None:
    """Add the driving-quality measures to *measures*."""
    for driving in DRIVING_MEASURES:
        measures[driving.key] = Measure(
            driving.key, driving.read, driving.unit, driving.text
        )


_add_driving_measures(BUILT_IN_MEASURES)


# ---------------------------------------------------------------------------
# The running scenario
# ---------------------------------------------------------------------------

_MEASURED: dict[str, Optional["BaseScenario"]] = {"scenario": None}


def set_measured_scenario(scenario: Optional["BaseScenario"]) -> None:
    """Make *scenario*'s measures the ones :func:`read_measure` reads.

    The runner sets the scenario it runs, and clears it when the run ends.
    """
    _MEASURED["scenario"] = scenario
    _CACHE.at, _CACHE.around = None, None
    _SET_OFF.clear()
    reset_driving_measures()


def measured_scenario() -> Optional["BaseScenario"]:
    """The scenario whose measures are read, if one is running."""
    return _MEASURED["scenario"]


def read_measure(key: str, world: "carla.World") -> Optional[float]:
    """The running scenario's measure *key*, or the built-in one outside a run.

    ``None`` when there is nothing to read, the key is not measured, or the
    measure raised.
    """
    scenario = _MEASURED["scenario"]
    if scenario is not None:
        return scenario.measure(key, world)
    if key not in BUILT_IN_MEASURES:
        return None
    measure = BUILT_IN_MEASURES[key]
    try:
        return measure.read(world)
    except Exception:
        return None
