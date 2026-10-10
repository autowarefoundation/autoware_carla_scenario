"""Scenario measures: what a run sets up, read off the world under fixed keys.

A scenario stages road users -- a vehicle ahead in the next lane, a pedestrian
running out -- and its parameters say how: how far ahead, how fast.  A
**measure** reads such a quantity off the world while the run drives, under a
key the framework fixes (:data:`VEHICLE_AHEAD_GAP_M`, ...), so every scenario
speaks the same vocabulary of what it sets up, whoever wrote it.

Every :class:`~autoware_carla_scenario.BaseScenario` has the built-in measures
(:data:`BUILT_IN_MEASURES`), which read the world generically, in the ego's
frame.  A scenario that knows better -- which actor *is* its cut-in vehicle --
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
from typing import TYPE_CHECKING, Any, Callable, Optional

from .conditions.base import find_actor_in_list
from .constants import EGO_ROLE_NAME

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
    read: Callable[["carla.World"], Any]
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


_CACHE: dict[str, Any] = {"frame": None, "around": None}


def _frame_of(world: "carla.World") -> Optional[int]:
    try:
        return int(world.get_snapshot().frame)
    except Exception:
        return None


def in_ego_frame(world: "carla.World", actor: Any, ego: Any) -> tuple[float, float]:
    """(metres ahead of *ego*, metres to its side) of *actor*.

    Along the ego's heading and across it; the side's sign is CARLA's (right
    positive), which no measure here depends on.
    """
    del world
    transform = ego.get_transform()
    here = transform.location
    yaw = math.radians(transform.rotation.yaw)
    at = actor.get_location()
    dx, dy = at.x - here.x, at.y - here.y
    return (
        dx * math.cos(yaw) + dy * math.sin(yaw),
        -dx * math.sin(yaw) + dy * math.cos(yaw),
    )


def _around(world: "carla.World") -> Optional[list[_Around]]:
    """Every other vehicle and pedestrian, in the ego's frame; once per frame."""
    frame = _frame_of(world)
    if frame is not None and frame == _CACHE["frame"]:
        return _CACHE["around"]
    found: Optional[list[_Around]]
    try:
        actors = list(world.get_actors())
        ego = find_actor_in_list(actors, EGO_ROLE_NAME)
        found = None if ego is None else _relative_to(ego, actors)
    except Exception:
        found = None
    _CACHE["frame"], _CACHE["around"] = frame, found
    return found


def _relative_to(ego: Any, actors: list[Any]) -> list[_Around]:
    transform = ego.get_transform()
    here = transform.location
    yaw = math.radians(transform.rotation.yaw)
    forward = (math.cos(yaw), math.sin(yaw))
    ego_velocity = ego.get_velocity()
    ego_along = ego_velocity.x * forward[0] + ego_velocity.y * forward[1]
    found: list[_Around] = []
    for actor in actors:
        kind = actor.type_id
        if actor.id == ego.id or not kind.startswith(("vehicle.", "walker.")):
            continue
        at = actor.get_location()
        dx, dy = at.x - here.x, at.y - here.y
        velocity = actor.get_velocity()
        found.append(
            _Around(
                actor_id=actor.id,
                vehicle=kind.startswith("vehicle."),
                ahead=dx * forward[0] + dy * forward[1],
                lateral=-dx * forward[1] + dy * forward[0],
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
        a
        for a in around
        if a.vehicle
        and 0.0 < a.ahead <= AHEAD_RANGE_M
        and abs(a.lateral) <= 3 * LANE_HALF_WIDTH_M
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
    return None if ahead is None else ahead.ahead


def _vehicle_ahead_relative_speed_kph(world: "carla.World") -> Optional[float]:
    ahead = _vehicle_ahead(world)
    return None if ahead is None else ahead.closing * _KMH_PER_MS


def _crossing_pedestrian_gap_m(world: "carla.World") -> Optional[float]:
    pedestrian = _crossing_pedestrian(world)
    return None if pedestrian is None else pedestrian.ahead


def _crossing_pedestrian_speed_ms(world: "carla.World") -> Optional[float]:
    pedestrian = _crossing_pedestrian(world)
    return None if pedestrian is None else pedestrian.speed


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
            "To the nearest pedestrian ahead that is moving: one standing at "
            "the kerb is not measured until it sets off, so the first value is "
            "how far ahead it was when it did",
        ),
        Measure(
            CROSSING_PEDESTRIAN_SPEED_MS,
            _crossing_pedestrian_speed_ms,
            "m/s",
            "That pedestrian's speed",
        ),
    )
}


# ---------------------------------------------------------------------------
# The running scenario
# ---------------------------------------------------------------------------

_MEASURED: dict[str, Any] = {"scenario": None}


def set_measured_scenario(scenario: Optional["BaseScenario"]) -> None:
    """Make *scenario*'s measures the ones :func:`read_measure` reads.

    The runner sets the scenario it runs, and clears it when the run ends.
    """
    _MEASURED["scenario"] = scenario
    _CACHE["frame"], _CACHE["around"] = None, None


def measured_scenario() -> Optional["BaseScenario"]:
    """The scenario whose measures are read, if one is running."""
    return _MEASURED["scenario"]


def read_measure(key: str, world: "carla.World") -> Any:
    """The running scenario's measure *key*, or the built-in one outside a run.

    ``None`` when there is nothing to read, the key is not measured, or the
    measure raised.
    """
    scenario = _MEASURED["scenario"]
    if scenario is not None:
        return scenario.measure(key, world)
    measure = BUILT_IN_MEASURES.get(key)
    if measure is None:
        return None
    try:
        return measure.read(world)
    except Exception:
        return None
