"""Probes: functions that read one ODD attribute from the world.

Every probe takes the ``carla.World`` and returns the attribute's value, or
``None`` when there is nothing to read: no ego yet, a simulator that does not
report the weather, no Lanelet2 map loaded.  ``None`` means the value is
missing.  A probe that raises is treated the same way.

Probes run on every tick, so they share what they read:

* per frame: the actors near the ego, its waypoint and the weather;
* per run: the ego actor, the CARLA map, whether a Lanelet2 map is loaded,
  the lanelet the ego was last on, and the lane counts already worked out.

:func:`reset_probes` forgets both, and the runner calls it at the start of
every run.

The scenery attributes come from two maps:

* CARLA: the waypoint (junction, lanes) and ``get_speed_limit()``;
* Lanelet2: the tags of the lanelet the ego is on (``location``, ``subtype``,
  ``speed_limit``), for a run with a Lanelet2 map loaded.

Weather values are CARLA's 0-100 intensities, not physical units such as mm/h,
so rain and fog are named levels.

The probes whose value the Lanelet2 map alone decides also read it off a
lanelet, without a world: ``probe.on_lanelet(lanelet, lanelet_map,
routing_graph)``.  That is what checks a scenario's planned route against an
ODD before it runs (:mod:`~autoware_carla_scenario.odd.route`).  It returns
the value, ``None`` where the run would read none, or
:data:`~autoware_carla_scenario.odd.model.UNDECIDED` where only the run knows
it.  The others (speed, weather, traffic) have no ``on_lanelet``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..conditions.base import find_actor_in_list
from ..constants import EGO_ROLE_NAME
from ..kinematics import Vector3
from .model import UNDECIDED

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "ILLUMINATION_LEVELS",
    "INTENSITY_LEVELS",
    "NEARBY_RADIUS_M",
    "TRAFFIC_DENSITY_LEVELS",
    "ego_speed_kph",
    "fog",
    "illumination",
    "illumination_level",
    "in_junction",
    "intensity_level",
    "lane_count",
    "lanelet_location",
    "lanelet_speed_limit_kph",
    "lanelet_subtype",
    "pedestrian_nearby",
    "rain",
    "reset_probes",
    "speed_limit_kph",
    "traffic_density",
    "traffic_density_level",
]

#: Radius around the ego in which other road users count as near it, in metres.
NEARBY_RADIUS_M = 50.0

#: Illumination levels, brightest first.
ILLUMINATION_LEVELS = ["day", "low_sun", "twilight", "night"]

#: Rain and fog levels, from CARLA's 0-100 intensity.
INTENSITY_LEVELS = ["none", "light", "moderate", "heavy"]

#: Traffic density levels, from the number of other vehicles near the ego.
TRAFFIC_DENSITY_LEVELS = ["none", "low", "medium", "high"]

_KMH_PER_MS = 3.6


def illumination_level(sun_altitude_deg: float) -> str:
    """The illumination level for a sun at *sun_altitude_deg* above the horizon.

    Day from 15 degrees up; a low sun (glare, long shadows) below that; civil
    twilight down to 6 degrees below the horizon; night under it.
    """
    if sun_altitude_deg >= 15.0:
        return "day"
    if sun_altitude_deg >= 0.0:
        return "low_sun"
    if sun_altitude_deg >= -6.0:
        return "twilight"
    return "night"


def intensity_level(intensity: float) -> str:
    """The level of a CARLA 0-100 weather intensity (rain, fog)."""
    if intensity < 1.0:
        return "none"
    if intensity < 30.0:
        return "light"
    if intensity < 70.0:
        return "moderate"
    return "heavy"


def traffic_density_level(vehicles_nearby: int) -> str:
    """The traffic density level for *vehicles_nearby* other vehicles."""
    if vehicles_nearby == 0:
        return "none"
    if vehicles_nearby <= 2:
        return "low"
    if vehicles_nearby <= 5:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# What the probes share
# ---------------------------------------------------------------------------

_UNREAD = object()


@dataclass
class _Run:
    """What stays true for a whole run."""

    ego: Any = None
    carla_map: Any = None
    #: ``None`` until asked; then whether a Lanelet2 map is loaded.
    lanelet2: Optional[bool] = None
    lanelet: Any = None
    tags: dict[int, dict[str, str]] = field(default_factory=dict)
    lane_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    frame: Optional[int] = None
    #: What was read this frame, by key.
    cache: dict[str, Any] = field(default_factory=dict)


_RUN = _Run()


def reset_probes() -> None:
    """Forget what the probes have read; the runner calls this for every run."""
    global _RUN
    _RUN = _Run()


def _frame(world: "carla.World") -> dict[str, Any]:
    """This frame's cache: emptied when the world has moved on."""
    try:
        frame: Optional[int] = int(world.get_snapshot().frame)
    except Exception:
        frame = None
    if frame is None or frame != _RUN.frame:
        _RUN.frame = frame
        _RUN.cache = {}
    return _RUN.cache


def _read(cache: dict[str, Any], key: str, read: Callable[[], Any]) -> Any:
    value = cache.get(key, _UNREAD)
    if value is _UNREAD:
        try:
            value = read()
        except Exception:
            value = None
        cache[key] = value
    return value


def _ego(world: "carla.World") -> Any:
    """The ego actor, found once per run (again if it is gone)."""
    ego = _RUN.ego
    if ego is None or not getattr(ego, "is_alive", True):
        try:
            _RUN.ego = find_actor_in_list(world.get_actors(), EGO_ROLE_NAME)
        except Exception:
            _RUN.ego = None
    return _RUN.ego


def _waypoint(world: "carla.World", cache: dict[str, Any]) -> Any:
    def read() -> Any:
        ego = _ego(world)
        if ego is None:
            return None
        if _RUN.carla_map is None:
            from ..coordinate.map_manager import MapManager  # noqa: PLC0415

            _RUN.carla_map = MapManager.get_instance().carla_map or world.get_map()
        return _RUN.carla_map.get_waypoint(ego.get_location())

    return _read(cache, "waypoint", read)


def _weather(world: "carla.World", cache: dict[str, Any]) -> Any:
    return _read(cache, "weather", world.get_weather)


def _lanelet_tags(world: "carla.World") -> Optional[dict[str, str]]:
    """The tags of the lanelet the ego is on, or ``None`` without a Lanelet2 map."""
    if _RUN.lanelet2 is False:
        return None
    cache = _frame(world)

    def read() -> Optional[dict[str, str]]:
        import lanelet2.core  # noqa: PLC0415
        import lanelet2.geometry  # noqa: PLC0415

        from ..coordinate import CarlaWorldPose, to_lanelet2  # noqa: PLC0415
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.transform import _carla_to_lanelet2_frame  # noqa: PLC0415

        manager = MapManager.get_instance()
        if _RUN.lanelet2 is None:
            try:
                manager.lanelet_map
            except RuntimeError:
                _RUN.lanelet2 = False
                return None
            _RUN.lanelet2 = True
        ego = _ego(world)
        if ego is None:
            return None
        pose = CarlaWorldPose.from_carla_transform(ego.get_transform())
        x, y, _ = _carla_to_lanelet2_frame(pose)
        here = lanelet2.core.BasicPoint2d(x, y)
        # Still on the lanelet it was on: no map-matching needed.
        if _RUN.lanelet is None or not lanelet2.geometry.inside(_RUN.lanelet, here):
            lanelet_id = to_lanelet2(pose).lanelet_id
            _RUN.lanelet = manager.lanelet_map.laneletLayer[lanelet_id]
        lanelet = _RUN.lanelet
        if lanelet.id not in _RUN.tags:
            _RUN.tags[lanelet.id] = {
                str(k): str(v) for k, v in lanelet.attributes.items()
            }
        return _RUN.tags[lanelet.id]

    return _read(cache, "lanelet_tags", read)


def _nearby(world: "carla.World") -> Optional[tuple[int, int]]:
    """(other vehicles, pedestrians) within :data:`NEARBY_RADIUS_M` of the ego."""

    def read() -> Optional[tuple[int, int]]:
        ego = _ego(world)
        if ego is None:
            return None
        here = ego.get_location()
        limit = NEARBY_RADIUS_M**2
        vehicles = walkers = 0
        for actor in world.get_actors():
            kind = actor.type_id
            if actor.id == ego.id or not kind.startswith(("vehicle.", "walker.")):
                continue
            at = actor.get_location()
            if (at.x - here.x) ** 2 + (at.y - here.y) ** 2 + (
                at.z - here.z
            ) ** 2 <= limit:
                if kind.startswith("vehicle."):
                    vehicles += 1
                else:
                    walkers += 1
        return vehicles, walkers

    return _read(_frame(world), "nearby", read)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def ego_speed_kph(world: "carla.World") -> Optional[float]:
    """The ego's speed, in km/h."""
    ego = _ego(world)
    if ego is None:
        return None
    return Vector3.from_carla_vector3d(ego.get_velocity()).magnitude() * _KMH_PER_MS


def in_junction(world: "carla.World") -> Optional[bool]:
    """Whether the ego is inside a junction (CARLA waypoint)."""
    waypoint = _waypoint(world, _frame(world))
    return None if waypoint is None else bool(waypoint.is_junction)


def lanelet_speed_limit_kph(world: "carla.World") -> Optional[float]:
    """The Lanelet2 ``speed_limit`` tag of the ego's lanelet, in km/h."""
    tags = _lanelet_tags(world)
    if tags is None or "speed_limit" not in tags:
        return None
    try:
        return float(tags["speed_limit"])
    except ValueError:
        return None


def speed_limit_kph(world: "carla.World") -> Optional[float]:
    """The speed limit for the ego's lane, in km/h.

    The Lanelet2 ``speed_limit`` tag of the ego's lanelet when there is one,
    otherwise CARLA's ``get_speed_limit()``.
    """
    limit = lanelet_speed_limit_kph(world)
    if limit is not None:
        return limit
    ego = _ego(world)
    if ego is None:
        return None
    # The speed limit is a vehicle's: a plain carla.Actor does not have it.
    vehicle = ego.as_vehicle() if hasattr(ego, "as_vehicle") else ego
    limit = float(vehicle.get_speed_limit() or 0.0)
    return limit if limit > 0.0 else None


def lane_count(world: "carla.World") -> Optional[int]:
    """Driving lanes in the ego's direction of travel, its own included.

    Read from the CARLA waypoint, and only outside junctions.
    """
    waypoint = _waypoint(world, _frame(world))
    if waypoint is None or waypoint.is_junction:
        return None
    key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id)
    if key not in _RUN.lane_counts:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        # typesafe_carla's lane type is an int (an IntEnum on the API side).
        driving = int(carla.LaneType.Driving)
        count = 1
        for step in ("get_left_lane", "get_right_lane"):
            lane = getattr(waypoint, step)()
            seen = 0
            while (
                lane is not None
                and seen < 16
                and lane.lane_id * waypoint.lane_id > 0
                and int(lane.lane_type) == driving
            ):
                count += 1
                seen += 1
                lane = getattr(lane, step)()
        _RUN.lane_counts[key] = count
    return _RUN.lane_counts[key]


def lanelet_location(world: "carla.World") -> Optional[str]:
    """The Lanelet2 ``location`` tag of the ego's lanelet (urban, nonurban, ...)."""
    tags = _lanelet_tags(world)
    return None if tags is None else tags.get("location")


def lanelet_subtype(world: "carla.World") -> Optional[str]:
    """The Lanelet2 ``subtype`` tag of the ego's lanelet (road, highway, ...)."""
    tags = _lanelet_tags(world)
    return None if tags is None else tags.get("subtype")


def illumination(world: "carla.World") -> Optional[str]:
    """The illumination level, from the sun's altitude (:func:`illumination_level`)."""
    weather = _weather(world, _frame(world))
    if weather is None:
        return None
    return illumination_level(float(weather.sun_altitude_angle))


def rain(world: "carla.World") -> Optional[str]:
    """The rain level, from CARLA's 0-100 precipitation."""
    weather = _weather(world, _frame(world))
    return None if weather is None else intensity_level(float(weather.precipitation))


def fog(world: "carla.World") -> Optional[str]:
    """The fog level, from CARLA's 0-100 fog density."""
    weather = _weather(world, _frame(world))
    return None if weather is None else intensity_level(float(weather.fog_density))


def traffic_density(world: "carla.World") -> Optional[str]:
    """The traffic density level, from other vehicles within :data:`NEARBY_RADIUS_M`."""
    nearby = _nearby(world)
    return None if nearby is None else traffic_density_level(nearby[0])


def pedestrian_nearby(world: "carla.World") -> Optional[bool]:
    """Whether a pedestrian is within :data:`NEARBY_RADIUS_M` of the ego."""
    nearby = _nearby(world)
    return None if nearby is None else nearby[1] > 0


# ---------------------------------------------------------------------------
# The same probes, read off a lanelet of the map (a planned route)
# ---------------------------------------------------------------------------


def _on_lanelet(probe: Callable[..., Any]) -> Callable[[Callable[..., Any]], Any]:
    """Make the decorated function *probe*'s ``on_lanelet``."""

    def attach(read: Callable[..., Any]) -> Any:
        setattr(probe, "on_lanelet", read)  # noqa: B010 - a function attribute
        return read

    return attach


def _tag(lanelet: Any, key: str) -> Optional[str]:
    attributes = lanelet.attributes
    return str(attributes[key]) if key in attributes else None


def _is_junction_lanelet(lanelet: Any) -> bool:
    """Whether *lanelet* lies in a junction: it has a ``turn_direction`` tag.

    Autoware's Lanelet2 maps tag every lanelet inside an intersection with
    ``turn_direction`` (``straight``, ``left`` or ``right``), and only those;
    the sweeper's ``is_junction`` constraint reads it the same way.
    """
    return "turn_direction" in lanelet.attributes


@_on_lanelet(in_junction)
def _in_junction_on_lanelet(
    lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> Optional[bool]:
    return _is_junction_lanelet(lanelet)


@_on_lanelet(lanelet_speed_limit_kph)
def _lanelet_speed_limit_on_lanelet(
    lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> Optional[float]:
    tag = _tag(lanelet, "speed_limit")
    if tag is None:
        return None
    try:
        return float(tag)
    except ValueError:
        return None


@_on_lanelet(speed_limit_kph)
def _speed_limit_on_lanelet(lanelet: Any, lanelet_map: Any, routing_graph: Any) -> Any:
    # Without the tag, the run reads CARLA's limit, which the map cannot tell.
    tag = _tag(lanelet, "speed_limit")
    if tag is None:
        return UNDECIDED
    return _lanelet_speed_limit_on_lanelet(lanelet, lanelet_map, routing_graph)


@_on_lanelet(lane_count)
def _lane_count_on_lanelet(
    lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> Optional[int]:
    # As the run reads it: nothing inside a junction.  The lanelets beside it
    # in the routing graph run in its direction, and include it.
    if _is_junction_lanelet(lanelet):
        return None
    return max(1, len(routing_graph.besides(lanelet)))


@_on_lanelet(lanelet_location)
def _location_on_lanelet(
    lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> Optional[str]:
    return _tag(lanelet, "location")


@_on_lanelet(lanelet_subtype)
def _subtype_on_lanelet(
    lanelet: Any, lanelet_map: Any, routing_graph: Any
) -> Optional[str]:
    return _tag(lanelet, "subtype")
