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
from typing import TYPE_CHECKING, Optional, Union

from ..entity_role import find_actor_in_world
from ..constants import EGO_ROLE_NAME
from ..kinematics import Vector3
from .model import UNDECIDED

if TYPE_CHECKING:
    import typesafe_carla.carla as carla
    from lanelet2.core import BasicPoint2d, ConstLanelet, Lanelet, LaneletMap
    from lanelet2.routing import RoutingGraph

    from .model import Undecided

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
    "lanelet_in_junction",
    "lanelet_location",
    "lanelet_road_type",
    "lanelet_speed_limit",
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


@dataclass
class _Frame:
    """What the probes read in one frame.

    A read's name is in :attr:`read` once it has been read, whatever it read:
    a value, or ``None`` (nothing to read, or a read that raised).
    """

    read: set[str] = field(default_factory=set)
    waypoint: Optional[carla.Waypoint] = None
    geolocation: Optional[tuple[float, float]] = None
    weather: Optional[carla.WeatherParameters] = None
    lanelet_tags: Optional[dict[str, str]] = None
    nearby: Optional[tuple[int, int]] = None


@dataclass
class _Run:
    """What stays true for a whole run."""

    ego: Optional[carla.Actor] = None
    carla_map: Optional[carla.Map] = None
    #: ``None`` until asked; then whether a Lanelet2 map is loaded.
    lanelet2: Optional[bool] = None
    lanelet: Optional[Lanelet] = None
    tags: dict[int, dict[str, str]] = field(default_factory=dict)
    lane_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    frame: Optional[int] = None
    #: What was read this frame.
    cache: _Frame = field(default_factory=_Frame)


_RUN = _Run()


def reset_probes() -> None:
    """Forget what the probes have read; the runner calls this for every run."""
    global _RUN
    _RUN = _Run()


def _frame(world: carla.World) -> _Frame:
    """This frame's cache: emptied when the world has moved on."""
    try:
        frame: Optional[int] = int(world.get_snapshot().frame)
    except Exception:
        frame = None
    if frame is None or frame != _RUN.frame:
        _RUN.frame = frame
        _RUN.cache = _Frame()
    return _RUN.cache


def _alive(actor: carla.Actor) -> bool:
    """Whether *actor* is still in the world (one that cannot tell is)."""
    if TYPE_CHECKING:
        return actor.is_alive
    else:
        return bool(getattr(actor, "is_alive", True))


def _ego(world: carla.World) -> Optional[carla.Actor]:
    """The ego actor, found once per run (again if it is gone)."""
    ego = _RUN.ego
    if ego is None or not _alive(ego):
        try:
            _RUN.ego = find_actor_in_world(world, EGO_ROLE_NAME)
        except Exception:
            _RUN.ego = None
    return _RUN.ego


def _carla_map(world: carla.World) -> carla.Map:
    """The run's CARLA map, fetched once."""
    if _RUN.carla_map is None:
        from ..coordinate.map_manager import run_carla_map  # noqa: PLC0415

        _RUN.carla_map = run_carla_map(world)
    carla_map: carla.Map = _RUN.carla_map
    return carla_map


def _read_waypoint(world: carla.World) -> Optional[carla.Waypoint]:
    ego = _ego(world)
    if ego is None:
        return None
    return _carla_map(world).get_waypoint(ego.get_location())


def _waypoint(world: carla.World) -> Optional[carla.Waypoint]:
    """The ego's CARLA waypoint, read once per frame."""
    cache = _frame(world)
    if "waypoint" not in cache.read:
        try:
            cache.waypoint = _read_waypoint(world)
        except Exception:
            cache.waypoint = None
        cache.read.add("waypoint")
    return cache.waypoint


def _read_geolocation(world: carla.World) -> Optional[tuple[float, float]]:
    ego = _ego(world)
    if ego is None:
        return None
    geo = _carla_map(world).transform_to_geolocation(ego.get_location())
    return (float(geo.latitude), float(geo.longitude))


def ego_geolocation(world: carla.World) -> Optional[tuple[float, float]]:
    """The ego's ``(latitude, longitude)``, from the map's OpenDRIVE geoReference."""
    cache = _frame(world)
    if "geolocation" not in cache.read:
        try:
            cache.geolocation = _read_geolocation(world)
        except Exception:
            cache.geolocation = None
        cache.read.add("geolocation")
    return cache.geolocation


def _weather(world: carla.World) -> Optional[carla.WeatherParameters]:
    """The world's weather, read once per frame."""
    cache = _frame(world)
    if "weather" not in cache.read:
        try:
            cache.weather = world.get_weather()
        except Exception:
            cache.weather = None
        cache.read.add("weather")
    return cache.weather


def _still_on(lanelet: Lanelet, here: BasicPoint2d) -> bool:
    """Whether *here* lies on *lanelet*."""
    import lanelet2.geometry  # noqa: PLC0415

    return bool(lanelet2.geometry.inside(lanelet, here))


def _read_lanelet_tags(world: carla.World) -> Optional[dict[str, str]]:
    import lanelet2.core  # noqa: PLC0415

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
    if _RUN.lanelet is None or not _still_on(_RUN.lanelet, here):
        lanelet_id = to_lanelet2(pose).lanelet_id
        _RUN.lanelet = manager.lanelet_map.laneletLayer[lanelet_id]
    lanelet: Lanelet = _RUN.lanelet
    if lanelet.id not in _RUN.tags:
        _RUN.tags[lanelet.id] = {str(k): str(v) for k, v in lanelet.attributes.items()}
    return _RUN.tags[lanelet.id]


def _lanelet_tags(world: carla.World) -> Optional[dict[str, str]]:
    """The tags of the lanelet the ego is on, or ``None`` without a Lanelet2 map."""
    if _RUN.lanelet2 is False:
        return None
    cache = _frame(world)
    if "lanelet_tags" not in cache.read:
        try:
            cache.lanelet_tags = _read_lanelet_tags(world)
        except Exception:
            cache.lanelet_tags = None
        cache.read.add("lanelet_tags")
    return cache.lanelet_tags


def _read_nearby(world: carla.World) -> Optional[tuple[int, int]]:
    ego = _ego(world)
    if ego is None:
        return None
    here = ego.get_location()
    limit = NEARBY_RADIUS_M**2
    vehicles = walkers = 0
    for actor in world.get_actors():
        kind = actor.type_id
        if actor.id == ego.id or not (
            kind.startswith("vehicle.") or kind.startswith("walker.")
        ):
            continue
        at = actor.get_location()
        if (at.x - here.x) ** 2 + (at.y - here.y) ** 2 + (at.z - here.z) ** 2 <= limit:
            if kind.startswith("vehicle."):
                vehicles += 1
            else:
                walkers += 1
    return vehicles, walkers


def _nearby(world: carla.World) -> Optional[tuple[int, int]]:
    """(other vehicles, pedestrians) within :data:`NEARBY_RADIUS_M` of the ego."""
    cache = _frame(world)
    if "nearby" not in cache.read:
        try:
            cache.nearby = _read_nearby(world)
        except Exception:
            cache.nearby = None
        cache.read.add("nearby")
    return cache.nearby


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def ego_position(world: carla.World) -> Optional[tuple[float, float, float]]:
    """Where the ego is, in CARLA world coordinates (m); used to measure distance."""
    ego = _ego(world)
    if ego is None:
        return None
    location = ego.get_location()
    return (float(location.x), float(location.y), float(location.z))


def ego_speed_kph(world: carla.World) -> Optional[float]:
    """The ego's speed, in km/h."""
    ego = _ego(world)
    if ego is None:
        return None
    return Vector3.from_carla_vector3d(ego.get_velocity()).magnitude() * _KMH_PER_MS


def in_junction(world: carla.World) -> Optional[bool]:
    """Whether the ego is inside a junction (CARLA waypoint)."""
    waypoint = _waypoint(world)
    if waypoint is None:
        return None
    return bool(waypoint.is_junction)


def lanelet_speed_limit_kph(world: carla.World) -> Optional[float]:
    """The Lanelet2 ``speed_limit`` tag of the ego's lanelet, in km/h."""
    tags = _lanelet_tags(world)
    if tags is None or "speed_limit" not in tags:
        return None
    try:
        return float(tags["speed_limit"])
    except ValueError:
        return None


def speed_limit_kph(world: carla.World) -> Optional[float]:
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
    if TYPE_CHECKING:
        vehicle = ego.as_vehicle()
    else:
        vehicle = ego.as_vehicle() if hasattr(ego, "as_vehicle") else ego
    carla_limit = float(vehicle.get_speed_limit() or 0.0)
    if carla_limit > 0.0:
        return carla_limit
    return None


def _beside(waypoint: carla.Waypoint, left: bool) -> Optional[carla.Waypoint]:
    """The waypoint of the lane to the left (or right) of *waypoint*'s."""
    if left:
        return waypoint.get_left_lane()
    return waypoint.get_right_lane()


def lane_count(world: carla.World) -> Optional[int]:
    """Driving lanes in the ego's direction of travel, its own included.

    Read from the CARLA waypoint, and only outside junctions.
    """
    waypoint = _waypoint(world)
    if waypoint is None or waypoint.is_junction:
        return None
    key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id)
    if key not in _RUN.lane_counts:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        # typesafe_carla's lane type is an int (an IntEnum on the API side).
        driving = int(carla.LaneType.Driving)
        count = 1
        for left in (True, False):
            lane = _beside(waypoint, left)
            seen = 0
            while (
                lane is not None
                and seen < 16
                and lane.lane_id * waypoint.lane_id > 0
                and int(lane.lane_type) == driving
            ):
                count += 1
                seen += 1
                lane = _beside(lane, left)
        _RUN.lane_counts[key] = count
    return _RUN.lane_counts[key]


def lanelet_location(world: carla.World) -> Optional[str]:
    """The Lanelet2 ``location`` tag of the ego's lanelet (urban, nonurban, ...)."""
    tags = _lanelet_tags(world)
    if tags is None:
        return None
    return tags.get("location")


def lanelet_subtype(world: carla.World) -> Optional[str]:
    """The Lanelet2 ``subtype`` tag of the ego's lanelet (road, highway, ...)."""
    tags = _lanelet_tags(world)
    if tags is None:
        return None
    return tags.get("subtype")


def illumination(world: carla.World) -> Optional[str]:
    """The illumination level, from the sun's altitude (:func:`illumination_level`)."""
    weather = _weather(world)
    if weather is None:
        return None
    return illumination_level(float(weather.sun_altitude_angle))


def rain(world: carla.World) -> Optional[str]:
    """The rain level, from CARLA's 0-100 precipitation."""
    weather = _weather(world)
    if weather is None:
        return None
    return intensity_level(float(weather.precipitation))


def fog(world: carla.World) -> Optional[str]:
    """The fog level, from CARLA's 0-100 fog density."""
    weather = _weather(world)
    if weather is None:
        return None
    return intensity_level(float(weather.fog_density))


def traffic_density(world: carla.World) -> Optional[str]:
    """The traffic density level, from other vehicles within :data:`NEARBY_RADIUS_M`."""
    nearby = _nearby(world)
    if nearby is None:
        return None
    return traffic_density_level(nearby[0])


def pedestrian_nearby(world: carla.World) -> Optional[bool]:
    """Whether a pedestrian is within :data:`NEARBY_RADIUS_M` of the ego."""
    nearby = _nearby(world)
    if nearby is None:
        return None
    return nearby[1] > 0


# ---------------------------------------------------------------------------
# The same probes, read off a lanelet of the map (a planned route)
# ---------------------------------------------------------------------------


#: What :func:`speed_limit_kph` reads off a lanelet: the tag's number, or
#: :data:`~autoware_carla_scenario.odd.model.UNDECIDED`.  An alias, so the
#: check keeps the union (it drops one written out in an annotation) and
#: Codon converts each return to it.
_SpeedLimitOnLanelet = Union[float, "Undecided"]


def _tag(lanelet: ConstLanelet, key: str) -> Optional[str]:
    attributes = lanelet.attributes
    if key in attributes:
        return str(attributes[key])
    return None


def lanelet_in_junction(lanelet: ConstLanelet) -> bool:
    """Whether *lanelet* lies in a junction: it has a ``turn_direction`` tag.

    What ``in_junction`` reads off a lanelet (``scenery.junction`` of the
    default ODD); the adversities' location gate reads the same.

    Autoware's Lanelet2 maps tag every lanelet inside an intersection with
    ``turn_direction`` (``straight``, ``left`` or ``right``), and only those;
    the sweeper's ``is_junction`` constraint reads it the same way.
    """
    return "turn_direction" in lanelet.attributes


def _in_junction_on_lanelet(
    lanelet: ConstLanelet, lanelet_map: LaneletMap, routing_graph: RoutingGraph
) -> Optional[bool]:
    return lanelet_in_junction(lanelet)


def lanelet_speed_limit(lanelet: ConstLanelet) -> Optional[float]:
    """*lanelet*'s ``speed_limit`` tag in km/h, or None untagged or not a number.

    What ``lanelet_speed_limit_kph`` reads off a lanelet.
    """
    tag = _tag(lanelet, "speed_limit")
    if tag is None:
        return None
    try:
        return float(tag)
    except ValueError:
        return None


def lanelet_road_type(lanelet: ConstLanelet) -> Optional[str]:
    """*lanelet*'s ``subtype`` tag (road, highway, ...), or None.

    What ``lanelet_subtype`` reads off a lanelet (``scenery.road_type`` of
    the default ODD).
    """
    return _tag(lanelet, "subtype")


def _lanelet_speed_limit_on_lanelet(
    lanelet: ConstLanelet, lanelet_map: LaneletMap, routing_graph: RoutingGraph
) -> Optional[float]:
    return lanelet_speed_limit(lanelet)


def _speed_limit_on_lanelet(
    lanelet: ConstLanelet, lanelet_map: LaneletMap, routing_graph: RoutingGraph
) -> _SpeedLimitOnLanelet:
    # Without a tag it can read as a number, the run reads CARLA's limit,
    # which the map cannot tell.
    limit = _lanelet_speed_limit_on_lanelet(lanelet, lanelet_map, routing_graph)
    if limit is None:
        return UNDECIDED
    return float(limit)


def _neighbour(
    routing_graph: RoutingGraph, lanelet: ConstLanelet, left: bool
) -> Optional[ConstLanelet]:
    """The lanelet beside *lanelet*, to its left (or right), in its direction.

    The routing graph's left()/right() are the lane-changeable neighbours,
    adjacentLeft()/adjacentRight() the ones behind a solid line; both run in
    the lanelet's direction (``besides()`` would stop at a solid line).
    """
    there = routing_graph.left(lanelet) if left else routing_graph.right(lanelet)
    if there is None:
        if left:
            there = routing_graph.adjacentLeft(lanelet)
        else:
            there = routing_graph.adjacentRight(lanelet)
    return there


def _lane_count_on_lanelet(
    lanelet: ConstLanelet, lanelet_map: LaneletMap, routing_graph: RoutingGraph
) -> Optional[int]:
    # As the run reads it: nothing inside a junction, and every lane in the
    # same direction, whether or not a lane change into it is allowed.
    if lanelet_in_junction(lanelet):
        return None
    seen = {lanelet.id}
    for left in (True, False):
        here = lanelet
        while len(seen) < 64:  # a guard against a malformed map's loops
            there = _neighbour(routing_graph, here, left)
            if there is None or there.id in seen:
                break
            seen.add(there.id)
            here = there
    return len(seen)


def _location_on_lanelet(
    lanelet: ConstLanelet, lanelet_map: LaneletMap, routing_graph: RoutingGraph
) -> Optional[str]:
    return _tag(lanelet, "location")


def _subtype_on_lanelet(
    lanelet: ConstLanelet, lanelet_map: LaneletMap, routing_graph: RoutingGraph
) -> Optional[str]:
    return lanelet_road_type(lanelet)


# Each probe the map decides has the function reading it off a lanelet as its
# ``on_lanelet`` attribute, which odd.route looks for.  A function attribute
# is Python's alone (Codon has none), so the check compiles each reader on its
# own, as it compiles every function of the module.
if not TYPE_CHECKING:
    in_junction.on_lanelet = _in_junction_on_lanelet
    lanelet_speed_limit_kph.on_lanelet = _lanelet_speed_limit_on_lanelet
    speed_limit_kph.on_lanelet = _speed_limit_on_lanelet
    lane_count.on_lanelet = _lane_count_on_lanelet
    lanelet_location.on_lanelet = _location_on_lanelet
    lanelet_subtype.on_lanelet = _subtype_on_lanelet
