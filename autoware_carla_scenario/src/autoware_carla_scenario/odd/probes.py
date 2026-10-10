"""Probes: functions that read one ODD attribute from the world.

Every probe takes the ``carla.World`` and returns the attribute's value, or
``None`` when there is nothing to read: no ego yet, a simulator that does not
report the weather, no Lanelet2 map loaded.  ``None`` is *unknown*.  A probe
that raises is treated the same way by the collector.

Probes are sampled on every tick, so what several of them read is looked up
once per tick: the ego, its CARLA waypoint, the actor list, the weather and
the ego's lanelet.  The CARLA map, which is slow to fetch, is read once per
run.  :func:`reset_probes` forgets all of that, and the runner calls it at the
start of every run.

The scenery attributes come from two maps:

* CARLA: the waypoint (junction, lanes) and ``get_speed_limit()``;
* Lanelet2: the tags of the lanelet the ego is on (``location``, ``subtype``,
  ``speed_limit``), for a run with a Lanelet2 map loaded.

Weather values are CARLA's 0-100 intensities, not physical units such as mm/h,
so rain and fog are named levels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..constants import EGO_ROLE_NAME

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
# What a tick's probes share
# ---------------------------------------------------------------------------


class _State:
    def __init__(self) -> None:
        self.carla_map: Any = None
        self.frame: Optional[int] = None
        self.cache: dict[str, Any] = {}


_STATE = _State()


def reset_probes() -> None:
    """Forget what the probes have read; the runner calls this for every run."""
    global _STATE
    _STATE = _State()


def _cached(world: "carla.World", key: str, read: Callable[[], Any]) -> Any:
    try:
        frame: Optional[int] = int(world.get_snapshot().frame)
    except Exception:
        frame = None
    if frame is None or frame != _STATE.frame:
        _STATE.frame = frame
        _STATE.cache = {}
    if key not in _STATE.cache:
        try:
            _STATE.cache[key] = read()
        except Exception:
            _STATE.cache[key] = None
    return _STATE.cache[key]


def _actors(world: "carla.World") -> list[Any]:
    return _cached(world, "actors", lambda: list(world.get_actors())) or []


def _ego(world: "carla.World") -> Any:
    name = str(EGO_ROLE_NAME)

    def read() -> Any:
        return next(
            (a for a in _actors(world) if a.attributes.get("role_name") == name),
            None,
        )

    return _cached(world, "ego", read)


def _waypoint(world: "carla.World") -> Any:
    def read() -> Any:
        ego = _ego(world)
        if ego is None:
            return None
        if _STATE.carla_map is None:
            _STATE.carla_map = world.get_map()
        return _STATE.carla_map.get_waypoint(ego.get_location())

    return _cached(world, "waypoint", read)


def _weather(world: "carla.World") -> Any:
    return _cached(world, "weather", world.get_weather)


def _lanelet_attributes(world: "carla.World") -> Optional[dict[str, str]]:
    """The tags of the lanelet the ego is on, or ``None`` without a Lanelet2 map."""

    def read() -> Optional[dict[str, str]]:
        from ..coordinate import CarlaWorldPose, to_lanelet2  # noqa: PLC0415
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415

        ego = _ego(world)
        if ego is None:
            return None
        pose = to_lanelet2(CarlaWorldPose.from_carla_transform(ego.get_transform()))
        lanelet = MapManager.get_instance().lanelet_map.laneletLayer[pose.lanelet_id]
        return {str(k): str(v) for k, v in lanelet.attributes.items()}

    return _cached(world, "lanelet_attributes", read)


def _distance(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _nearby(world: "carla.World", prefix: str) -> Optional[int]:
    ego = _ego(world)
    if ego is None:
        return None
    here = ego.get_location()
    return sum(
        1
        for a in _actors(world)
        if a.id != ego.id
        and a.type_id.startswith(prefix)
        and _distance(a.get_location(), here) <= NEARBY_RADIUS_M
    )


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def ego_speed_kph(world: "carla.World") -> Optional[float]:
    """The ego's speed, in km/h."""
    ego = _ego(world)
    if ego is None:
        return None
    v = ego.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6


def in_junction(world: "carla.World") -> Optional[bool]:
    """Whether the ego is inside a junction (CARLA waypoint)."""
    waypoint = _waypoint(world)
    return None if waypoint is None else bool(waypoint.is_junction)


def speed_limit_kph(world: "carla.World") -> Optional[float]:
    """The speed limit for the ego's lane, in km/h.

    The Lanelet2 ``speed_limit`` tag of the ego's lanelet when there is one,
    otherwise CARLA's ``get_speed_limit()``.
    """
    tags = _lanelet_attributes(world)
    if tags is not None and "speed_limit" in tags:
        try:
            return float(tags["speed_limit"])
        except ValueError:
            pass
    ego = _ego(world)
    if ego is None:
        return None
    try:
        limit = float(ego.get_speed_limit() or 0.0)
    except Exception:
        return None
    return limit if limit > 0.0 else None


def lane_count(world: "carla.World") -> Optional[int]:
    """Driving lanes in the ego's direction of travel, its own included.

    Read from the CARLA waypoint, and only outside junctions.
    """
    waypoint = _waypoint(world)
    if waypoint is None or waypoint.is_junction:
        return None
    count = 1
    for step in ("get_left_lane", "get_right_lane"):
        lane = getattr(waypoint, step)()
        seen = 0
        while (
            lane is not None
            and seen < 16
            and lane.lane_id * waypoint.lane_id > 0
            and str(lane.lane_type).endswith("Driving")
        ):
            count += 1
            seen += 1
            lane = getattr(lane, step)()
    return count


def lanelet_location(world: "carla.World") -> Optional[str]:
    """The Lanelet2 ``location`` tag of the ego's lanelet (urban, nonurban, ...)."""
    tags = _lanelet_attributes(world)
    return None if tags is None else tags.get("location")


def lanelet_subtype(world: "carla.World") -> Optional[str]:
    """The Lanelet2 ``subtype`` tag of the ego's lanelet (road, highway, ...)."""
    tags = _lanelet_attributes(world)
    return None if tags is None else tags.get("subtype")


def lanelet_speed_limit_kph(world: "carla.World") -> Optional[float]:
    """The Lanelet2 ``speed_limit`` tag of the ego's lanelet, in km/h."""
    tags = _lanelet_attributes(world)
    if tags is None or "speed_limit" not in tags:
        return None
    try:
        return float(tags["speed_limit"])
    except ValueError:
        return None


def illumination(world: "carla.World") -> Optional[str]:
    """The illumination level, from the sun's altitude (:func:`illumination_level`)."""
    weather = _weather(world)
    if weather is None:
        return None
    return illumination_level(float(weather.sun_altitude_angle))


def rain(world: "carla.World") -> Optional[str]:
    """The rain level, from CARLA's 0-100 precipitation."""
    weather = _weather(world)
    return None if weather is None else intensity_level(float(weather.precipitation))


def fog(world: "carla.World") -> Optional[str]:
    """The fog level, from CARLA's 0-100 fog density."""
    weather = _weather(world)
    return None if weather is None else intensity_level(float(weather.fog_density))


def traffic_density(world: "carla.World") -> Optional[str]:
    """The traffic density level, from other vehicles within :data:`NEARBY_RADIUS_M`."""
    count = _nearby(world, "vehicle.")
    return None if count is None else traffic_density_level(count)


def pedestrian_nearby(world: "carla.World") -> Optional[bool]:
    """Whether a pedestrian is within :data:`NEARBY_RADIUS_M` of the ego."""
    count = _nearby(world, "walker.")
    return None if count is None else count > 0
