"""The ODD cover items every run samples.

ODD coverage asks which operating conditions the runs exercised, whatever the
scenario was, so these items are sampled by the runner on every tick rather
than declared by each scenario.  They are a first subset of the ISO 34503
taxonomy, one or more attributes from each of its three top-level categories:

* scenery: whether the ego is in a junction, the posted speed limit, the number
  of lanes it could drive in;
* environmental conditions: rain, fog, illumination (from the sun's altitude:
  CARLA has no clock);
* dynamic elements: the ego's speed, how many other vehicles are near it,
  whether a pedestrian is.

Each reads the world through the CARLA client.  A reading the simulator does
not support -- weather on a server that has none -- returns nothing, and the
item is then simply not covered; it does not fail the run.

The weather values are CARLA's 0-100 intensities, not physical units such as
mm/h, so the rain and fog buckets are named levels rather than measurements.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..constants import EGO_ROLE_NAME
from .items import CoverGroup, CoverItem, SamplingEvent

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "ILLUMINATION_LEVELS",
    "NEARBY_RADIUS_M",
    "OddProbe",
    "illumination_level",
    "intensity_level",
    "odd_cover_items",
    "traffic_density_level",
]

#: Radius around the ego in which other road users count as near it, in metres.
NEARBY_RADIUS_M = 50.0

#: Illumination levels, brightest first.
ILLUMINATION_LEVELS = ("day", "low_sun", "twilight", "night")

#: Rain and fog levels, from CARLA's 0-100 intensity.
INTENSITY_LEVELS = ("none", "light", "moderate", "heavy")

#: Traffic density levels, from the number of other vehicles near the ego.
TRAFFIC_DENSITY_LEVELS = ("none", "low", "medium", "high")


def illumination_level(sun_altitude_deg: float) -> str:
    """The illumination level for a sun at *sun_altitude_deg* above the horizon.

    Day from 15 degrees up; a low sun (glare, long shadows) below that;
    civil twilight down to 6 degrees below the horizon; night under it.
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


def _distance(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


class OddProbe:
    """Reads the ODD attributes of one run from the world.

    One probe serves every ODD item of a run, so the ego, its waypoint and the
    actor list are looked up once per tick however many items read them, and
    the CARLA map (slow to fetch) once per run.
    """

    def __init__(self, nearby_radius_m: float = NEARBY_RADIUS_M) -> None:
        self.nearby_radius_m = nearby_radius_m
        self._map: Any = None
        self._frame: Optional[int] = None
        self._cache: dict[str, Any] = {}

    def _cached(self, world: "carla.World", key: str, read: Callable[[], Any]) -> Any:
        try:
            frame: Optional[int] = int(world.get_snapshot().frame)
        except Exception:
            frame = None
        if frame is None or frame != self._frame:
            self._frame = frame
            self._cache = {}
        if key not in self._cache:
            self._cache[key] = read()
        return self._cache[key]

    def actors(self, world: "carla.World") -> list[Any]:
        return self._cached(world, "actors", lambda: list(world.get_actors()))

    def ego(self, world: "carla.World") -> Any:
        name = str(EGO_ROLE_NAME)

        def read() -> Any:
            return next(
                (
                    a
                    for a in self.actors(world)
                    if a.attributes.get("role_name") == name
                ),
                None,
            )

        return self._cached(world, "ego", read)

    def waypoint(self, world: "carla.World") -> Any:
        def read() -> Any:
            ego = self.ego(world)
            if ego is None:
                return None
            if self._map is None:
                self._map = world.get_map()
            return self._map.get_waypoint(ego.get_location())

        return self._cached(world, "waypoint", read)

    def weather(self, world: "carla.World") -> Any:
        return self._cached(world, "weather", world.get_weather)

    # -- attributes ----------------------------------------------------

    def ego_speed_kph(self, world: "carla.World") -> Optional[float]:
        ego = self.ego(world)
        if ego is None:
            return None
        v = ego.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6

    def in_junction(self, world: "carla.World") -> Optional[bool]:
        waypoint = self.waypoint(world)
        return None if waypoint is None else bool(waypoint.is_junction)

    def speed_limit_kph(self, world: "carla.World") -> Optional[float]:
        ego = self.ego(world)
        if ego is None:
            return None
        limit = float(ego.get_speed_limit() or 0.0)
        return limit if limit > 0.0 else None

    def lane_count(self, world: "carla.World") -> Optional[int]:
        """Driving lanes in the ego's direction of travel, its own included."""
        waypoint = self.waypoint(world)
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

    def illumination(self, world: "carla.World") -> Optional[str]:
        weather = self.weather(world)
        return illumination_level(float(weather.sun_altitude_angle))

    def rain(self, world: "carla.World") -> Optional[str]:
        return intensity_level(float(self.weather(world).precipitation))

    def fog(self, world: "carla.World") -> Optional[str]:
        return intensity_level(float(self.weather(world).fog_density))

    def _nearby(self, world: "carla.World", prefix: str) -> Optional[int]:
        ego = self.ego(world)
        if ego is None:
            return None
        here = ego.get_location()
        return sum(
            1
            for a in self.actors(world)
            if a.id != ego.id
            and a.type_id.startswith(prefix)
            and _distance(a.get_location(), here) <= self.nearby_radius_m
        )

    def traffic_density(self, world: "carla.World") -> Optional[str]:
        count = self._nearby(world, "vehicle.")
        return None if count is None else traffic_density_level(count)

    def pedestrian_nearby(self, world: "carla.World") -> Optional[bool]:
        count = self._nearby(world, "walker.")
        return None if count is None else count > 0


def odd_cover_items(probe: Optional[OddProbe] = None) -> list[CoverItem]:
    """The ODD cover items, sampled on every tick.

    Args:
        probe: What reads the world; a fresh :class:`OddProbe` when ``None``.
            Use one probe per run.
    """
    p = probe or OddProbe()

    def odd(name: str, expression: Callable[[Any], Any], **kw: Any) -> CoverItem:
        return CoverItem(
            name=f"odd.{name}",
            expression=expression,
            event=SamplingEvent.TICK,
            group=CoverGroup.ODD,
            **kw,
        )

    return [
        # Scenery
        odd(
            "scenery.junction",
            p.in_junction,
            values=[False, True],
            text="Ego inside a junction",
        ),
        odd(
            "scenery.speed_limit",
            p.speed_limit_kph,
            unit="km/h",
            buckets=[0, 30, 40, 50, 60, 80, 100, 130],
            text="Speed limit posted for the ego's lane",
        ),
        odd(
            "scenery.lane_count",
            p.lane_count,
            values=[1, 2, 3, 4],
            text="Driving lanes in the ego's direction (outside junctions)",
        ),
        # Environmental conditions
        odd(
            "environment.illumination",
            p.illumination,
            values=ILLUMINATION_LEVELS,
            text="Illumination, from the sun's altitude",
        ),
        odd(
            "environment.rain",
            p.rain,
            values=INTENSITY_LEVELS,
            text="Rain, from CARLA's 0-100 precipitation",
        ),
        odd(
            "environment.fog",
            p.fog,
            values=INTENSITY_LEVELS,
            text="Fog, from CARLA's 0-100 fog density",
        ),
        # Dynamic elements
        odd(
            "dynamic.ego_speed",
            p.ego_speed_kph,
            unit="km/h",
            range=(0.0, 120.0),
            every=10.0,
            text="Ego speed",
        ),
        odd(
            "dynamic.traffic_density",
            p.traffic_density,
            values=TRAFFIC_DENSITY_LEVELS,
            text=f"Other vehicles within {NEARBY_RADIUS_M:g} m of the ego",
        ),
        odd(
            "dynamic.pedestrian_nearby",
            p.pedestrian_nearby,
            values=[False, True],
            text=f"A pedestrian within {NEARBY_RADIUS_M:g} m of the ego",
        ),
    ]
