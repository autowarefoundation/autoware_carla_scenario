"""How far a driver sees: the visibility that night, rain and fog leave.

CARLA's weather is drawn, never seen by its traffic: the TrafficManager's
vehicles know where every other actor is whatever the fog.  A driver does
not, and drives so that they can stop within the road they can see.  So the
visibility is a quantity of the run of its own, in metres, which the weather
bounds (:func:`visibility_range`, by the weather rules of
:mod:`autoware_carla_scenario.weather_rules`) and a traffic backend that models drivers'
sight follows -- the ``sumo`` backend holds its traffic to the speed its
drivers keep in it (:func:`sight_speed_at`: the rules' ``sight_speed``, which
reads the visibility, never above the speed a car stops from within it,
:func:`sight_speed`).  The TrafficManager has no such
notion of its own: the ``traffic_manager`` backend holds its vehicles to it
with ``follow_visibility: true``, and otherwise logs what it would hold them
to.

:class:`~autoware_carla_scenario.EnvironmentAction` sets it, when it sets the
fog, the rain or the sun (or names a visibility), for the world it acts on
(:func:`set_world_visibility`); a backend reads it there every tick
(:func:`world_visibility`), so a weather that changes during a run changes it
too.  The runner forgets it at the start of every run
(:func:`forget_visibility`): a run that sets no weather leaves its traffic
seeing as it always did.

Like :mod:`autoware_carla_scenario.traffic.base`, this module imports neither
CARLA nor any traffic simulator.
"""

from __future__ import annotations

import math
from typing import Optional

from .weather_rules import weather_at, weather_range

__all__ = [
    "DEFAULT_DECEL",
    "DEFAULT_REACTION_S",
    "at_quantile",
    "driver_quantile",
    "forget_visibility",
    "set_world_visibility",
    "sight_speed",
    "sight_speed_at",
    "visibility",
    "visibility_range",
    "world_visibility",
]

#: The braking (m/s²) and reaction time (s) of a car nothing says more of:
#: SUMO's passenger car's ``decel`` and ``tau``.
DEFAULT_DECEL: float = 4.5
DEFAULT_REACTION_S: float = 1.0


def visibility_range(
    fog_density: float = 0.0,
    precipitation: float = 0.0,
    sun_altitude_angle: float = 90.0,
) -> tuple[float, float]:
    """The ``(low, high)`` distance a driver sees in this weather, metres.

    The weather rules' ``visibility``
    (:mod:`autoware_carla_scenario.weather_rules`): by default, the fog, the
    rain and the light each bound it, and the shortest bound holds -- fog at
    night is seen through no further than either.  Each range spans what
    eyes, windscreens, headlights and the road's own lighting make of the same
    weather.

    Args:
        fog_density: The weather's ``fog_density``, 0-100.
        precipitation: Its ``precipitation``, 0-100.
        sun_altitude_angle: Its ``sun_altitude_angle``, degrees above the
            horizon.
    """
    return weather_range(
        "visibility",
        {
            "fog_density": float(fog_density),
            "precipitation": float(precipitation),
            "sun_altitude_angle": float(sun_altitude_angle),
        },
    )


def visibility(
    fog_density: float = 0.0,
    precipitation: float = 0.0,
    sun_altitude_angle: float = 90.0,
    quantile: float = 0.5,
) -> float:
    """The distance at *quantile* (0-1) of the distribution of
    :func:`visibility_range`, metres.

    0.5, the default, is its median.  Each of the fog, the rain and the light
    gives its distance at *quantile*, and the shortest holds.
    """
    return weather_at(
        "visibility",
        {
            "fog_density": float(fog_density),
            "precipitation": float(precipitation),
            "sun_altitude_angle": float(sun_altitude_angle),
        },
        quantile,
    )


def at_quantile(low: float, high: float, quantile: float) -> float:
    """The point *quantile* (clamped to 0-1) of the way from *low* to *high*:
    the quantile of a uniform distribution.  The weather rules' quantities
    take theirs from their own distributions (:func:`visibility`)."""
    q = min(max(float(quantile), 0.0), 1.0)
    return low + q * (high - low)


def sight_speed(visibility_m: float, decel: float, reaction_s: float) -> float:
    """The highest speed, m/s, a driver stops from within *visibility_m*.

    The distance driven while reacting and then braking, ``v * reaction_s +
    v**2 / (2 * decel)``, is the visibility: 17 m/s (62 km/h) in 50 m of fog
    for a car that brakes at 4.5 m/s² a second after it sees.

    Args:
        visibility_m: How far the driver sees, metres.
        decel: The deceleration the driver brakes at, m/s².
        reaction_s: The time before they brake, seconds.

    Raises:
        ValueError: If *decel* is not positive.
    """
    if not decel > 0.0:
        raise ValueError(f"sight_speed: decel must be positive, got {decel}")
    distance = max(float(visibility_m), 0.0)
    lag = decel * max(float(reaction_s), 0.0)
    return -lag + math.sqrt(lag * lag + 2.0 * decel * distance)


def sight_speed_at(
    visibility_m: float,
    quantile: float = 0.5,
    decel: float = DEFAULT_DECEL,
    reaction_s: float = DEFAULT_REACTION_S,
) -> float:
    """The speed, m/s, a driver who sees *visibility_m* metres keeps: the
    weather rules' ``sight_speed`` at *quantile* (0-1).

    The rules read the visibility (a quantity of their own: the chain
    weather -> visibility -> speed) and are written for SUMO's default car,
    which brakes at :data:`DEFAULT_DECEL` a :data:`DEFAULT_REACTION_S` after
    it sees.  A car that brakes at *decel* after *reaction_s* scales that by
    its own :func:`sight_speed` over the default car's, and never keeps more
    than its own: whatever the rules say, it stops within what it sees.

    Raises:
        ValueError: If *decel* is not positive.
    """
    own = sight_speed(visibility_m, decel, reaction_s)
    default = sight_speed(visibility_m, DEFAULT_DECEL, DEFAULT_REACTION_S)
    kept = weather_at("sight_speed", {"visibility": float(visibility_m)}, quantile)
    if default > 0.0:
        kept = kept * own / default
    return min(kept, own)


def driver_quantile(key: str) -> float:
    """A quantile, 0-1, of the driver *key* names (a vehicle's id): the same
    for the same key in every run, and spread evenly over keys, so traffic
    keeps speeds of its own (:func:`sight_speed_at`) and a run repeats."""
    # FNV-1a, 32 bits, then murmur3's finalizer so that keys a character
    # apart ("1", "2") land far apart: no hash() (salted per process) and no
    # zlib (which Codon lacks).
    h = 2166136261
    for c in key:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return (h + 0.5) / 4294967296.0


#: World (episode) id -> the visibility, metres, set in it.
_VISIBILITY: dict[int, float] = {}


def set_world_visibility(world_id: int, visibility_m: float) -> None:
    """Record that drivers in the world *world_id* see *visibility_m* metres.

    Replaces what was set there before; another world's is dropped (an
    episode once left is not come back to).

    Raises:
        ValueError: If *visibility_m* is not positive.
    """
    if not visibility_m > 0.0:
        raise ValueError(f"visibility must be positive, got {visibility_m}")
    for old in [w for w in _VISIBILITY if w != world_id]:
        del _VISIBILITY[old]
    _VISIBILITY[world_id] = float(visibility_m)


def world_visibility(world_id: Optional[int]) -> Optional[float]:
    """The visibility set in the world *world_id*, metres, or ``None``.

    *world_id* is what :func:`~autoware_carla_scenario.world_reset.episode_of`
    reads off a world: ``None``, a world that names none, has none set.
    """
    if world_id is None:
        return None
    if world_id in _VISIBILITY:
        return _VISIBILITY[world_id]
    return None


def forget_visibility() -> None:
    """Forget every world's visibility; the runner calls this for every run."""
    _VISIBILITY.clear()
