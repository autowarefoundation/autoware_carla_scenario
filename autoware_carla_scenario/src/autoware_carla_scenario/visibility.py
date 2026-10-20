"""How far a driver sees: the visibility that night, rain and fog leave.

CARLA's weather is drawn, never seen by its traffic: the TrafficManager's
vehicles know where every other actor is whatever the fog.  A driver does
not, and drives so that they can stop within the road they can see.  So the
visibility is a quantity of the run of its own, in metres, which the weather
bounds (:func:`visibility_range`) and a traffic backend that models drivers'
sight follows -- the ``sumo`` backend holds its traffic to the speed it can
stop from within it (:func:`sight_speed`).  The TrafficManager has no such
notion: the ``traffic_manager`` backend logs what it would hold its traffic
to, and leaves it as it was.

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

__all__ = [
    "DEFAULT_DECEL",
    "DEFAULT_REACTION_S",
    "FOG_VISIBILITY_RANGES",
    "LIGHT_VISIBILITY_RANGES",
    "RAIN_VISIBILITY_RANGES",
    "forget_visibility",
    "set_world_visibility",
    "sight_speed",
    "visibility",
    "visibility_range",
    "world_visibility",
]

#: The braking (m/s²) and reaction time (s) of a car nothing says more of:
#: SUMO's passenger car's ``decel`` and ``tau``.
DEFAULT_DECEL: float = 4.5
DEFAULT_REACTION_S: float = 1.0

#: The distance a driver sees, metres, by CARLA's ``fog_density`` (0-100):
#: ``(fog_density, low, high)``, interpolated linearly between the rows.  A
#: clear day's is what a driver looks ahead on an open road; heavy fog's is
#: meteorology's dense fog (under 50 m).
FOG_VISIBILITY_RANGES: tuple[tuple[float, float, float], ...] = (
    (0.0, 250.0, 400.0),
    (30.0, 120.0, 250.0),
    (70.0, 40.0, 100.0),
    (100.0, 15.0, 40.0),
)

#: The same by the rain's ``precipitation`` (0-100): falling rain and the
#: spray the traffic throws up shorten the view less than fog does.
RAIN_VISIBILITY_RANGES: tuple[tuple[float, float, float], ...] = (
    (0.0, 250.0, 400.0),
    (30.0, 180.0, 320.0),
    (70.0, 100.0, 220.0),
    (100.0, 60.0, 150.0),
)

#: The same by the sun's altitude, degrees: from sunset to the end of civil
#: twilight (6 degrees below the horizon,
#: :func:`~autoware_carla_scenario.odd.probes.illumination_level`'s night)
#: down to what headlights light, from a dipped beam's 50 m to a main beam's
#: or a lit street's 150 m.
LIGHT_VISIBILITY_RANGES: tuple[tuple[float, float, float], ...] = (
    (-90.0, 50.0, 150.0),
    (-6.0, 50.0, 150.0),
    (0.0, 250.0, 400.0),
    (90.0, 250.0, 400.0),
)


def _interpolate(
    rows: tuple[tuple[float, float, float], ...], at: float
) -> tuple[float, float]:
    """``(low, high)`` of *rows* at *at*, clamped to the rows' first and last."""
    first = rows[0]
    if at <= first[0]:
        return first[1], first[2]
    for (x0, lo0, hi0), (x1, lo1, hi1) in zip(rows, rows[1:]):
        if at <= x1:
            t = (at - x0) / (x1 - x0)
            return lo0 + t * (lo1 - lo0), hi0 + t * (hi1 - hi0)
    last = rows[len(rows) - 1]
    return last[1], last[2]


def visibility_range(
    fog_density: float = 0.0,
    precipitation: float = 0.0,
    sun_altitude_angle: float = 90.0,
) -> tuple[float, float]:
    """The ``(low, high)`` distance a driver sees in this weather, metres.

    Each of the fog (:data:`FOG_VISIBILITY_RANGES`), the rain
    (:data:`RAIN_VISIBILITY_RANGES`) and the light
    (:data:`LIGHT_VISIBILITY_RANGES`) bounds it, and the shortest bound
    holds: fog at night is seen through no further than either.  Each row
    spans what eyes, windscreens, headlights and the road's own lighting make
    of the same weather.

    Args:
        fog_density: The weather's ``fog_density``, 0-100.
        precipitation: Its ``precipitation``, 0-100.
        sun_altitude_angle: Its ``sun_altitude_angle``, degrees above the
            horizon.
    """
    fog_low, fog_high = _interpolate(FOG_VISIBILITY_RANGES, float(fog_density))
    rain_low, rain_high = _interpolate(RAIN_VISIBILITY_RANGES, float(precipitation))
    light_low, light_high = _interpolate(
        LIGHT_VISIBILITY_RANGES, float(sun_altitude_angle)
    )
    return (
        min(fog_low, rain_low, light_low),
        min(fog_high, rain_high, light_high),
    )


def visibility(
    fog_density: float = 0.0,
    precipitation: float = 0.0,
    sun_altitude_angle: float = 90.0,
    quantile: float = 0.5,
) -> float:
    """The distance at *quantile* (0-1) of :func:`visibility_range`, metres.

    0.5, the default, is the middle of the range.
    """
    low, high = visibility_range(fog_density, precipitation, sun_altitude_angle)
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


def world_visibility(world_id: int) -> Optional[float]:
    """The visibility set in the world *world_id*, metres, or ``None``."""
    if world_id in _VISIBILITY:
        return _VISIBILITY[world_id]
    return None


def forget_visibility() -> None:
    """Forget every world's visibility; the runner calls this for every run."""
    _VISIBILITY.clear()
