"""What the weather rules read and which vehicle entry a vehicle follows.

The weather rules (:mod:`autoware_carla_scenario.weather_rules`) are read
with openodd; this is the plain part behind them, in plain Python so that
modules checked with Codon can use it: the quantities the framework reads
(:data:`QUANTITIES`), the causes a rule may read (:data:`WEATHER_INPUTS`)
and the settings that give them (:data:`CAUSE_SETTINGS`), the quantity each
of CARLA's weather fields is (:data:`CARLA_QUANTITIES`), the values a rule
reads (:func:`rule_values`), and which ``vehicles`` entry of ``environment.rules``
a vehicle follows (:func:`vehicle_entry`).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping

__all__ = [
    "CARLA_QUANTITIES",
    "CAUSE_SETTINGS",
    "EGO_KEY",
    "QUANTITIES",
    "WEATHER_INPUTS",
    "rule_values",
    "vehicle_entry",
]

#: The quantities the framework reads: CARLA's weather settings, which
#: render the conditions a run is given, and what those do to the road and
#: the drivers.
QUANTITIES: tuple[str, ...] = (
    "carla_precipitation",
    "carla_cloudiness",
    "carla_precipitation_deposits",
    "carla_wetness",
    "carla_wind_intensity",
    "carla_fog_density",
    "carla_sun_altitude",
    "carla_sun_azimuth",
    "tire_friction",
    "visibility",
    "sight_speed",
)

#: The causes a part may read: concepts of the taxonomy the rules name
#: (openodd's bundled default, a subset of tier4/taxonomy) -- the rainfall
#: rate (mm/h), the visibility through the fog (m), the time of day, the wind
#: speed (km/h) and the cloudiness.
WEATHER_INPUTS: tuple[str, ...] = (
    "rainfall_rate",
    "fog_visibility_range",
    "time_of_day",
    "wind_speed",
    "cloudiness",
)

#: The ``environment`` setting that gives each cause, in the unit its name
#: says: what a run's config, and the sampler, write.
CAUSE_SETTINGS: dict[str, str] = {
    "rainfall_rate_mm_h": "rainfall_rate",
    "fog_visibility_range_m": "fog_visibility_range",
    "time_of_day": "time_of_day",
    "wind_speed_kmh": "wind_speed",
    "cloud_cover": "cloudiness",
}

#: The quantity that is each of CARLA's weather fields, by the field's name
#: (``carla.WeatherParameters``).
CARLA_QUANTITIES: dict[str, str] = {
    "precipitation": "carla_precipitation",
    "cloudiness": "carla_cloudiness",
    "precipitation_deposits": "carla_precipitation_deposits",
    "wetness": "carla_wetness",
    "wind_intensity": "carla_wind_intensity",
    "fog_density": "carla_fog_density",
    "sun_altitude_angle": "carla_sun_altitude",
    "sun_azimuth_angle": "carla_sun_azimuth",
}

#: The ``vehicles`` key naming the ego (``role_name`` ``Ego``).
EGO_KEY: str = "ego"
_EGO_ROLE: str = "Ego"


def rule_values(values: Mapping[str, float]) -> dict[str, float]:
    """What the rules read of *values*: each as a float."""
    return {str(k): float(v) for k, v in values.items()}


def vehicle_entry(patterns: list[str], type_id: str, role: str) -> int:
    """The index of the first of the ``vehicles`` *patterns* the vehicle of
    blueprint *type_id* and ``role_name`` *role* matches, ``-1`` when none
    does: :data:`EGO_KEY` (in any case) matches the ego (*role* ``Ego``),
    any other pattern is a shell pattern of the blueprint
    (:func:`fnmatch.fnmatchcase`)."""
    for index, pattern in enumerate(patterns):
        if pattern.lower() == EGO_KEY:
            if role == _EGO_ROLE:
                return index
        elif fnmatch.fnmatchcase(type_id, pattern):
            return index
    return -1
