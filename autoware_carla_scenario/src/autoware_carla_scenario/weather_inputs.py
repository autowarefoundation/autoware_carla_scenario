"""What the weather rules read and which vehicle entry a vehicle follows.

The weather rules (:mod:`autoware_carla_scenario.weather_rules`) are read
with openodd; this is the plain part behind them, in plain Python so that
modules checked with Codon can use it: the quantities the framework reads
(:data:`QUANTITIES`), the weather's fields a rule may read
(:data:`WEATHER_INPUTS`), the values a rule reads of the weather
(:func:`rule_values`), and which ``vehicles`` entry of ``environment.rules``
a vehicle follows (:func:`vehicle_entry`).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping

__all__ = [
    "EGO_KEY",
    "QUANTITIES",
    "WEATHER_INPUTS",
    "rule_values",
    "vehicle_entry",
]

#: The quantities the framework reads.
QUANTITIES: tuple[str, ...] = ("tire_friction", "visibility", "sight_speed")

#: The weather's fields a part may read, as CARLA's ``WeatherParameters``
#: name them, and ``water``: the larger of ``wetness`` and
#: ``precipitation_deposits``, which the road's grip follows.
WEATHER_INPUTS: tuple[str, ...] = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
    "fog_distance",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "water",
)

#: The ``vehicles`` key naming the ego (``role_name`` ``Ego``).
EGO_KEY: str = "ego"
_EGO_ROLE: str = "Ego"


def rule_values(values: Mapping[str, float]) -> dict[str, float]:
    """What the rules read of the weather *values*: those, and ``water``
    from ``wetness`` and ``precipitation_deposits`` when not given."""
    out = {str(k): float(v) for k, v in values.items()}
    if "water" not in out and ("wetness" in out or "precipitation_deposits" in out):
        out["water"] = max(
            out.get("wetness", 0.0), out.get("precipitation_deposits", 0.0)
        )
    return out


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
