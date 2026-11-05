"""The traffic parameters' probes by name: their knob keys and their ranges.

Every numeric traffic parameter has a probe for the mean and one for the
standard deviation of its spread
(:mod:`autoware_carla_scenario.odd.traffic_probes`).  This is the plain table
behind them, in plain Python so that modules checked with Codon can use it:
the unit and the ranges of each parameter (:data:`PARAMETER_RANGES`), what a
probe is called (:func:`probe_name`), the key its knob writes
(:func:`knob_key`), and the limits that knob keeps a value within
(:func:`knob_limits`).
"""

from __future__ import annotations

from ..traffic.parameters import BICYCLE_BOUNDS, PEDESTRIAN_BOUNDS, VEHICLE_BOUNDS

__all__ = [
    "PARAMETER_RANGES",
    "knob_key",
    "knob_limits",
    "probe_name",
]

#: ``(space, field)`` -> ``(unit, (mean low, mean high), (std low, std high))``:
#: the unit of the parameter, and the ranges a knob draws its mean and its
#: standard deviation from -- what a run can bring about, within the
#: parameter's bounds.
PARAMETER_RANGES: dict[
    tuple[str, str], tuple[str, tuple[float, float], tuple[float, float]]
] = {
    ("vehicles", "speed_factor"): ("", (0.5, 1.5), (0.0, 0.3)),
    ("vehicles", "min_gap_m"): ("m", (0.5, 10.0), (0.0, 3.0)),
    ("vehicles", "ignore_lights"): ("", (0.0, 1.0), (0.0, 0.5)),
    ("vehicles", "ignore_signs"): ("", (0.0, 1.0), (0.0, 0.5)),
    ("vehicles", "ignore_vehicles"): ("", (0.0, 1.0), (0.0, 0.5)),
    ("vehicles", "ignore_walkers"): ("", (0.0, 1.0), (0.0, 0.5)),
    ("vehicles", "random_lane_changes"): ("", (0.0, 1.0), (0.0, 0.5)),
    ("vehicles", "accel"): ("m/s^2", (0.5, 5.0), (0.0, 1.5)),
    ("vehicles", "decel"): ("m/s^2", (1.0, 9.0), (0.0, 2.0)),
    ("vehicles", "reaction_time_s"): ("s", (0.3, 3.0), (0.0, 1.0)),
    ("vehicles", "imperfection"): ("", (0.0, 1.0), (0.0, 0.5)),
    ("vehicles", "max_lateral_speed_mps"): ("m/s", (0.2, 2.0), (0.0, 0.5)),
    ("vehicles", "max_lateral_accel"): ("m/s^2", (0.2, 3.0), (0.0, 0.5)),
    ("pedestrians", "walking_speed_mps"): ("m/s", (0.5, 2.5), (0.0, 0.5)),
    ("pedestrians", "min_gap_m"): ("m", (0.1, 2.0), (0.0, 0.5)),
    ("pedestrians", "red_light_violation_s"): ("s", (0.0, 60.0), (0.0, 30.0)),
    ("bicycles", "max_speed_mps"): ("m/s", (2.0, 8.0), (0.0, 1.5)),
    ("bicycles", "accel"): ("m/s^2", (0.5, 2.0), (0.0, 0.5)),
    ("bicycles", "decel"): ("m/s^2", (1.0, 4.0), (0.0, 1.0)),
    ("bicycles", "min_gap_m"): ("m", (0.2, 2.0), (0.0, 0.5)),
    ("bicycles", "reaction_time_s"): ("s", (0.3, 2.0), (0.0, 0.5)),
    ("bicycles", "imperfection"): ("", (0.0, 1.0), (0.0, 0.3)),
    ("bicycles", "max_lateral_speed_mps"): ("m/s", (0.2, 2.0), (0.0, 0.5)),
    ("bicycles", "red_light_violation_s"): ("s", (0.0, 30.0), (0.0, 10.0)),
}

#: A probe's name begins with the singular of its space.
_PREFIX: dict[str, str] = {
    "vehicles": "vehicle",
    "pedestrians": "pedestrian",
    "bicycles": "bicycle",
}


def _space_bounds(space: str) -> dict[str, tuple[float, float]]:
    """The bounds of the parameters of *space* (``vehicles``, ``pedestrians``
    or ``bicycles``); a ``KeyError`` for any other space."""
    if space == "vehicles":
        return VEHICLE_BOUNDS
    if space == "pedestrians":
        return PEDESTRIAN_BOUNDS
    if space == "bicycles":
        return BICYCLE_BOUNDS
    raise KeyError(space)


def probe_name(space: str, field: str, statistic: str) -> str:
    """The probe of *statistic* (``mean`` or ``std``) of the parameter *field*
    of *space*: ``vehicle_speed_factor_mean``."""
    return f"{_PREFIX[space]}_{field}_{statistic}"


def knob_key(space: str, field: str, statistic: str) -> str:
    """The key the probe's knob writes: ``traffic.vehicles.speed_factor.mean``."""
    return f"traffic.{space}.{field}.{statistic}"


def knob_limits(space: str, field: str, statistic: str) -> tuple[float, float]:
    """The limits of the probe's knob: a mean must lie within the parameter's
    own bounds; a standard deviation is any value from 0 up."""
    if statistic == "mean":
        return _space_bounds(space)[field]
    return (0.0, float("inf"))
