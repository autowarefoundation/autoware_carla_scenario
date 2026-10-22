"""The traffic's parameters as ODD attributes: how its road users are drawn.

Every numeric traffic parameter
(:class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`,
:class:`~autoware_carla_scenario.traffic.parameters.PedestrianParameters`) is
a spread each vehicle or pedestrian draws its own value from.  The spread's
mean and standard deviation are each a built-in probe here -- what the run
set, read on every tick (``None``, a missing value, when it set none) -- and
each has a knob, so an ODD that names one bounds it and a sweep draws it per
case::

    # binding.yaml
    driver_speed_factor: {probe: vehicle_speed_factor_mean}
    driver_speed_spread: {probe: vehicle_speed_factor_std}

A case drawn with ``driver_speed_factor`` in ``[0.8, 1.0)`` and
``driver_speed_spread`` in ``[0.05, 0.15)`` writes
``traffic.vehicles.speed_factor.mean`` and ``.std``, and every vehicle of its
run draws its own speed factor from that normal distribution.

A parameter the run's backend does not support stays the backend's error
(:class:`~autoware_carla_scenario.traffic.parameters.TrafficParameterError`):
an ODD that draws a reaction time is an ODD for the ``sumo`` backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from openodd import OddKnob

from ..traffic.parameters import (
    PEDESTRIAN_BOUNDS,
    VEHICLE_BOUNDS,
    Spread,
    run_parameters,
)

__all__ = [
    "PARAMETER_RANGES",
    "TRAFFIC_KNOBS",
    "TRAFFIC_PROBES",
    "ParameterProbe",
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
    ("pedestrians", "walking_speed_mps"): ("m/s", (0.5, 2.5), (0.0, 0.5)),
    ("pedestrians", "min_gap_m"): ("m", (0.1, 2.0), (0.0, 0.5)),
    ("pedestrians", "red_light_violation_s"): ("s", (0.0, 60.0), (0.0, 30.0)),
}

#: A probe's name begins with the singular of its space.
_PREFIX = {"vehicles": "vehicle", "pedestrians": "pedestrian"}


class ParameterProbe:
    """Reads one statistic (``mean`` or ``std``) of one traffic parameter's
    spread, as the run set it."""

    def __init__(self, space: str, field: str, statistic: str) -> None:
        self.space = space
        self.field = field
        self.statistic = statistic
        self.__name__ = f"{_PREFIX[space]}_{field}_{statistic}"

    def __call__(self, world: Any) -> Optional[float]:
        del world  # the run's, not the world's
        recorded = run_parameters()
        if recorded is None:
            return None
        vehicles, pedestrians = recorded
        source: Any = vehicles if self.space == "vehicles" else pedestrians
        spread = getattr(source, self.field)
        if not isinstance(spread, Spread):
            return None
        return spread.mean if self.statistic == "mean" else spread.std

    def __repr__(self) -> str:
        return f"ParameterProbe({self.__name__})"


def _probes() -> (
    tuple[
        dict[str, tuple[Callable[[Any], Any], str]], dict[Callable[..., Any], OddKnob]
    ]
):
    probes: dict[str, tuple[Callable[[Any], Any], str]] = {}
    knobs: dict[Callable[..., Any], OddKnob] = {}
    for (space, field), (unit, mean_range, std_range) in PARAMETER_RANGES.items():
        bounds = (VEHICLE_BOUNDS if space == "vehicles" else PEDESTRIAN_BOUNDS)[field]
        for statistic, value_range in (("mean", mean_range), ("std", std_range)):
            probe = ParameterProbe(space, field, statistic)
            probes[probe.__name__] = (probe, unit)
            knobs[probe] = OddKnob(
                f"traffic.{space}.{field}.{statistic}",
                range=value_range,
                # The mean must lie within the parameter's own bounds.
                limits=bounds if statistic == "mean" else (0.0, float("inf")),
            )
    return probes, knobs


#: Probe name -> (probe, unit): ``vehicle_<field>_mean`` and ``_std`` for each
#: numeric vehicle parameter, ``pedestrian_<field>_mean`` and ``_std`` for
#: each pedestrian parameter.
TRAFFIC_PROBES: dict[str, tuple[Callable[[Any], Any], str]]
#: Probe -> the knob that sets it: ``traffic.<space>.<field>.mean`` or ``.std``.
TRAFFIC_KNOBS: dict[Callable[..., Any], OddKnob]
TRAFFIC_PROBES, TRAFFIC_KNOBS = _probes()
