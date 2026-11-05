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

The bicycles' parameters
(:class:`~autoware_carla_scenario.traffic.parameters.BicycleParameters`) are
the same: ``bicycle_<field>_mean`` and ``_std``, with knobs writing
``traffic.bicycles.<field>.mean`` and ``.std``.  Only SUMO simulates
bicycles, so an ODD that draws them is an ODD for the ``sumo`` backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from openodd import OddKnob

from ..traffic.parameters import Spread, run_bicycle_parameters, run_parameters
from .traffic_ranges import _PREFIX as _PREFIX
from .traffic_ranges import PARAMETER_RANGES as PARAMETER_RANGES
from .traffic_ranges import knob_key as knob_key
from .traffic_ranges import knob_limits as knob_limits
from .traffic_ranges import probe_name as probe_name

__all__ = [
    "PARAMETER_RANGES",
    "TRAFFIC_KNOBS",
    "TRAFFIC_PROBES",
    "ParameterProbe",
]

# The naming and the ranges (PARAMETER_RANGES, _PREFIX, probe_name(),
# knob_key(), knob_limits()) are the checked odd.traffic_ranges'; this module
# builds the probes and openodd's knobs from them.


class ParameterProbe:
    """Reads one statistic (``mean`` or ``std``) of one traffic parameter's
    spread, as the run set it."""

    def __init__(self, space: str, field: str, statistic: str) -> None:
        self.space = space
        self.field = field
        self.statistic = statistic
        self.__name__ = probe_name(space, field, statistic)

    def __call__(self, world: Any) -> Optional[float]:
        del world  # the run's, not the world's
        source: Any
        if self.space == "bicycles":
            source = run_bicycle_parameters()
        else:
            recorded = run_parameters()
            if recorded is None:
                return None
            vehicles, pedestrians = recorded
            source = vehicles if self.space == "vehicles" else pedestrians
        if source is None:
            return None
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
        for statistic, value_range in (("mean", mean_range), ("std", std_range)):
            probe = ParameterProbe(space, field, statistic)
            probes[probe.__name__] = (probe, unit)
            knobs[probe] = OddKnob(
                knob_key(space, field, statistic),
                range=value_range,
                # The mean must lie within the parameter's own bounds.
                limits=knob_limits(space, field, statistic),
            )
    return probes, knobs


#: Probe name -> (probe, unit): ``vehicle_<field>_mean`` and ``_std`` for each
#: numeric vehicle parameter, ``pedestrian_<field>_mean`` and ``_std`` for
#: each pedestrian parameter, ``bicycle_<field>_mean`` and ``_std`` for each
#: bicycle parameter.
TRAFFIC_PROBES: dict[str, tuple[Callable[[Any], Any], str]]
#: Probe -> the knob that sets it: ``traffic.<space>.<field>.mean`` or ``.std``.
TRAFFIC_KNOBS: dict[Callable[..., Any], OddKnob]
TRAFFIC_PROBES, TRAFFIC_KNOBS = _probes()
