"""How a run's traffic behaves, in words every traffic backend reads.

A backend's own options (:mod:`.config`, :mod:`.sumo.config`) say how the
simulator behind it is set up -- a port, a network, a step.  How its road
users behave is a different question, asked of every backend alike, and it
has two parameter spaces of its own:

* :class:`VehicleParameters`, ``traffic.vehicles``: how fast against the
  limit, how close behind the car ahead, whether drivers change lanes,
  whether they slow for what they cannot see;
* :class:`PedestrianParameters`, ``traffic.pedestrians``: how fast people
  walk, how close they keep, whether they cross on red.

::

    traffic:
      backend: sumo
      vehicles:
        speed_factor: {mean: 0.9, std: 0.1}
        reaction_time_s: {mean: 1.5, std: 0.3}
      pedestrians:
        walking_speed_mps: {mean: 1.2, std: 0.2}

Every numeric parameter is a :class:`Spread` -- a mean and a standard
deviation -- and each road user draws its own value from it, a normal
distribution cut to the parameter's bounds (:data:`VEHICLE_BOUNDS`,
:data:`PEDESTRIAN_BOUNDS`): a population of drivers, not one driver copied.
A plain number -- in a config or a constructor -- is a spread of no width:
every road user takes that value.  An ODD bounds the mean and the
spread of each like any other attribute it samples
(:mod:`autoware_carla_scenario.odd.traffic_probes`).

Not every backend can do everything, and a parameter one cannot honour must not
be dropped in silence: a run that asked for drivers who react in 1.5 s and got
the TrafficManager's, which has no reaction time, is a run that tested
something other than what it says.  So each field names the backends that
support it (:data:`VEHICLE_PARAMETER_BACKENDS`,
:data:`PEDESTRIAN_PARAMETER_BACKENDS`), each backend names the fields it
supports (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.supported_vehicle_parameters`,
:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.supported_pedestrian_parameters`),
and a field set for a backend that does not support it is an error before the
run starts.  A field left at ``None`` is the backend's own default.

========================== ====================== =============================
``traffic.vehicles``       ``traffic_manager``    ``sumo``
========================== ====================== =============================
``speed_factor``           global speed diff.     each vehicle's ``speedFactor``
``min_gap_m``              distance to leader     ``minGap``
``lane_changes``           ``auto_lane_change``   lane-change mode
``ignore_lights``          per vehicle            --
``ignore_signs``           per vehicle            --
``ignore_vehicles``        per vehicle            --
``ignore_walkers``         per vehicle            --
``random_lane_changes``    per vehicle            --
``accel``                  --                     ``accel``
``decel``                  --                     ``decel``
``reaction_time_s``        --                     ``tau``
``imperfection``           --                     ``sigma``
``max_lateral_speed_mps``  --                     ``maxSpeedLat``
``max_lateral_accel``      --                     ``lcAccelLat`` (sublane)
``follow_visibility``      --                     held to sight
========================== ====================== =============================

========================== ====================== =============================
``traffic.pedestrians``    ``traffic_manager``    ``sumo``
========================== ====================== =============================
``walking_speed_mps``      --                     ``maxSpeed``
``min_gap_m``              --                     ``minGap``
``red_light_violation_s``  --                     ``jmDriveAfterRedTime``
========================== ====================== =============================

The TrafficManager drives no pedestrians: a scenario's own walk as the
scenario steers them, and it has no pedestrian traffic of its own.

The visibility (:mod:`autoware_carla_scenario.visibility`) is a vehicle
parameter: ``follow_visibility`` holds the traffic to the speed it stops from
within what its drivers see, with ``reaction_time_s`` and ``decel`` as the
reaction and the braking that speed is worked out from.

Like :mod:`.config`, this module imports nothing heavier than the standard
library, so the config layer and the editor can name it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

__all__ = [
    "PEDESTRIAN_BOUNDS",
    "PEDESTRIAN_PARAMETER_BACKENDS",
    "Spread",
    "SpreadOrValue",
    "as_spread",
    "VEHICLE_BOUNDS",
    "record_run_parameters",
    "run_parameters",
    "PedestrianParameters",
    "TrafficParameterError",
    "VEHICLE_PARAMETER_BACKENDS",
    "VEHICLE_PARAMETER_REQUIREMENTS",
    "VehicleParameters",
]

#: Each vehicle parameter, and the built-in backends that support it.  A
#: built-in backend's support is read off it; a third-party backend states its
#: own (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.supported_vehicle_parameters`).
#: This table is what an error names as the alternatives.
VEHICLE_PARAMETER_BACKENDS: dict[str, list[str]] = {
    "speed_factor": ["traffic_manager", "sumo"],
    "min_gap_m": ["traffic_manager", "sumo"],
    "lane_changes": ["traffic_manager", "sumo"],
    "ignore_lights": ["traffic_manager"],
    "ignore_signs": ["traffic_manager"],
    "ignore_vehicles": ["traffic_manager"],
    "ignore_walkers": ["traffic_manager"],
    "random_lane_changes": ["traffic_manager"],
    "accel": ["sumo"],
    "decel": ["sumo"],
    "reaction_time_s": ["sumo"],
    "imperfection": ["sumo"],
    "max_lateral_speed_mps": ["sumo"],
    "max_lateral_accel": ["sumo"],
    "follow_visibility": ["sumo"],
}

#: A vehicle parameter a built-in backend supports only with one of its options
#: set, and that option: what an error names when the option is not set.
VEHICLE_PARAMETER_REQUIREMENTS: dict[str, str] = {
    "max_lateral_accel": (
        "traffic.options.lateral_resolution_m (SUMO's sublane model, the only "
        "one with a lateral acceleration)"
    ),
}

#: The same for each pedestrian parameter.  Only SUMO simulates pedestrian
#: traffic: the TrafficManager drives vehicles alone.
PEDESTRIAN_PARAMETER_BACKENDS: dict[str, list[str]] = {
    "walking_speed_mps": ["sumo"],
    "min_gap_m": ["sumo"],
    "red_light_violation_s": ["sumo"],
}


class TrafficParameterError(ValueError):
    """A traffic parameter was set for a backend that cannot honour it."""

    # Spelled out: Codon 0.19 cannot raise an exception subclass that leaves
    # its constructor to the built-in base.
    def __init__(self, message: str) -> None:
        super().__init__(message)


def check_supported(
    node: str,
    backend: str,
    set_fields: list[str],
    supported: set[str],
    table: dict[str, list[str]],
    during_run: bool = False,
    before_run: Optional[set[str]] = None,
) -> None:
    """Refuse every field of *set_fields* that *backend* does not support.

    Args:
        node: The config node the fields are under (``traffic.vehicles``).
        backend: The backend's name, for the message.
        set_fields: The fields that are set.
        supported: The fields the backend supports.
        table: Each field and the built-in backends that support it.
        during_run: Whether the fields are to change during a run (a
            :class:`~autoware_carla_scenario.TrafficParametersAction`), which
            the message says, as *supported* is then what the backend can
            change then.
        before_run: With *during_run*, the fields the backend supports in its
            config, before the run: such a field is refused as one to set
            there instead.

    Raises:
        TrafficParameterError: Naming each unsupported field and the backends
            it can be used with.
    """
    problems: list[str] = []
    for name in set_fields:
        if name in supported:
            continue
        if before_run is not None and name in before_run:
            problems.append(
                f"{node}.{name} can only be set before the run, in its config"
            )
            continue
        others = table.get(name, [])
        if backend in others and name in VEHICLE_PARAMETER_REQUIREMENTS:
            problems.append(
                f"{node}.{name} needs {VEHICLE_PARAMETER_REQUIREMENTS[name]}"
            )
            continue
        if others:
            where = " or ".join(f"'{b}'" for b in others)
            plural = "s" if len(others) > 1 else ""
            problems.append(
                f"{node}.{name} can only be used with the {where} backend{plural}"
            )
        else:
            problems.append(f"{node}.{name} is supported by no built-in backend")
    if problems:
        what = "cannot change during a run:" if during_run else "does not support"
        raise TrafficParameterError(
            f"The '{backend}' traffic backend {what} "
            + "; ".join(problems)
            + ". Leave the field unset (null) or select a backend that "
            "supports it."
        )


#: The values a numeric parameter may take, ``(low, high)``: what a value is
#: checked against, and what each draw from its spread is cut to.
INF = float("inf")
VEHICLE_BOUNDS: dict[str, tuple[float, float]] = {
    "speed_factor": (0.05, INF),
    "min_gap_m": (0.0, INF),
    "ignore_lights": (0.0, 1.0),
    "ignore_signs": (0.0, 1.0),
    "ignore_vehicles": (0.0, 1.0),
    "ignore_walkers": (0.0, 1.0),
    "random_lane_changes": (0.0, 1.0),
    "accel": (0.05, INF),
    "decel": (0.05, INF),
    "reaction_time_s": (0.0, INF),
    "imperfection": (0.0, 1.0),
    "max_lateral_speed_mps": (0.05, INF),
    "max_lateral_accel": (0.05, INF),
}
PEDESTRIAN_BOUNDS: dict[str, tuple[float, float]] = {
    "walking_speed_mps": (0.05, INF),
    "min_gap_m": (0.0, INF),
    "red_light_violation_s": (0.0, INF),
}

#: How many draws a spread tries for one inside its bounds before it settles
#: for its mean.
_DRAW_TRIES = 32


@dataclass(frozen=True)
class Spread:
    """A numeric parameter as each road user draws it: a normal distribution
    of *mean* and standard deviation *std*, cut to the parameter's bounds.

    ``std`` 0 (the default) gives every road user *mean*.
    """

    mean: float
    std: float = 0.0

    def __post_init__(self) -> None:
        if not self.std >= 0.0:
            raise ValueError(f"std must not be negative (got {self.std!r})")

    def draw(self, rng: random.Random, bounds: tuple[float, float]) -> float:
        """One road user's value: from the normal distribution, cut to
        *bounds* by drawing again (a truncated normal); the mean when the
        bounds keep refusing."""
        low, high = bounds
        if self.std > 0.0:
            for _ in range(_DRAW_TRIES):
                value = rng.gauss(self.mean, self.std)
                if low <= value <= high:
                    return value
        return min(max(self.mean, low), high)

    def describe(self) -> dict[str, float]:
        """``{"mean": ..., "std": ...}``, for a result record."""
        return {"mean": self.mean, "std": self.std}


#: A numeric traffic parameter as given: a :class:`Spread`, or a plain number,
#: which every road user takes as it is (a spread of no width).
SpreadOrValue = Union[Spread, float]


def as_spread(value: Optional[SpreadOrValue]) -> Optional[Spread]:
    """*value* as a :class:`Spread`: a number is a spread of no width."""
    if value is None:
        return None
    if isinstance(value, Spread):
        return value
    return Spread(float(value))


def _check(
    name: str, value: Optional[Spread], bounds: dict[str, tuple[float, float]]
) -> None:
    if value is None:
        return
    low, high = bounds[name]
    if not (low <= value.mean <= high):
        raise ValueError(
            f"{name} must be within [{low:g}, {high:g}] (got a mean of {value.mean!r})"
        )


def _spread_of(name: str, value: Any) -> Optional[Spread]:
    """A config value as a :class:`Spread`: a number is a spread of no width;
    ``{mean, std}`` is one; ``{mean: null}`` (what the Hydra groups declare)
    or ``null`` is none."""
    if value is None or isinstance(value, Spread):
        return value
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"mean", "std"})
        if unknown:
            raise ValueError(f"{name}: unknown key(s) {unknown}; expected mean and std")
        mean = value.get("mean")
        std = value.get("std")
        if mean is None:
            if std is not None:
                raise ValueError(f"{name}: a std needs a mean")
            return None
        return Spread(float(mean), float(std or 0.0))
    return Spread(float(value))


def _spreads_from(
    cls: type, mapping: Mapping[str, Any], numeric: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    from ..utils.config import checked_options  # noqa: PLC0415

    checked = checked_options(cls, mapping)
    return {
        key: _spread_of(key, value) if key in numeric else value
        for key, value in checked.items()
    }


def _described(value: Any) -> Any:
    return value.describe() if isinstance(value, Spread) else value


@dataclass(init=False)
class VehicleParameters:
    """How the vehicles a traffic backend drives behave (``traffic.vehicles``).

    Every field is optional: ``None`` leaves the backend's own default.  Which
    backend supports which field is :data:`VEHICLE_PARAMETER_BACKENDS`.  Each
    numeric field is a :class:`Spread`: every vehicle draws its own value from
    it (a truncated normal, cut to :data:`VEHICLE_BOUNDS`), so a spread of no
    width gives all of them the same.

    Attributes:
        speed_factor: Desired speed as a multiple of the speed limit (1.0: at
            the limit, 0.8: 20 % under it).  TrafficManager: the vehicle's
            speed difference, which a speed set on it (a background vehicle's
            or a scenario's ``SetSpeedAction``) overrides.  SUMO: the
            vehicle's speed factor.
        min_gap_m: Gap kept to the vehicle ahead when standing, metres.
        lane_changes: Whether vehicles change lanes of their own accord (to
            overtake, to keep right).  ``False`` under SUMO still lets a
            vehicle change to a lane its route needs.
        ignore_lights: Probability (0-1) that a vehicle runs a red light.
        ignore_signs: Probability (0-1) that a vehicle ignores a stop sign.
        ignore_vehicles: Probability (0-1) that a vehicle ignores the vehicles
            in its way.
        ignore_walkers: Probability (0-1) that a vehicle ignores pedestrians.
        random_lane_changes: Probability (0-1), per step, that a vehicle
            changes lane to either side for no reason.
        accel: Acceleration, m/s².
        decel: Braking a driver uses, m/s²; also what ``follow_visibility``
            works the speed out from.
        reaction_time_s: Reaction time, seconds (SUMO's ``tau``, which also
            sets the time headway kept); also what ``follow_visibility`` works
            the speed out from.
        imperfection: Driver imperfection, 0 (perfect) to 1 (SUMO's
            ``sigma``).
        max_lateral_speed_mps: The fastest a vehicle moves sideways, m/s
            (SUMO's ``maxSpeedLat``; 1.0 by default): a lane change takes at
            least a lane's width over this, whatever
            ``traffic.options.lane_change_duration_s`` says.
        max_lateral_accel: The hardest a vehicle accelerates sideways, m/s²
            (SUMO's ``lcAccelLat``; 1.0 by default).  Only SUMO's sublane
            model has one: it needs ``traffic.options.lateral_resolution_m``.
        follow_visibility: Hold the backend's own traffic to the speed it stops
            from within what its drivers see, once the run's weather says how
            far that is (:mod:`autoware_carla_scenario.visibility`, set by the
            ``EnvironmentAction``), and give it its own speed back when the
            weather lifts.  The SUMO backend does so by default.
    """

    speed_factor: Optional[Spread] = None
    min_gap_m: Optional[Spread] = None
    lane_changes: Optional[bool] = None
    ignore_lights: Optional[Spread] = None
    ignore_signs: Optional[Spread] = None
    ignore_vehicles: Optional[Spread] = None
    ignore_walkers: Optional[Spread] = None
    random_lane_changes: Optional[Spread] = None
    accel: Optional[Spread] = None
    decel: Optional[Spread] = None
    reaction_time_s: Optional[Spread] = None
    imperfection: Optional[Spread] = None
    max_lateral_speed_mps: Optional[Spread] = None
    max_lateral_accel: Optional[Spread] = None
    follow_visibility: Optional[bool] = None

    def __init__(
        self,
        speed_factor: Optional[SpreadOrValue] = None,
        min_gap_m: Optional[SpreadOrValue] = None,
        lane_changes: Optional[bool] = None,
        ignore_lights: Optional[SpreadOrValue] = None,
        ignore_signs: Optional[SpreadOrValue] = None,
        ignore_vehicles: Optional[SpreadOrValue] = None,
        ignore_walkers: Optional[SpreadOrValue] = None,
        random_lane_changes: Optional[SpreadOrValue] = None,
        accel: Optional[SpreadOrValue] = None,
        decel: Optional[SpreadOrValue] = None,
        reaction_time_s: Optional[SpreadOrValue] = None,
        imperfection: Optional[SpreadOrValue] = None,
        max_lateral_speed_mps: Optional[SpreadOrValue] = None,
        max_lateral_accel: Optional[SpreadOrValue] = None,
        follow_visibility: Optional[bool] = None,
    ) -> None:
        # Written out rather than generated, to take a plain number as well as
        # a spread (as_spread) and keep the fields themselves spreads.
        self.speed_factor = as_spread(speed_factor)
        self.min_gap_m = as_spread(min_gap_m)
        self.lane_changes = lane_changes
        self.ignore_lights = as_spread(ignore_lights)
        self.ignore_signs = as_spread(ignore_signs)
        self.ignore_vehicles = as_spread(ignore_vehicles)
        self.ignore_walkers = as_spread(ignore_walkers)
        self.random_lane_changes = as_spread(random_lane_changes)
        self.accel = as_spread(accel)
        self.decel = as_spread(decel)
        self.reaction_time_s = as_spread(reaction_time_s)
        self.imperfection = as_spread(imperfection)
        self.max_lateral_speed_mps = as_spread(max_lateral_speed_mps)
        self.max_lateral_accel = as_spread(max_lateral_accel)
        self.follow_visibility = follow_visibility
        self.__post_init__()

    def __post_init__(self) -> None:
        _check("speed_factor", self.speed_factor, VEHICLE_BOUNDS)
        _check("min_gap_m", self.min_gap_m, VEHICLE_BOUNDS)
        _check("ignore_lights", self.ignore_lights, VEHICLE_BOUNDS)
        _check("ignore_signs", self.ignore_signs, VEHICLE_BOUNDS)
        _check("ignore_vehicles", self.ignore_vehicles, VEHICLE_BOUNDS)
        _check("ignore_walkers", self.ignore_walkers, VEHICLE_BOUNDS)
        _check("random_lane_changes", self.random_lane_changes, VEHICLE_BOUNDS)
        _check("accel", self.accel, VEHICLE_BOUNDS)
        _check("decel", self.decel, VEHICLE_BOUNDS)
        _check("reaction_time_s", self.reaction_time_s, VEHICLE_BOUNDS)
        _check("imperfection", self.imperfection, VEHICLE_BOUNDS)
        _check("max_lateral_speed_mps", self.max_lateral_speed_mps, VEHICLE_BOUNDS)
        _check("max_lateral_accel", self.max_lateral_accel, VEHICLE_BOUNDS)

    def set_fields(self) -> list[str]:
        """The names of the fields that are not ``None``, in declared order."""
        values: list[tuple[str, bool]] = [
            ("speed_factor", self.speed_factor is not None),
            ("min_gap_m", self.min_gap_m is not None),
            ("lane_changes", self.lane_changes is not None),
            ("ignore_lights", self.ignore_lights is not None),
            ("ignore_signs", self.ignore_signs is not None),
            ("ignore_vehicles", self.ignore_vehicles is not None),
            ("ignore_walkers", self.ignore_walkers is not None),
            ("random_lane_changes", self.random_lane_changes is not None),
            ("accel", self.accel is not None),
            ("decel", self.decel is not None),
            ("reaction_time_s", self.reaction_time_s is not None),
            ("imperfection", self.imperfection is not None),
            ("max_lateral_speed_mps", self.max_lateral_speed_mps is not None),
            ("max_lateral_accel", self.max_lateral_accel is not None),
            ("follow_visibility", self.follow_visibility is not None),
        ]
        return [name for name, is_set in values if is_set]

    def check_supported(self, backend: str, supported: set[str]) -> None:
        """Refuse every field set here that *backend* does not support
        (:func:`check_supported`, under ``traffic.vehicles``)."""
        check_supported(
            "traffic.vehicles",
            backend,
            self.set_fields(),
            supported,
            VEHICLE_PARAMETER_BACKENDS,
        )

    def restricted_to(self, supported: set[str]) -> "VehicleParameters":
        """These parameters with every field outside *supported* unset.

        For a backend that hands part of its traffic to another (the SUMO
        backend's scenario vehicles, driven by the TrafficManager): what that
        one can honour of them.
        """
        return VehicleParameters(
            speed_factor=self.speed_factor if "speed_factor" in supported else None,
            min_gap_m=self.min_gap_m if "min_gap_m" in supported else None,
            lane_changes=self.lane_changes if "lane_changes" in supported else None,
            ignore_lights=self.ignore_lights if "ignore_lights" in supported else None,
            ignore_signs=self.ignore_signs if "ignore_signs" in supported else None,
            ignore_vehicles=(
                self.ignore_vehicles if "ignore_vehicles" in supported else None
            ),
            ignore_walkers=(
                self.ignore_walkers if "ignore_walkers" in supported else None
            ),
            random_lane_changes=(
                self.random_lane_changes if "random_lane_changes" in supported else None
            ),
            accel=self.accel if "accel" in supported else None,
            decel=self.decel if "decel" in supported else None,
            reaction_time_s=(
                self.reaction_time_s if "reaction_time_s" in supported else None
            ),
            imperfection=self.imperfection if "imperfection" in supported else None,
            max_lateral_speed_mps=(
                self.max_lateral_speed_mps
                if "max_lateral_speed_mps" in supported
                else None
            ),
            max_lateral_accel=(
                self.max_lateral_accel if "max_lateral_accel" in supported else None
            ),
            follow_visibility=(
                self.follow_visibility if "follow_visibility" in supported else None
            ),
        )

    def overridden_by(self, overrides: "VehicleParameters") -> "VehicleParameters":
        """These parameters with every field *overrides* sets replaced by its
        value; the rest kept (a :class:`~autoware_carla_scenario.TrafficParametersAction`)."""
        return VehicleParameters(
            speed_factor=(
                overrides.speed_factor
                if overrides.speed_factor is not None
                else self.speed_factor
            ),
            min_gap_m=(
                overrides.min_gap_m
                if overrides.min_gap_m is not None
                else self.min_gap_m
            ),
            lane_changes=(
                overrides.lane_changes
                if overrides.lane_changes is not None
                else self.lane_changes
            ),
            ignore_lights=(
                overrides.ignore_lights
                if overrides.ignore_lights is not None
                else self.ignore_lights
            ),
            ignore_signs=(
                overrides.ignore_signs
                if overrides.ignore_signs is not None
                else self.ignore_signs
            ),
            ignore_vehicles=(
                overrides.ignore_vehicles
                if overrides.ignore_vehicles is not None
                else self.ignore_vehicles
            ),
            ignore_walkers=(
                overrides.ignore_walkers
                if overrides.ignore_walkers is not None
                else self.ignore_walkers
            ),
            random_lane_changes=(
                overrides.random_lane_changes
                if overrides.random_lane_changes is not None
                else self.random_lane_changes
            ),
            accel=(overrides.accel if overrides.accel is not None else self.accel),
            decel=(overrides.decel if overrides.decel is not None else self.decel),
            reaction_time_s=(
                overrides.reaction_time_s
                if overrides.reaction_time_s is not None
                else self.reaction_time_s
            ),
            imperfection=(
                overrides.imperfection
                if overrides.imperfection is not None
                else self.imperfection
            ),
            max_lateral_speed_mps=(
                overrides.max_lateral_speed_mps
                if overrides.max_lateral_speed_mps is not None
                else self.max_lateral_speed_mps
            ),
            max_lateral_accel=(
                overrides.max_lateral_accel
                if overrides.max_lateral_accel is not None
                else self.max_lateral_accel
            ),
            follow_visibility=(
                overrides.follow_visibility
                if overrides.follow_visibility is not None
                else self.follow_visibility
            ),
        )

    def describe(self) -> dict[str, Any]:
        """The fields that are set, for a run's result record: a spread as
        ``{"mean": ..., "std": ...}``."""
        values: dict[str, Any] = {
            "speed_factor": self.speed_factor,
            "min_gap_m": self.min_gap_m,
            "lane_changes": self.lane_changes,
            "ignore_lights": self.ignore_lights,
            "ignore_signs": self.ignore_signs,
            "ignore_vehicles": self.ignore_vehicles,
            "ignore_walkers": self.ignore_walkers,
            "random_lane_changes": self.random_lane_changes,
            "accel": self.accel,
            "decel": self.decel,
            "reaction_time_s": self.reaction_time_s,
            "imperfection": self.imperfection,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_lateral_accel": self.max_lateral_accel,
            "follow_visibility": self.follow_visibility,
        }
        return {
            name: _described(value)
            for name, value in values.items()
            if value is not None
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "VehicleParameters":
        """Return parameters built from a plain mapping (e.g. a Hydra node):
        a numeric field is a number or ``{mean, std}``."""
        return cls(**_spreads_from(cls, mapping, VEHICLE_BOUNDS))


@dataclass(init=False)
class PedestrianParameters:
    """How the pedestrians a traffic backend simulates behave
    (``traffic.pedestrians``).

    Every field is optional: ``None`` leaves the backend's own default.  Which
    backend supports which field is :data:`PEDESTRIAN_PARAMETER_BACKENDS`.
    Each field is a :class:`Spread`: every pedestrian draws its own value
    from it (cut to :data:`PEDESTRIAN_BOUNDS`).

    Attributes:
        walking_speed_mps: The speed people walk at, m/s (SUMO's
            ``maxSpeed``; 1.39 by default, with SUMO's own spread of 0.1).
        min_gap_m: Gap kept to the person ahead, metres (SUMO's ``minGap``;
            0.25 by default).
        red_light_violation_s: Walk onto a crossing whose light has been red
            for less than this, seconds (SUMO's ``jmDriveAfterRedTime``): a
            large value has people cross on red whenever the way is free.
            ``None``: they wait for green.
    """

    walking_speed_mps: Optional[Spread] = None
    min_gap_m: Optional[Spread] = None
    red_light_violation_s: Optional[Spread] = None

    def __init__(
        self,
        walking_speed_mps: Optional[SpreadOrValue] = None,
        min_gap_m: Optional[SpreadOrValue] = None,
        red_light_violation_s: Optional[SpreadOrValue] = None,
    ) -> None:
        # Written out rather than generated, to take a plain number as well as
        # a spread (as_spread) and keep the fields themselves spreads.
        self.walking_speed_mps = as_spread(walking_speed_mps)
        self.min_gap_m = as_spread(min_gap_m)
        self.red_light_violation_s = as_spread(red_light_violation_s)
        self.__post_init__()

    def __post_init__(self) -> None:
        _check("walking_speed_mps", self.walking_speed_mps, PEDESTRIAN_BOUNDS)
        _check("min_gap_m", self.min_gap_m, PEDESTRIAN_BOUNDS)
        _check("red_light_violation_s", self.red_light_violation_s, PEDESTRIAN_BOUNDS)

    def set_fields(self) -> list[str]:
        """The names of the fields that are not ``None``, in declared order."""
        values: list[tuple[str, bool]] = [
            ("walking_speed_mps", self.walking_speed_mps is not None),
            ("min_gap_m", self.min_gap_m is not None),
            ("red_light_violation_s", self.red_light_violation_s is not None),
        ]
        return [name for name, is_set in values if is_set]

    def check_supported(self, backend: str, supported: set[str]) -> None:
        """Refuse every field set here that *backend* does not support
        (:func:`check_supported`, under ``traffic.pedestrians``)."""
        check_supported(
            "traffic.pedestrians",
            backend,
            self.set_fields(),
            supported,
            PEDESTRIAN_PARAMETER_BACKENDS,
        )

    def overridden_by(
        self, overrides: "PedestrianParameters"
    ) -> "PedestrianParameters":
        """These parameters with every field *overrides* sets replaced by its
        value; the rest kept (a :class:`~autoware_carla_scenario.TrafficParametersAction`)."""
        return PedestrianParameters(
            walking_speed_mps=(
                overrides.walking_speed_mps
                if overrides.walking_speed_mps is not None
                else self.walking_speed_mps
            ),
            min_gap_m=(
                overrides.min_gap_m
                if overrides.min_gap_m is not None
                else self.min_gap_m
            ),
            red_light_violation_s=(
                overrides.red_light_violation_s
                if overrides.red_light_violation_s is not None
                else self.red_light_violation_s
            ),
        )

    def describe(self) -> dict[str, Any]:
        """The fields that are set, for a run's result record."""
        values: dict[str, Any] = {
            "walking_speed_mps": self.walking_speed_mps,
            "min_gap_m": self.min_gap_m,
            "red_light_violation_s": self.red_light_violation_s,
        }
        return {
            name: _described(value)
            for name, value in values.items()
            if value is not None
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PedestrianParameters":
        """Return parameters built from a plain mapping (e.g. a Hydra node):
        each field is a number or ``{mean, std}``."""
        return cls(**_spreads_from(cls, mapping, PEDESTRIAN_BOUNDS))


#: The parameters the current run's traffic backend was given, for the ODD's
#: probes to read (:func:`record_run_parameters`).
_RUN_PARAMETERS: list[tuple[VehicleParameters, PedestrianParameters]] = []


def record_run_parameters(
    vehicles: VehicleParameters, pedestrians: PedestrianParameters
) -> None:
    """Record the parameters the run's traffic is driven with; the runner
    calls this for every run, after the backend is prepared."""
    _RUN_PARAMETERS.clear()
    _RUN_PARAMETERS.append((vehicles, pedestrians))


def run_parameters() -> Optional[tuple[VehicleParameters, PedestrianParameters]]:
    """What :func:`record_run_parameters` last recorded, or ``None``."""
    if not _RUN_PARAMETERS:
        return None
    return _RUN_PARAMETERS[0]
