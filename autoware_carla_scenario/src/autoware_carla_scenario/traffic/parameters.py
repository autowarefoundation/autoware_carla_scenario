"""How a run's traffic drives, in words every traffic backend reads.

A backend's own options (:mod:`.config`, :mod:`.sumo.config`) say how the
simulator behind it is set up -- a port, a network, a step.  How its drivers
behave is a different question, asked of every backend alike: how fast
against the limit, how close behind the car ahead, whether they change lanes,
whether they slow for what they cannot see.  :class:`TrafficParameters` is that
question's one type, read from the ``traffic.parameters`` node::

    traffic:
      backend: sumo
      parameters:
        speed_factor: 0.9
        reaction_time_s: 1.5
        follow_visibility: true

Not every backend can do everything, and a parameter one cannot honour must not
be dropped in silence: a run that asked for drivers who react in 1.5 s and got
the TrafficManager's, which has no reaction time, is a run that tested
something other than what it says.  So each field names the backends that
support it (:data:`PARAMETER_BACKENDS`), each backend names the fields it
supports (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.supported_parameters`),
and a field set for a backend that does not support it is an error before the
run starts (:meth:`TrafficParameters.check_supported`).  A field left at
``None`` is the backend's own default.

========================== ====================== =============================
Field                      ``traffic_manager``    ``sumo``
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
``follow_visibility``      --                     held to sight
========================== ====================== =============================

The visibility (:mod:`autoware_carla_scenario.visibility`) is one of them:
``follow_visibility`` holds the traffic to the speed it stops from within what
its drivers see, with ``reaction_time_s`` and ``decel`` as the reaction and the
braking that speed is worked out from.

Like :mod:`.config`, this module imports nothing heavier than the standard
library, so the config layer and the editor can name it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = [
    "PARAMETER_BACKENDS",
    "TrafficParameterError",
    "TrafficParameters",
]

#: Each parameter, and the built-in backends that support it.  A third-party
#: backend states its own support
#: (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.supported_parameters`);
#: this table is what the error a backend raises names as the alternatives.
PARAMETER_BACKENDS: dict[str, list[str]] = {
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
    "follow_visibility": ["sumo"],
}


class TrafficParameterError(ValueError):
    """A traffic parameter was set for a backend that cannot honour it."""

    # Spelled out: Codon 0.19 cannot raise an exception subclass that leaves
    # its constructor to the built-in base.
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _check_fraction(name: str, value: Optional[float]) -> None:
    if value is not None and not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be within [0, 1] (got {value!r})")


def _check_positive(name: str, value: Optional[float]) -> None:
    if value is not None and not value > 0.0:
        raise ValueError(f"{name} must be positive (got {value!r})")


def _check_non_negative(name: str, value: Optional[float]) -> None:
    if value is not None and not value >= 0.0:
        raise ValueError(f"{name} must not be negative (got {value!r})")


@dataclass(frozen=True)
class TrafficParameters:
    """How the vehicles a traffic backend drives behave.

    Every field is optional: ``None`` leaves the backend's own default.  Which
    backend supports which field is :data:`PARAMETER_BACKENDS`.

    Attributes:
        speed_factor: Desired speed as a multiple of the speed limit (1.0: at
            the limit, 0.8: 20 % under it).  TrafficManager: its global speed
            difference, which a speed set on a vehicle (a background vehicle's
            or a scenario's ``SetSpeedAction``) overrides.  SUMO: each
            vehicle's own speed factor, drawn by SUMO around 1.0, scaled by it
            -- so the spread SUMO's drivers have is kept.
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
        follow_visibility: Hold the backend's own traffic to the speed it stops
            from within what its drivers see, once the run's weather says how
            far that is (:mod:`autoware_carla_scenario.visibility`, set by the
            ``EnvironmentAction``), and give it its own speed back when the
            weather lifts.  The SUMO backend does so by default.
    """

    speed_factor: Optional[float] = None
    min_gap_m: Optional[float] = None
    lane_changes: Optional[bool] = None
    ignore_lights: Optional[float] = None
    ignore_signs: Optional[float] = None
    ignore_vehicles: Optional[float] = None
    ignore_walkers: Optional[float] = None
    random_lane_changes: Optional[float] = None
    accel: Optional[float] = None
    decel: Optional[float] = None
    reaction_time_s: Optional[float] = None
    imperfection: Optional[float] = None
    follow_visibility: Optional[bool] = None

    def __post_init__(self) -> None:
        _check_positive("speed_factor", self.speed_factor)
        _check_non_negative("min_gap_m", self.min_gap_m)
        _check_fraction("ignore_lights", self.ignore_lights)
        _check_fraction("ignore_signs", self.ignore_signs)
        _check_fraction("ignore_vehicles", self.ignore_vehicles)
        _check_fraction("ignore_walkers", self.ignore_walkers)
        _check_fraction("random_lane_changes", self.random_lane_changes)
        _check_positive("accel", self.accel)
        _check_positive("decel", self.decel)
        _check_non_negative("reaction_time_s", self.reaction_time_s)
        _check_fraction("imperfection", self.imperfection)

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
            ("follow_visibility", self.follow_visibility is not None),
        ]
        return [name for name, is_set in values if is_set]

    def check_supported(self, backend: str, supported: set[str]) -> None:
        """Refuse every field set here that *backend* does not support.

        Args:
            backend: The backend's name, for the message.
            supported: The fields it supports.

        Raises:
            TrafficParameterError: Naming each unsupported field and the
                backends it can be used with.
        """
        problems: list[str] = []
        for name in self.set_fields():
            if name in supported:
                continue
            others = PARAMETER_BACKENDS.get(name, [])
            if others:
                where = " or ".join(f"'{b}'" for b in others)
                plural = "s" if len(others) > 1 else ""
                problems.append(
                    f"traffic.parameters.{name} can only be used with the "
                    f"{where} backend{plural}"
                )
            else:
                problems.append(
                    f"traffic.parameters.{name} is supported by no built-in backend"
                )
        if problems:
            raise TrafficParameterError(
                f"The '{backend}' traffic backend does not support "
                + "; ".join(problems)
                + ". Leave the field unset (null) or select a backend that "
                "supports it."
            )

    def restricted_to(self, supported: set[str]) -> "TrafficParameters":
        """These parameters with every field outside *supported* unset.

        For a backend that hands part of its traffic to another (the SUMO
        backend's scenario vehicles, driven by the TrafficManager): what that
        one can honour of them.
        """
        return TrafficParameters(
            speed_factor=self.speed_factor if "speed_factor" in supported else None,
            min_gap_m=self.min_gap_m if "min_gap_m" in supported else None,
            lane_changes=self.lane_changes if "lane_changes" in supported else None,
            ignore_lights=self.ignore_lights if "ignore_lights" in supported else None,
            ignore_signs=self.ignore_signs if "ignore_signs" in supported else None,
            ignore_vehicles=self.ignore_vehicles
            if "ignore_vehicles" in supported
            else None,
            ignore_walkers=self.ignore_walkers
            if "ignore_walkers" in supported
            else None,
            random_lane_changes=(
                self.random_lane_changes if "random_lane_changes" in supported else None
            ),
            accel=self.accel if "accel" in supported else None,
            decel=self.decel if "decel" in supported else None,
            reaction_time_s=self.reaction_time_s
            if "reaction_time_s" in supported
            else None,
            imperfection=self.imperfection if "imperfection" in supported else None,
            follow_visibility=(
                self.follow_visibility if "follow_visibility" in supported else None
            ),
        )

    def describe(self) -> dict[str, Any]:
        """The fields that are set, for a run's result record."""
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
            "follow_visibility": self.follow_visibility,
        }
        return {name: value for name, value in values.items() if value is not None}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrafficParameters":
        """Return parameters built from a plain mapping (e.g. a Hydra node)."""
        # Imported here: utils.config reads a dataclass's fields from its class,
        # which Codon cannot express, and the library check compiles no import
        # in a function it does not call (typecheck/library.py, UNCALLED).
        from ..utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))
