"""Which vehicles a traffic backend simulates mesoscopically, and when.

A run needs lane-level traffic only where something can see it: around the
ego, and around the vehicles CARLA's side drives, which the backend's traffic
has to react to.  Everywhere else a vehicle only has to get to the right place
at the right time, which a queue model does for a fraction of the cost and
without an actor in CARLA (:class:`~.base.SimulationFidelity`).

This module is the part of that which is the same for every backend: the
options, and the decision, from where each vehicle is and where the *focus
points* are, of which vehicles change level.  How a backend changes a
vehicle's level -- taking it out of its simulator and out of CARLA, and back --
is the backend's own.  Like :mod:`.config`, it imports nothing heavier than the
standard library, so the config layer can name the options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..config_flags import check_flag
from ..value_text import shown
from .base import SimulationFidelity

__all__ = [
    "MESOSCOPIC_ENGINES",
    "FidelityCandidate",
    "MesoscopicConfig",
    "closest_approach",
    "plan_fidelity",
]

#: The models far traffic may run in: SUMO's own mesoscopic simulation (a
#: second SUMO with ``--mesosim``), or the cut-down queue model of
#: :mod:`.mesoscopic`, in-process.
MESOSCOPIC_ENGINES = ("sumo", "queue")


def _number(key: str, value: float, requirement: str) -> float:
    """*value* as a float, or the keyed ``ValueError`` saying *requirement*.

    A traffic option's check compares the value and calls ``math.isfinite``
    on it, so whatever is not a number has to be refused first: a bool would
    pass as ``0`` or ``1`` (``period_s: true``, a 1 s period), and a string
    or null in a required field would raise a ``TypeError`` naming no key
    (#1663).  An int is taken as the float it is, unless it is too big for
    one (``10**400``), which is refused as not finite (#1652).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be {requirement} (got {shown(value)})")
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"{key} must be {requirement} (got {shown(value)})") from None


@dataclass
class MesoscopicConfig:
    """When a traffic backend simulates a vehicle mesoscopically.

    Distances are to the nearest *focus point* -- the ego and every vehicle
    CARLA's side drives -- in the backend's own frame, in metres.

    Attributes:
        enabled: Simulate far traffic mesoscopically at all.  Off, every
            vehicle is microscopic, as before this option existed.
        micro_radius_m: A mesoscopic vehicle this close to a focus point
            becomes microscopic, and an actor in CARLA.
        meso_radius_m: A microscopic vehicle further than this from every
            focus point becomes mesoscopic.  At least ``micro_radius_m``; the
            gap between the two keeps a vehicle on the boundary from changing
            level every update.  ``None``: 1.25 times ``micro_radius_m``.
        update_interval_s: Simulated seconds between two decisions.  Changing a
            vehicle's level is the costly part; the mesoscopic model itself
            steps every tick.
        scenario_vehicles: Let the scenario's own vehicles the backend drives
            (its NPCs, when the backend drives them) become mesoscopic too.
            Off by default: such a vehicle leaves the CARLA world while it is
            far, and a scenario that reads its actor -- a condition on its
            speed, say -- reads ``None`` meanwhile.  The ego never does.
        headway_s: The mesoscopic model's time gap between two vehicles
            leaving a road, per lane: its capacity.
        jam_spacing_m: Road length a vehicle takes up in a queue: a road holds
            ``lanes * length / jam_spacing_m`` vehicles before the road behind
            it has to wait.
        junction_control: Hold a mesoscopic vehicle at a red light, where the
            backend's signals say what is red.
        engine: The model far traffic runs in.  ``sumo`` (the default):
            SUMO's own mesoscopic simulation, a second SUMO started with
            ``--mesosim`` on the same network, with SUMO's junction model and
            the run's signals.  ``queue``: the cut-down queue model of
            :mod:`.mesoscopic`, in-process, with ``headway_s`` and
            ``jam_spacing_m`` as its capacity.
        segment_length_m: With ``engine: sumo``, the length of the segments
            SUMO's mesoscopic model cuts each road into
            (``--meso-edgelength``): a vehicle moves from one to the next, so
            a shorter one follows a speed it is told sooner, at some cost.
        horizon_s: How far ahead a decision looks, simulated seconds.  A
            vehicle's distance to a focus point is the closest the two come
            within this time, each at its velocity now: a mesoscopic vehicle
            that will come within ``micro_radius_m`` becomes microscopic
            before it does -- settled on its lane by the time it is near,
            rather than appearing there -- and a microscopic one that will
            come back within ``meso_radius_m`` stays.  ``0``: where they are
            now, as before this option existed.
        max_microscopic: At most this many of the vehicles that may change
            level are microscopic at once: the nearest, by the distance
            above; the rest are mesoscopic, however near.  It bounds what the
            far traffic costs in SUMO and CARLA on a dense map.  ``None``: no
            limit.  The ego, every vehicle CARLA drives and every vehicle
            that may not leave (a scenario's, unless ``scenario_vehicles``)
            are not counted, and are microscopic whatever it says.
    """

    enabled: bool = False
    micro_radius_m: float = 200.0
    meso_radius_m: Optional[float] = None
    update_interval_s: float = 1.0
    scenario_vehicles: bool = False
    headway_s: float = 1.5
    jam_spacing_m: float = 7.5
    junction_control: bool = True
    engine: str = "sumo"
    segment_length_m: float = 20.0
    horizon_s: float = 0.0
    max_microscopic: Optional[int] = None

    def __post_init__(self) -> None:
        # Every number must be finite as well as in range: NaN fails every
        # comparison, so a NaN radius would never demote anything and a NaN
        # interval would never decide again, with no error.  ``inf`` is
        # refused too: nothing reads it as "unbounded" here -- an infinite
        # radius is ``enabled=False`` said another way, an infinite interval
        # freezes every level after the first decision, and an infinite
        # headway or spacing stops the queue model -- and ``enabled`` already
        # says "off".  Each is a number first: a bool, a string or null is
        # refused by its key, and an int is kept as its float (#1663).
        # The switches are read strictly (config_flags.check_flag): "false"
        # is false and a typo is refused, named under ``mesoscopic.`` since
        # a bare ``scenario_vehicles`` is the SUMO option of that name (#1757).
        self.enabled = check_flag("mesoscopic.enabled", self.enabled)
        self.scenario_vehicles = check_flag(
            "mesoscopic.scenario_vehicles", self.scenario_vehicles
        )
        self.junction_control = check_flag(
            "mesoscopic.junction_control", self.junction_control
        )
        self.micro_radius_m = _number(
            "micro_radius_m", self.micro_radius_m, "positive and finite"
        )
        if self.meso_radius_m is not None:
            self.meso_radius_m = _number(
                "meso_radius_m", self.meso_radius_m, "finite or null"
            )
        self.update_interval_s = _number(
            "update_interval_s", self.update_interval_s, "finite and not negative"
        )
        if self.engine not in MESOSCOPIC_ENGINES:
            raise ValueError(
                f"mesoscopic.engine must be one of {', '.join(MESOSCOPIC_ENGINES)} "
                f"(got {self.engine!r})"
            )
        both = "positive and finite"
        self.segment_length_m = _number("segment_length_m", self.segment_length_m, both)
        if not (math.isfinite(self.segment_length_m) and self.segment_length_m > 0.0):
            raise ValueError(
                f"segment_length_m must be positive and finite (got {self.segment_length_m!r})"
            )
        self.horizon_s = _number("horizon_s", self.horizon_s, "finite and not negative")
        if not (math.isfinite(self.horizon_s) and self.horizon_s >= 0.0):
            raise ValueError(
                f"horizon_s must be finite and not negative (got {self.horizon_s!r})"
            )
        count = self.max_microscopic
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError(
                "max_microscopic must be a whole number, not negative, or null "
                f"(got {count!r})"
            )
        self.headway_s = _number("headway_s", self.headway_s, both)
        self.jam_spacing_m = _number("jam_spacing_m", self.jam_spacing_m, both)
        if not (math.isfinite(self.micro_radius_m) and self.micro_radius_m > 0.0):
            raise ValueError(
                "micro_radius_m must be positive and finite "
                f"(got {self.micro_radius_m!r})"
            )
        if self.meso_radius_m is not None and not math.isfinite(self.meso_radius_m):
            raise ValueError(
                f"meso_radius_m must be finite or null (got {self.meso_radius_m!r})"
            )
        if self.meso_radius_m is not None and self.meso_radius_m < self.micro_radius_m:
            raise ValueError(
                "meso_radius_m must be at least micro_radius_m "
                f"(got {self.meso_radius_m!r} < {self.micro_radius_m!r})"
            )
        if not (
            math.isfinite(self.update_interval_s) and self.update_interval_s >= 0.0
        ):
            raise ValueError(
                "update_interval_s must be finite and not negative "
                f"(got {self.update_interval_s!r})"
            )
        if not (
            math.isfinite(self.headway_s)
            and self.headway_s > 0.0
            and math.isfinite(self.jam_spacing_m)
            and self.jam_spacing_m > 0.0
        ):
            raise ValueError(
                "headway_s and jam_spacing_m must be positive and finite "
                f"(got {self.headway_s!r} and {self.jam_spacing_m!r})"
            )

    @property
    def demote_radius_m(self) -> float:
        """How far from every focus point a microscopic vehicle becomes mesoscopic."""
        if self.meso_radius_m is not None:
            return float(self.meso_radius_m)
        return 1.25 * float(self.micro_radius_m)

    def decide(
        self, current: SimulationFidelity, distance_m: float
    ) -> SimulationFidelity:
        """The level a vehicle at *current* and *distance_m* from the nearest
        focus point should be at.

        With hysteresis: between the two radii a vehicle keeps its level.
        """
        if (
            current is SimulationFidelity.MICROSCOPIC
            and distance_m > self.demote_radius_m
        ):
            return SimulationFidelity.MESOSCOPIC
        if (
            current is SimulationFidelity.MESOSCOPIC
            and distance_m <= self.micro_radius_m
        ):
            return SimulationFidelity.MICROSCOPIC
        return current

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MesoscopicConfig":
        """Return a config built from a plain mapping (e.g. a Hydra node)."""
        # Imported here: utils.config reads a dataclass's fields from its class,
        # which Codon cannot express, and the library check compiles no import
        # in a function it does not call (typecheck/library.py, UNCALLED).
        from ..utils.config import checked_options  # noqa: PLC0415

        return cls(**checked_options(cls, mapping))


@dataclass
class FidelityCandidate:
    """A vehicle whose level may change: its handle, its level, where it is,
    and its velocity (m/s, in the same frame), which only a decision that
    looks ahead (``horizon_s``) reads."""

    handle: str
    fidelity: SimulationFidelity
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


def closest_approach(
    dx: float, dy: float, vx: float, vy: float, horizon_s: float
) -> float:
    """The closest two points come within *horizon_s*, metres.

    One is at ``(dx, dy)`` from the other and moves at ``(vx, vy)`` relative
    to it; ``horizon_s=0`` is how far apart they are now.
    """
    t = 0.0
    speed2 = vx * vx + vy * vy
    if horizon_s > 0.0 and speed2 > 0.0:
        t = min(max(-(dx * vx + dy * vy) / speed2, 0.0), horizon_s)
    return math.hypot(dx + vx * t, dy + vy * t)


def plan_fidelity(
    config: MesoscopicConfig,
    candidates: Sequence[FidelityCandidate],
    focus: Sequence[tuple[float, float]],
    focus_velocities: Sequence[tuple[float, float]] = (),
) -> tuple[list[str], list[str]]:
    """Which of *candidates* change level: ``(to_mesoscopic, to_microscopic)``.

    Each list holds handles, the nearest to a focus point first in the second
    and the furthest first in the first, so a backend that can change only
    some of them in one update changes the ones that matter most.  With no
    focus point there is nothing to measure from, and nothing changes.

    Args:
        config: The radii.
        candidates: The vehicles that may change level, with their level and
            position.
        focus: The points traffic must be microscopic around, in the same
            frame as the candidates.
        focus_velocities: Each focus point's velocity, in the same order (a
            missing one stands); read with ``horizon_s`` only.
    """
    demote: list[tuple[float, str]] = []
    promote: list[tuple[float, str]] = []
    if not focus or not config.enabled:
        return [], []
    horizon = float(config.horizon_s)
    planned: list[tuple[float, FidelityCandidate, SimulationFidelity]] = []
    for candidate in candidates:
        distance = math.inf
        for i, (fx, fy) in enumerate(focus):
            fvx, fvy = focus_velocities[i] if i < len(focus_velocities) else (0.0, 0.0)
            distance = min(
                distance,
                closest_approach(
                    candidate.x - fx,
                    candidate.y - fy,
                    candidate.vx - fvx,
                    candidate.vy - fvy,
                    horizon,
                ),
            )
        planned.append(
            (distance, candidate, config.decide(candidate.fidelity, distance))
        )
    # Over the budget: the furthest of those that would be microscopic,
    # mesoscopic instead.
    over: set[str] = set()
    if config.max_microscopic is not None:
        micro = sorted(
            (distance, candidate.handle)
            for distance, candidate, wanted in planned
            if wanted is SimulationFidelity.MICROSCOPIC
        )
        over = {handle for _d, handle in micro[int(config.max_microscopic) :]}
    for distance, candidate, wanted in planned:
        if candidate.handle in over:
            wanted = SimulationFidelity.MESOSCOPIC
        if wanted is candidate.fidelity:
            continue
        if wanted is SimulationFidelity.MESOSCOPIC:
            demote.append((-distance, candidate.handle))
        else:
            promote.append((distance, candidate.handle))
    demote.sort()
    promote.sort()
    return [h for _d, h in demote], [h for _d, h in promote]
