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

from .base import SimulationFidelity

__all__ = ["FidelityCandidate", "MesoscopicConfig", "plan_fidelity"]


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
    """

    enabled: bool = False
    micro_radius_m: float = 200.0
    meso_radius_m: Optional[float] = None
    update_interval_s: float = 1.0
    scenario_vehicles: bool = False
    headway_s: float = 1.5
    jam_spacing_m: float = 7.5
    junction_control: bool = True

    def __post_init__(self) -> None:
        if not self.micro_radius_m > 0.0:
            raise ValueError(
                f"micro_radius_m must be positive (got {self.micro_radius_m!r})"
            )
        if self.meso_radius_m is not None and self.meso_radius_m < self.micro_radius_m:
            raise ValueError(
                "meso_radius_m must be at least micro_radius_m "
                f"(got {self.meso_radius_m!r} < {self.micro_radius_m!r})"
            )
        if not self.update_interval_s >= 0.0:
            raise ValueError(
                f"update_interval_s must not be negative (got {self.update_interval_s!r})"
            )
        if not (self.headway_s > 0.0 and self.jam_spacing_m > 0.0):
            raise ValueError("headway_s and jam_spacing_m must be positive")

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
    """A vehicle whose level may change: its handle, its level, where it is."""

    handle: str
    fidelity: SimulationFidelity
    x: float
    y: float


def plan_fidelity(
    config: MesoscopicConfig,
    candidates: Sequence[FidelityCandidate],
    focus: Sequence[tuple[float, float]],
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
    """
    demote: list[tuple[float, str]] = []
    promote: list[tuple[float, str]] = []
    if not focus or not config.enabled:
        return [], []
    for candidate in candidates:
        distance = min(
            math.hypot(candidate.x - fx, candidate.y - fy) for fx, fy in focus
        )
        wanted = config.decide(candidate.fidelity, distance)
        if wanted is candidate.fidelity:
            continue
        if wanted is SimulationFidelity.MESOSCOPIC:
            demote.append((-distance, candidate.handle))
        else:
            promote.append((distance, candidate.handle))
    demote.sort()
    promote.sort()
    return [h for _d, h in demote], [h for _d, h in promote]
