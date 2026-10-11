"""The time on a trajectory's clock: the condition a vertex time stands for."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


class TrajectoryTimeCondition(BaseCondition):
    """Satisfied once a trajectory's clock has reached *time*.

    This is how a vertex is given a time: ``TrajectoryVertex(position,
    advance=TrajectoryTimeCondition(t))`` departs that vertex when the
    trajectory's clock reaches ``t`` -- time is one kind of departure
    condition among the others a vertex can carry, and
    :attr:`~autoware_carla_scenario.TrajectoryVertex.time` reads it back.

    As a vertex's whole ``advance`` it is evaluated by
    :class:`~autoware_carla_scenario.actions.FollowTrajectoryAction` against
    the trajectory's clock -- the scenario clock mapped through the action's
    :class:`~autoware_carla_scenario.trajectory.TrajectoryTiming` (domain,
    scale, offset) and its ``initial_distance_offset`` -- not by calling
    :meth:`check`.  Anywhere else (inside an ``AndCondition``, as a trigger)
    there is no trajectory clock to read, and :meth:`check` compares the
    scenario's elapsed time instead: the clock of an ``ABSOLUTE`` timing with
    no scale and no offset.  The Scenario Editor allows it only as a whole
    waypoint condition.

    Two of them with the same time are equal, so vertices given the same
    time compare (and hash) alike.

    Args:
        time: Seconds on the trajectory's clock.  Must be finite.
        label: Human-readable identifier.

    Raises:
        ValueError: If *time* is not finite.
    """

    #: Marks the class for :class:`~autoware_carla_scenario.TrajectoryVertex`,
    #: which recognises a time without importing this package (it pulls CARLA
    #: in, and the editor builds trajectories without a simulator).
    IS_TRAJECTORY_TIME: ClassVar[bool] = True

    # Declared for the static check (docs/typecheck.md).
    _time: float

    def __init__(self, time: float, *, label: str = "trajectory_time") -> None:
        super().__init__(label=label)
        if not math.isfinite(float(time)):
            raise ValueError("TrajectoryTimeCondition.time must be finite")
        self._time = float(time)

    @property
    def time(self) -> float:
        """The trajectory-clock time the condition waits for."""
        return self._time

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryTimeCondition):
            return NotImplemented
        return self._time == other._time

    def __hash__(self) -> int:
        # The class by name: Codon cannot hash a class.
        return hash(("TrajectoryTimeCondition", self._time))

    def __repr__(self) -> str:
        return f"TrajectoryTimeCondition({self._time!r})"

    def get_details(self) -> dict[str, Any]:
        return {"time": self._time}

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Satisfied once *elapsed* has reached :attr:`time` (see the class doc)."""
        if elapsed < self._time:
            return None
        return ScenarioResult(
            passed=True,
            message=f"trajectory clock {elapsed:.2f}s reached {self._time:.2f}s",
            elapsed_seconds=elapsed,
        )
