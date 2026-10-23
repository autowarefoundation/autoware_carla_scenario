"""Pace an entity's own driver to its waypoints' times, and log the arrivals.

A :class:`~.routing.RoutingAction` in ``driver`` mode hands the route to
whatever drives the entity -- the TrafficManager, SUMO, a pedestrian's route
walker -- and has it keep the waypoints' timing by speed alone: every tick
:class:`RoutePacer` works out the speed that reaches the next timed waypoint
on time (the distance left along the planned path over the time left),
clamped to ``[0, limit]`` and smoothed, and the action sends it as the
driver's desired speed.  The driver still stops for lights and yields; a late
entity catches up as fast as the limit allows.

A waypoint with a departure condition rather than a time is met when the
condition is predicted to hold: for a rising ``route_progress`` (the
condition a logical scenario times its road users by) from the measured
entity's progress rate.  One that cannot be predicted -- another kind of
condition, or an entity that is not moving forward -- is approached at
cruise speed, braking to stop :data:`HOLD_MARGIN_M` short of the waypoint,
and held there until it holds.  A hold in the measured entity's way can
deadlock both (the ego waits behind the entity that waits for the ego's
progress): prediction slows the entity down well before it would hold, which
is why a prediction is preferred to a hold wherever there is one.

Whatever the mode, :meth:`RoutePacer.observe` notes when each timed waypoint
is reached against when it was meant to be, for the trajectory log
(:class:`~autoware_carla_scenario.waypoint_arrival.WaypointArrival`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..conditions.base import BaseCondition
from ..conditions.route_progress import RouteProgressCondition
from ..waypoint_arrival import WaypointArrival

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["HOLD_MARGIN_M", "PaceMark", "RoutePacer"]

#: How far short (m) of a waypoint whose condition cannot be predicted the
#: entity stops to wait for it.
HOLD_MARGIN_M: float = 2.0
#: The deceleration (m/s²) the speed cap towards a hold brakes at.
_HOLD_DECELERATION_MPS2: float = 1.5
#: How much of the way to a new target speed the command moves each tick.
_SMOOTHING: float = 0.3
#: Below this progress rate (m/s) a measured entity is not moving forward,
#: and when it will have come far enough cannot be told.
_PREDICTABLE_RATE_MPS: float = 0.5
#: How much of a new progress rate reading the estimate takes each tick.
_RATE_SMOOTHING: float = 0.1
#: The shortest time to go (s) a target speed is worked out over: a waypoint
#: due now or past is caught up with at the limit.
_MIN_TIME_TO_GO_S: float = 0.5


class PaceMark:
    """A waypoint on the planned path: where, and when it is due.

    Args:
        vertex: Its index among the route's vertices.
        distance: How far along the planned path it is (m).
        time: The scenario time it is due at, or ``None``.
        gate: The condition it departs on, or ``None``.
    """

    vertex: int
    distance: float
    time: Optional[float]
    gate: Optional[BaseCondition]
    opened_at: Optional[float]
    reached_at: Optional[float]
    recorded: bool

    def __init__(
        self,
        vertex: int,
        distance: float,
        time: Optional[float] = None,
        gate: Optional[BaseCondition] = None,
    ) -> None:
        self.vertex = vertex
        self.distance = distance
        self.time = time
        self.gate = gate
        self.opened_at = None
        self.reached_at = None
        self.recorded = False

    @property
    def timed(self) -> bool:
        """Whether the waypoint has a time or a condition to meet."""
        return self.time is not None or self.gate is not None

    @property
    def passed(self) -> bool:
        """Whether the entity has reached it and may go on."""
        return self.reached_at is not None and (
            self.gate is None or self.opened_at is not None
        )


class RoutePacer:
    """The waypoints of one run of a route, and the speed that meets them.

    Args:
        entity: ``role_name`` of the entity routed, for the log.
        mode: The routing mode's value, for the log.
        marks: The waypoints, in the order the path passes them.
        tolerance: How near (m) a waypoint counts as reached.
        speed: The entity's speed when the run starts (m/s): where the
            smoothed command starts from.
    """

    _entity: str
    _mode: str
    _marks: list[PaceMark]
    _tolerance: float
    _command: float
    _reached: int
    _arrivals: list[WaypointArrival]
    _rate: float
    _rate_progress: Optional[float]
    _rate_time: float

    def __init__(
        self,
        entity: str,
        mode: str,
        marks: list[PaceMark],
        tolerance: float,
        speed: float = 0.0,
    ) -> None:
        self._entity = entity
        self._mode = mode
        self._marks = marks
        self._tolerance = tolerance
        self._command = speed
        self._reached = 0
        self._arrivals = []
        self._rate = 0.0
        self._rate_progress = None
        self._rate_time = 0.0

    @property
    def paced(self) -> bool:
        """Whether any waypoint has a time or a condition to meet."""
        for mark in self._marks:
            if mark.timed:
                return True
        return False

    @property
    def marks(self) -> list[PaceMark]:
        """The waypoints, in path order."""
        return self._marks

    def note_open(self, vertex: int, elapsed: float) -> None:
        """Vertex *vertex*'s condition held at *elapsed*, first time only."""
        for mark in self._marks:
            if mark.vertex == vertex and mark.opened_at is None:
                mark.opened_at = elapsed

    def observe(self, here: float, elapsed: float) -> None:
        """The entity is *here* (m along the path) at *elapsed*: note the
        waypoints it has reached, and the arrivals there are to log."""
        while (
            self._reached < len(self._marks)
            and here >= self._marks[self._reached].distance - self._tolerance
        ):
            self._marks[self._reached].reached_at = elapsed
            self._reached += 1
        for mark in self._marks[: self._reached]:
            if mark.recorded:
                continue
            reached = mark.reached_at
            assert reached is not None
            target: Optional[float] = None
            kind = "time"
            if mark.time is not None:
                target = mark.time
            elif mark.gate is not None:
                kind = "condition"
                target = mark.opened_at
            else:
                mark.recorded = True
                continue
            if target is None:
                continue  # its condition has not held yet
            due: float = target
            mark.recorded = True
            self._arrivals.append(
                WaypointArrival(
                    entity=self._entity,
                    vertex=mark.vertex,
                    kind=kind,
                    mode=self._mode,
                    target_t=due,
                    actual_t=reached,
                )
            )

    def take_arrivals(self) -> list[WaypointArrival]:
        """The arrivals noted since the last call."""
        arrivals = self._arrivals
        self._arrivals = []
        return arrivals

    def speed(
        self,
        world: "carla.World",
        here: float,
        elapsed: float,
        cruise: float,
        limit: float,
    ) -> float:
        """The speed (m/s) to command now, the entity being *here*.

        *cruise* is the speed where nothing ahead is due, *limit* the most
        it may go at.  Checks the departure condition of the next waypoint
        not yet passed (only that one, to keep a tick cheap).
        """
        target = cruise
        pending = self._pending()
        if pending is not None:
            mark: PaceMark = pending
            target = self._towards(world, mark, here, elapsed, cruise, limit)
        target = min(max(target, 0.0), limit)
        self._command += _SMOOTHING * (target - self._command)
        if target <= 0.0 and self._command < 0.1:
            self._command = 0.0
        return self._command

    # ------------------------------------------------------------------

    def _pending(self) -> Optional[PaceMark]:
        """The first waypoint with a time or a condition still to meet."""
        for mark in self._marks:
            if mark.passed or not mark.timed:
                continue
            if mark.time is not None and mark.reached_at is not None:
                continue
            return mark
        return None

    def _towards(
        self,
        world: "carla.World",
        mark: PaceMark,
        here: float,
        elapsed: float,
        cruise: float,
        limit: float,
    ) -> float:
        """The speed that meets *mark* from *here*."""
        remaining = max(0.0, mark.distance - here)
        due = mark.time
        if due is not None:
            to_go = due - elapsed
            if to_go < _MIN_TIME_TO_GO_S:
                return limit  # late, or due now: catch up
            return remaining / to_go
        condition = mark.gate
        assert condition is not None
        if mark.opened_at is None and condition.check(world, elapsed) is not None:
            mark.opened_at = elapsed
        if mark.opened_at is not None:
            return cruise
        if mark.reached_at is not None:
            return 0.0  # there, and it has not held: wait
        eta = self._eta(condition, elapsed)
        if eta is None:
            # Not predictable: at cruise, braking to a stop short of it.
            room = max(0.0, remaining - HOLD_MARGIN_M)
            return min(cruise, math.sqrt(2.0 * _HOLD_DECELERATION_MPS2 * room))
        arrive: float = eta
        speed = remaining / max(arrive - elapsed, _MIN_TIME_TO_GO_S)
        return min(speed, math.sqrt(2.0 * _HOLD_DECELERATION_MPS2 * remaining))

    def _eta(self, condition: BaseCondition, elapsed: float) -> Optional[float]:
        """When a rising ``route_progress`` *condition* will hold, from the
        measured entity's progress rate; ``None`` when it cannot be told."""
        gate = _progress_gate(condition)
        if gate is None:
            return None
        progress_gate: RouteProgressCondition = gate
        progress = progress_gate.progress
        threshold = progress_gate.target
        if progress is None or threshold is None:
            return None
        now: float = progress
        if self._rate_progress is not None and elapsed > self._rate_time:
            before: float = self._rate_progress
            reading = (now - before) / (elapsed - self._rate_time)
            self._rate += _RATE_SMOOTHING * (reading - self._rate)
        self._rate_progress = now
        self._rate_time = elapsed
        if self._rate < _PREDICTABLE_RATE_MPS:
            return None
        far: float = threshold
        return elapsed + max(0.0, far - now) / self._rate


def _progress_gate(condition: BaseCondition) -> Optional[RouteProgressCondition]:
    """*condition* as a rising ``route_progress``, or ``None``."""
    if TYPE_CHECKING:
        # Codon types an isinstance() narrowing of a class hierarchy by the
        # static type; the static check needs no prediction.
        none: Optional[RouteProgressCondition] = None
        return none
    else:
        if isinstance(condition, RouteProgressCondition) and condition.rises:
            return condition
        return None
