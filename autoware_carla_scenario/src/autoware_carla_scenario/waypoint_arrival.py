"""When an entity reached a waypoint, against when it was meant to (#521).

A module of its own at the package's top, importing nothing, as
:mod:`~autoware_carla_scenario.adversity_event` is: the trajectory recorder,
the runner and the routing action read it without pulling each other in.
"""

from __future__ import annotations

from dataclasses import dataclass
__all__ = ["WaypointArrival"]


@dataclass(frozen=True)
class WaypointArrival:
    """A timed waypoint of a route, as the trajectory log records it.

    Attributes:
        entity: ``role_name`` of the entity routed.
        vertex: The waypoint's index (0-based) among the route's vertices.
        kind: What timed it: ``"time"`` (an arrival time) or ``"condition"``
            (a departure condition; the target is when it first held).
        mode: The routing mode: ``"driver"``, ``"follow"`` or ``"position"``.
        target_t: Scenario time (s) the entity was meant to arrive at.
        actual_t: Scenario time (s) it arrived at.
    """

    entity: str
    vertex: int
    kind: str
    mode: str
    target_t: float
    actual_t: float

    @property
    def error_s(self) -> float:
        """How late (s) it arrived; negative when early."""
        return self.actual_t - self.target_t
