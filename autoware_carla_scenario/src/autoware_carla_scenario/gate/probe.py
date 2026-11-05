"""Sample the ego in the route's frame on every tick, for the gate.

A logical scenario with a gate (``docs/gate.md``) registers a
:class:`GateProbe` after every tick.  Each call projects the ego onto the
scenario's route -- as a ``route_progress`` condition does -- and keeps a row:
the simulated time since the first row, the route s, the signed lateral
offset from the route's reference line (left positive) and the speed.  The
rows go into the run's gate record (:mod:`.record`), so the trajectory
criteria compare runs on different maps without either map.

CARLA is only touched through the world the probe is called with, and the
modules that reach it are imported when it is first called.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union

from ..constants import EGO_ROLE_NAME
from ..entity_role import EntityRole

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["GateProbe", "signed_lateral"]


def signed_lateral(
    reference: tuple[float, float, float], x: float, y: float, distance: float
) -> float:
    """*distance* from the route, signed by the side of it ``(x, y)`` is on.

    *reference* is the route's reference point ``(x, y, heading)`` the point
    projected to (map frame, heading anticlockwise from +x); left of it, the
    way the route runs, is positive.
    """
    rx, ry, heading = reference
    side = math.cos(heading) * (y - ry) - math.sin(heading) * (x - rx)
    if side == 0.0:
        return 0.0
    return math.copysign(distance, side)


class GateProbe:
    """The ego's rows in the route's frame, one per tick it was found on it.

    Args:
        role_name: The entity sampled (the ego).
    """

    def __init__(self, role_name: Union[EntityRole, str] = EGO_ROLE_NAME) -> None:
        self._role_name = role_name
        self._t0: Optional[float] = None
        self._last_s: Optional[float] = None
        self.rows: list[tuple[float, float, float, float]] = []

    def reset(self) -> None:
        """Forget every row."""
        self._t0 = None
        self._last_s = None
        self.rows = []

    def __call__(self, world: "carla.World") -> None:
        """Keep a row for the ego where it is now; none when it is not on the
        map (not spawned, parked under it) or the scenario has no route."""
        from ..conditions.base import find_actor_by_role_name  # noqa: PLC0415
        from ..conditions.route_progress import (  # noqa: PLC0415
            HIDDEN_BELOW_ROUTE_M,
            route_below,
        )
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..route.active import scenario_route, scenario_route_frame  # noqa: PLC0415

        if scenario_route() is None:
            return
        actor = find_actor_by_role_name(world, self._role_name)
        if actor is None:
            return
        location = actor.get_transform().location
        offset_x, offset_y = MapManager.get_instance().mgrs_offset
        x, y = location.x + offset_x, -location.y + offset_y
        frame = scenario_route_frame()
        s, distance = frame.project(x, y, self._last_s)
        if not math.isfinite(distance):
            return
        if route_below(s, location.z) >= HIDDEN_BELOW_ROUTE_M:
            return
        self._last_s = s
        lateral = signed_lateral(frame.point(s), x, y, distance)
        velocity = actor.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)
        now = float(world.get_snapshot().timestamp.elapsed_seconds)
        if self._t0 is None:
            self._t0 = now
        self.rows.append((now - self._t0, s, lateral, speed))
