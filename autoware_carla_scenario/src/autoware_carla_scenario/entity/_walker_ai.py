"""Walk a routed pedestrian with CARLA's own walker controller (#521).

A pedestrian a ``RoutingAction`` sends along its route in ``driver`` mode is
walked by CARLA's ``controller.ai.walker`` where the map has a pedestrian
navigation mesh: the controller keeps to pavements, crosses at crossings
(waiting at the kerb as CARLA's walkers do) and walks round what stands in
its way, which a walker steered straight at its path does not.  It is given
the route a few targets at a time (:data:`TARGET_SPACING_M` apart along the
walk, the next once it is :data:`REACHED_M` past the last), and its top speed
is the walk's pace (0 holds it).

A map without a navigation mesh -- ``get_random_location_from_navigation()``
finds nothing -- is said once, and its walkers are stepped along the route as
before (:func:`~.pedestrian_entity.steer_walker`).
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["REACHED_M", "TARGET_SPACING_M", "drive", "has_navmesh", "release"]

#: Metres between the targets a walker is sent to along its walk.
TARGET_SPACING_M: float = 10.0
#: How near (m, along the walk) a target counts as reached.
REACHED_M: float = 2.0
#: How many draws decide a map has no navigation mesh (a draw may miss on
#: one that has).
_NAVMESH_PROBES = 20

#: How near (m) its walk a walker has to stand for its controller to start.
_START_NEAR_M = 3.0
#: A walker this far down (m) is out of the world, not yet brought in.
_HIDDEN_BELOW_M = -100.0

#: Whether each episode's map has a navigation mesh, by world id.
_NAVMESH: dict[int, bool] = {}


class _Driven:
    """A walker on its controller: where it is sent, and how fast."""

    def __init__(self, controller: Any, targets: list[tuple[float, Any]]) -> None:
        self.controller = controller
        self.targets = targets
        self.index = 0
        self.speed = -1.0


#: Each driven walker, by actor id.
_DRIVEN: dict[int, _Driven] = {}


def has_navmesh(world: Any) -> bool:
    """Whether *world*'s map has a pedestrian navigation mesh (asked once per
    episode; a map without one is said once)."""
    key = int(world.id) if hasattr(world, "id") else 0
    if key not in _NAVMESH:
        found = False
        for _ in range(_NAVMESH_PROBES):
            if world.get_random_location_from_navigation() is not None:
                found = True
                break
        _NAVMESH[key] = found
        if not found:
            logger.warning(
                "The map has no pedestrian navigation mesh: routed walkers are "
                "stepped along their routes, not walked by CARLA's walker "
                "controller (build one when importing the map)"
            )
    return _NAVMESH[key]


def drive(walker: Any, path: Any, here: float, speed: float) -> bool:
    """One tick of *walker* (``here`` m along *path*, a ResolvedTrajectory)
    on CARLA's walker controller at *speed* m/s; ``False`` when the map has
    no navigation mesh, for the caller to step it instead."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    world = walker.get_world()
    if not has_navmesh(world):
        return False
    driven = _DRIVEN.get(int(walker.id))
    if driven is None:
        # Not before the walker stands where its walk is: one just brought in
        # (set where it appears) still reads its old place until the world
        # ticks, and a controller started then registers it there -- under
        # the map, at its spawn -- and walks it from wherever that is.
        location = walker.get_location()
        on = path.at_distance(here)
        if (
            location.z < _HIDDEN_BELOW_M
            or math.hypot(location.x - on.x, location.y - on.y) > _START_NEAR_M
        ):
            return True
        targets: list[tuple[float, Any]] = []
        along = TARGET_SPACING_M
        while along < path.length:
            sample = path.at_distance(along)
            targets.append((along, carla.Location(x=sample.x, y=sample.y, z=sample.z)))
            along += TARGET_SPACING_M
        end = path.at_distance(path.length)
        targets.append((path.length, carla.Location(x=end.x, y=end.y, z=end.z)))
        blueprint = world.get_blueprint_library().find("controller.ai.walker")
        controller = world.spawn_actor(blueprint, carla.Transform(), attach_to=walker)
        controller.start()
        driven = _Driven(controller, targets)
        _DRIVEN[int(walker.id)] = driven
        controller.go_to_location(targets[0][1])
        logger.debug(
            "walker %d: on CARLA's controller from (%.1f, %.1f), %d targets to "
            "(%.1f, %.1f)",
            int(walker.id),
            location.x,
            location.y,
            len(targets),
            targets[-1][1].x,
            targets[-1][1].y,
        )
    moved = False
    while (
        driven.index < len(driven.targets) - 1
        and here >= driven.targets[driven.index][0] - REACHED_M
    ):
        driven.index += 1
        moved = True
    if moved:
        target = driven.targets[driven.index][1]
        driven.controller.go_to_location(target)
        logger.debug(
            "walker %d: %.1f m along its walk, on to (%.1f, %.1f)",
            int(walker.id),
            here,
            target.x,
            target.y,
        )
    wanted = max(speed, 0.0)
    if abs(wanted - driven.speed) > 0.05:
        driven.controller.set_max_speed(wanted)
        driven.speed = wanted
    return True


def release(walker_id: int) -> None:
    """Stop and destroy *walker_id*'s controller, if it has one."""
    driven = _DRIVEN.pop(int(walker_id), None)
    if driven is None:
        return
    for call in (driven.controller.stop, driven.controller.destroy):
        try:
            call()
        except RuntimeError:
            pass
