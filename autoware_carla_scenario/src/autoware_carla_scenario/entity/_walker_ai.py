"""Walk a routed pedestrian with CARLA's own walker controller (#521).

A pedestrian a ``RoutingAction`` sends along its route in ``driver`` mode is
walked by CARLA's ``controller.ai.walker`` where the map has a pedestrian
navigation mesh: the controller keeps to pavements, crosses at crossings
(waiting at the kerb as CARLA's walkers do) and walks round what stands in
its way, which a walker steered straight at its path does not.  It is given
the route a few targets at a time (:data:`TARGET_SPACING_M` apart along the
walk, the next once it is :data:`REACHED_M` short of the last along the
walk, or that near it), and its top speed is the walk's pace.

**Crossing a road away from a crossing** is a walker's own:
``cross_factor`` (0..1, ``None`` CARLA's default, 0) is the probability that
its routes may use the road anywhere.  CARLA keeps one factor for the world,
and reads it only when a walker's controller starts (``Navigation::AddWalker``,
on the client, which draws the walker's road filter from it then): so the
factor is set to the walker's own just before its controller starts, and
back to CARLA's default straight after -- deterministic for 0 and 1, the
draw CARLA makes for a share in between.  Everything runs on the tick loop's
thread, so no two walkers' factors meet.

**Holding.**  CARLA gives a walker that has stood 4 s a random new target
(``Navigation::UpdateCrowd``, "unblocking").  A walker held by its route's
timing is therefore taken off its controller while it stands, and put back
on it -- with its own cross factor -- when it walks on; and every walker's
current target is sent again every :data:`RETARGET_S` seconds, so an unblock
elsewhere costs it at most that.

A map without a navigation mesh -- ``get_random_location_from_navigation()``
finds nothing -- is said once, and its walkers are stepped along the route as
before (:func:`~.pedestrian_entity.steer_walker`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "CARLA_CROSS_FACTOR",
    "REACHED_M",
    "RETARGET_S",
    "TARGET_SPACING_M",
    "drive",
    "has_navmesh",
    "release",
]

#: CARLA's own pedestrian cross factor (``Navigation::_probability_crossing``).
CARLA_CROSS_FACTOR: float = 0.0
#: Metres between the targets a walker is sent to along its walk.
TARGET_SPACING_M: float = 10.0
#: How near (m) a target counts as reached: that far short of it along the
#: walk, or that far from it in a straight line.
REACHED_M: float = 2.0
#: How often (s of driving) a walker's current target is sent again.
RETARGET_S: float = 2.0
#: How many draws decide a map has no navigation mesh (a draw may miss on
#: one that has).
_NAVMESH_PROBES = 20
#: How near (m) its walk a walker has to stand for its controller to start.
_START_NEAR_M = 3.0
#: A walker this far down (m) is out of the world, not yet brought in.
_HIDDEN_BELOW_M = -100.0
#: Below this (m/s) a walker is held, off its controller.
_HOLD_MPS = 0.05
#: The tick, for the retargeting clock where the world does not say.
_DEFAULT_DT_S = 0.05

#: Whether each episode's map has a navigation mesh, by world id.
_NAVMESH: dict[int, bool] = {}


class _Driven:
    """A walker on its controller: where it is sent, and how fast."""

    def __init__(
        self,
        controller: Any,
        targets: list[tuple[float, Any]],
        cross_factor: Optional[float],
    ) -> None:
        self.controller = controller
        self.targets = targets
        self.cross_factor = cross_factor
        self.index = 0
        self.speed = -1.0
        self.running = False
        self.since_sent = 0.0


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


def _start(world: Any, driven: _Driven) -> None:
    """Put the walker on its controller, with its own cross factor."""
    factor = driven.cross_factor
    if factor is not None:
        # frand() <= factor: -1 is never, 1 is always.
        world.set_pedestrians_cross_factor(-1.0 if factor <= 0.0 else float(factor))
    try:
        driven.controller.start()
    finally:
        if factor is not None:
            world.set_pedestrians_cross_factor(CARLA_CROSS_FACTOR)
    driven.running = True
    driven.speed = -1.0
    driven.since_sent = RETARGET_S  # sent at once


def drive(
    walker: Any,
    path: Any,
    here: float,
    speed: float,
    location: Any = None,
    cross_factor: Optional[float] = None,
) -> bool:
    """One tick of *walker* (``here`` m along *path*, a ResolvedTrajectory, at
    *location*) on CARLA's walker controller at *speed* m/s, crossing roads
    away from crossings with probability *cross_factor*; ``False`` when the
    map has no navigation mesh, for the caller to step it instead."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    world = walker.get_world()
    if not has_navmesh(world):
        return False
    if location is None:
        location = walker.get_location()
    driven = _DRIVEN.get(int(walker.id))
    if driven is None:
        # Not before the walker stands where its walk is: one just brought in
        # (set where it appears) still reads its old place until the world
        # ticks, and a controller started then registers it there -- under
        # the map, at its spawn -- and walks it from wherever that is.
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
        driven = _Driven(controller, targets, cross_factor)
        _DRIVEN[int(walker.id)] = driven
        logger.debug(
            "walker %d: on CARLA's controller (cross factor %s), %d targets to "
            "(%.1f, %.1f)",
            int(walker.id),
            "default" if cross_factor is None else "%.2f" % cross_factor,
            len(targets),
            targets[-1][1].x,
            targets[-1][1].y,
        )
    while driven.index < len(driven.targets) - 1:
        along, target = driven.targets[driven.index]
        near = math.hypot(location.x - target.x, location.y - target.y) <= REACHED_M
        if here < along - REACHED_M and not near:
            break
        driven.index += 1
        driven.since_sent = RETARGET_S
    wanted = max(speed, 0.0)
    if wanted < _HOLD_MPS:
        # Held: off the controller, which would send a walker standing 4 s to
        # a random place, and standing.
        if driven.running:
            _stop(driven)
            walker.apply_control(carla.WalkerControl(speed=0.0))
        return True
    if not driven.running:
        _start(world, driven)
    driven.since_sent += _tick_seconds(world)
    if driven.since_sent >= RETARGET_S:
        driven.controller.go_to_location(driven.targets[driven.index][1])
        driven.since_sent = 0.0
    if abs(wanted - driven.speed) > 0.05:
        driven.controller.set_max_speed(wanted)
        driven.speed = wanted
    return True


def release(walker_id: int) -> None:
    """Stop and destroy *walker_id*'s controller, if it has one."""
    driven = _DRIVEN.pop(int(walker_id), None)
    if driven is None:
        return
    if driven.running:
        _stop(driven)
    try:
        driven.controller.destroy()
    except RuntimeError:
        pass


def _stop(driven: _Driven) -> None:
    """Take the walker off its controller."""
    driven.running = False
    try:
        driven.controller.stop()
    except RuntimeError:
        pass


def _tick_seconds(world: Any) -> float:
    try:
        return float(world.get_snapshot().timestamp.delta_seconds)
    except (AttributeError, RuntimeError):
        return _DEFAULT_DT_S
