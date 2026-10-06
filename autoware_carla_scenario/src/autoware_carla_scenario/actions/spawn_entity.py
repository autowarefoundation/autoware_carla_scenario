"""Spawn-entity action: bring a scenario entity into the world mid-run."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union

from ..conditions import BaseCondition
from ..conditions.base import find_actor_in_list
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["SPAWN_SIDES", "SpawnEntityAction", "SpawnsEntities"]

#: The lanes a relative spawn may be placed in, beside the reference's.
SPAWN_SIDES = ("same", "left", "right", "opposite")


class SpawnsEntities(Protocol):
    """What a scenario offers for an entity to be spawned while it runs."""

    def spawn_entity(
        self, role_name: str, world: "carla.World", pose: Optional[Lanelet2Pose]
    ) -> Any:
        """Spawn the entity *role_name* names, at *pose* or where it was authored."""
        ...


class SpawnEntityAction(BaseAction):
    """Spawn an entity the scenario declared but did not start with.

    Every entity used to enter the world in ``setup()``, so a road user that
    only appears partway through -- a car that turns in from a side road, a
    pedestrian that steps out once the ego is close -- had to exist, and be
    driven, from the first tick.  This brings it in when its trigger fires.

    Where it appears:

    * with no *anchor*, where the entity was authored to spawn -- the
      spawn a sweep resolved, searched or derived from the pick included;
    * with *anchor*, *gap_m* metres along the road from where that
      entity is **now** (positive ahead), in the lane *side* names, keeping
      the authored lateral offset and heading.  This is what keeps a relative
      position meaningful at the moment it matters: the reference has moved
      since the run began, and an authored lanelet would no longer be beside
      it.

    The spawning itself is the scenario's (:class:`SpawnsEntities`): it knows
    how the entity is built, snapped to the road and registered.

    Args:
        entity_name: ``role_name`` of the entity to spawn.
        scenario: The running scenario.
        anchor: ``role_name`` of the entity to place it relative to, or
            ``None`` for its authored spawn.
        gap_m: Distance along the road from *anchor*.
        side: ``same``, ``left``, ``right`` or ``opposite`` lane.
        condition: Trigger condition.
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.

    Raises:
        ValueError: If *side* is not one of :data:`SPAWN_SIDES`.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        scenario: Any,
        anchor: Optional[Union[EntityRole, str]] = None,
        gap_m: float = 0.0,
        side: str = "same",
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "spawn_entity",
        once: bool = True,
    ) -> None:
        if side not in SPAWN_SIDES:
            raise ValueError(f"side must be one of {SPAWN_SIDES}, got {side!r}")
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = entity_name
        self._scenario = scenario
        self._anchor = anchor or None
        self._gap_m = gap_m
        self._side = side

    def pose(self, world: "carla.World") -> Optional[Lanelet2Pose]:
        """Where to spawn: ``None`` for the authored spawn.

        Raises:
            LookupError: If *anchor* is not in the world, or the road
                does not reach *gap_m* from it in that lane.
        """
        if self._anchor is None:
            return None
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.transform import to_lanelet2  # noqa: PLC0415
        from ..sweeper.bindings import route_offset_pose  # noqa: PLC0415

        reference = find_actor_in_list(world.get_actors(), self._anchor)
        if reference is None:
            raise LookupError(f"'{self._anchor}' is not in the world")
        location = reference.get_location()
        here = to_lanelet2(
            CarlaWorldPose(x=location.x, y=location.y, z=location.z, yaw=0.0)
        )
        mm = MapManager.get_instance()
        try:
            lanelet_id, s = route_offset_pose(
                here.lanelet_id,
                mm.lanelet_map,
                mm.routing_graph,
                distance=here.s + self._gap_m,
                side=self._side,
            )
        except ValueError as error:
            raise LookupError(str(error)) from error
        return Lanelet2Pose(lanelet_id=lanelet_id, s=s)

    def execute(self, world: "carla.World") -> None:
        """Spawn the entity."""
        try:
            pose = self.pose(world)
        except LookupError as error:
            logger.warning(
                "SpawnEntityAction: '%s' not spawned: %s", self._entity_name, error
            )
            return
        self._scenario.spawn_entity(str(self._entity_name), world, pose)
