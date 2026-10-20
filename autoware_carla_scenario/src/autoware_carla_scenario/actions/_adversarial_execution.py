"""Drive a planned adversarial command, then hand the vehicle back (#391).

What the adversarial actions share once a trigger has fired and its command
is planned (:func:`~..adversity.command.plan_command`): the path is driven
by a :class:`~.follow_trajectory.FollowTrajectoryAction` on the path's own
clock -- ``FOLLOW`` (the vehicle's own motion, by default) or ``POSITION``
(TeraSim's ``moveToXY`` teleport) -- which takes the vehicle back from its
traffic backend first.  When the path is done a vehicle is handed back to
the backend (:meth:`~..traffic.driven.BackendDriven.resume_traffic`) still
moving (``stop_at_end=False``) and rejoins traffic from there, as TeraSim's
controller turns SUMO's checks back on; a pedestrian is left standing where
the path ended.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..action_state import ActionState
from ..adversity.geometry import PredictedState
from ..conditions.base import BaseCondition
from ..conditions.trajectory_time import TrajectoryTimeCondition
from ..entity.registry import find_vehicle_entity
from ..trajectory.model import (
    MapPose,
    ReferenceContext,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTiming,
    TrajectoryVertex,
)
from .base import TickTiming
from .follow_trajectory import FollowTrajectoryAction

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversarialExecution", "path_trajectory"]

logger = logging.getLogger(__name__)


def path_trajectory(name: str, path: list[PredictedState]) -> Trajectory:
    """*path* as a timed trajectory in the map frame, its times from 0."""
    if len(path) < 2:
        raise ValueError("a path to drive needs at least two states")
    start = path[0].t
    vertices: list[TrajectoryVertex] = []
    for p in path:
        # A typed local: the vertex takes a BaseCondition, which Codon does not
        # upcast a TrajectoryTimeCondition to inside the call.
        at: BaseCondition = TrajectoryTimeCondition(p.t - start)
        vertices.append(TrajectoryVertex(MapPose(p.x, p.y, p.yaw), at))
    return Trajectory(name, vertices)


class AdversarialExecution:
    """One planned command, driven on *entity_name* from its first tick.

    Args:
        entity_name: ``role_name`` of the vehicle or pedestrian.
        path: The planned path, in the Lanelet2 map frame.
        following_mode: ``FOLLOW`` (default) or ``POSITION``.
        label: For the logs and the follow action's label.
    """

    entity_name: str
    _follow: FollowTrajectoryAction
    _handed_back: bool

    def __init__(
        self,
        entity_name: str,
        path: list[PredictedState],
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
        label: str = "adversity",
    ) -> None:
        self.entity_name = str(entity_name)
        self._follow = FollowTrajectoryAction(
            self.entity_name,
            path_trajectory(label, path),
            time_reference=TrajectoryTiming(ReferenceContext.RELATIVE),
            following_mode=following_mode,
            timing=TickTiming.PRE_TICK,
            label=label,
            stop_at_end=False,
        )
        self._handed_back = False

    @property
    def done(self) -> bool:
        """Whether the path is driven and the vehicle handed back."""
        return self._handed_back

    def tick(self, world: carla.World, elapsed: float) -> None:
        """Drive one tick; once the path is done, hand the vehicle back."""
        if self._handed_back:
            return
        self._follow.tick(world, elapsed)
        if self._follow.state is not ActionState.COMPLETE:
            return
        self._handed_back = True
        vehicle = find_vehicle_entity(self.entity_name)
        if vehicle is not None:
            vehicle.resume_traffic(world)
            logger.info(
                "%s: '%s' handed back to traffic", self._follow.label, self.entity_name
            )

    def close(self) -> None:
        """Close the follow action's conditions."""
        self._follow.close()
