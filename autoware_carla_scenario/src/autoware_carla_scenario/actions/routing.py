"""Hand an ego that plans its own route the mission it needs.

An ego driven by a full autonomy stack is not told how to steer; it is told
*where to go* and plans the rest itself.  That hand-over belongs to the
scenario's initialization phase, alongside setting the traffic lights and
placing the NPCs -- it describes the state the run starts from, not something
the run does -- so :class:`RoutingAction` is registered with
:meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_init`
rather than on the tick loop.

It could not be a tick-loop action even if that read better: the runner waits
for the ego to report ready *before* the loop starts, and an Autoware ego only
becomes ready once it has localized, routed and engaged.  A goal delivered from
inside the loop would never arrive.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from ..coordinate import (
    CarlaWorldPose,
    GroundProjectionConfig,
    Lanelet2Pose,
    snap_to_carla_road,
    to_map_frame,
    to_opendrive,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

    from ..entity.ego import EgoVehicle

logger = logging.getLogger(__name__)


class RoutingAction(BaseAction):
    """Give the ego a goal to plan a route to, and a pose to localize at.

    Mirrors OpenSCENARIO's routing actions: the scenario states a destination
    and the stack works out the trajectory.  Only an ego that plans for itself
    has a mission to set -- an
    :class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`
    today.  Every other entity drives itself (TrafficManager, an external driver
    policy) and this is a no-op for it, so a scenario can register the action
    unconditionally and let ``ego.entity`` decide whether it means anything.

    The goal is snapped onto the CARLA road surface and converted into
    Autoware's ``map`` frame here rather than by the caller, because that is the
    same treatment the ego spawn gets and doing it anywhere else would put the
    goal in the wrong frame.

    Args:
        ego: The ego entity, or a callable returning it.  A callable is the
            useful form when the entity is swapped in after the scenario is
            constructed, which is how ``ego.entity`` reaches a packaged
            scenario.
        goal: Lanelet2 pose the ego is routed to.
        initial_pose: CARLA world pose the ego is expected to start at, used to
            initialize localization.  ``None`` leaves it to the entity, which
            reads the attached actor instead.
        ground_projection: Settings used to snap the goal to the road surface.
        label: Action label.  Defaults to ``"routing"``.
        condition: Optional condition, evaluated once during initialization.
    """

    def __init__(
        self,
        ego: "EgoVehicle | Callable[[], Optional[EgoVehicle]]",
        goal: Lanelet2Pose,
        *,
        initial_pose: Optional[CarlaWorldPose] = None,
        ground_projection: Optional[GroundProjectionConfig] = None,
        label: str = "routing",
        condition=None,  # noqa: ANN001 - BaseCondition, typed by the base class
    ) -> None:
        super().__init__(
            label=label,
            condition=condition,
            timing=TickTiming.PRE_TICK,
            once=True,
        )
        self._ego = ego
        self._goal = goal
        self._initial_pose = initial_pose
        self._ground_projection = ground_projection or GroundProjectionConfig()

    @property
    def goal(self) -> Lanelet2Pose:
        """The Lanelet2 pose the ego is routed to."""
        return self._goal

    def _resolve_ego(self) -> Optional["EgoVehicle"]:
        """Return the ego entity, calling the getter if one was given."""
        return self._ego() if callable(self._ego) else self._ego

    def execute(self, world: "carla.World") -> None:
        """Snap the goal, convert both poses to the map frame, hand them over."""
        from ..entity.autoware_entity import AutowareEgoEntity  # noqa: PLC0415

        ego = self._resolve_ego()
        if not isinstance(ego, AutowareEgoEntity):
            logger.debug(
                "%s: the ego drives itself (%s); no mission to set.",
                self.label,
                type(ego).__name__,
            )
            return

        goal_snapped = snap_to_carla_road(
            to_opendrive(self._goal),
            world,
            ground_projection=self._ground_projection,
        )
        logger.info(
            "%s: goal lanelet %d s=%.1f -> CARLA (%.1f, %.1f, %.1f)%s",
            self.label,
            self._goal.lanelet_id,
            self._goal.s,
            goal_snapped.x,
            goal_snapped.y,
            goal_snapped.z,
            (
                ""
                if self._initial_pose is None
                else f", starting from ({self._initial_pose.x:.1f}, "
                f"{self._initial_pose.y:.1f}, {self._initial_pose.z:.1f})"
            ),
        )
        ego.set_mission(
            None if self._initial_pose is None else to_map_frame(self._initial_pose),
            to_map_frame(goal_snapped),
        )
