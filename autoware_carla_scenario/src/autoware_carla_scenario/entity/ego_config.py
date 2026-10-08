"""Configuration of the ego vehicle.

Here rather than in :mod:`..scenario_base`, which re-exports it, because
:mod:`.ego` annotates with it and :mod:`..scenario_base` imports :mod:`.ego`:
a standalone binary (docs/standalone.md) runs every import, those for
annotations included, and Codon compiles no import cycle.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..constants import EGO_ROLE_NAME
from ..coordinate import Lanelet2Pose
from ._spawn import SpawnLocation
from .vehicle_entity import VehicleEntityConfig

__all__ = ["EgoConfig"]


class EgoConfig(VehicleEntityConfig):
    """Configuration for the ego vehicle.

    Inherits all fields from :class:`VehicleEntityConfig` (``vehicle_type``,
    ``initial_speed_kmh``, etc.).  The ``role_name`` is automatically set to
    :data:`~autoware_carla_scenario.constants.EGO_ROLE_NAME`.

    Both ends of the ego's run live here: *spawn_location* is where it starts
    and *goal_pose* is where it is being sent.  Every scenario needs a goal --
    an ego that plans its own route will not move without one, and one that is
    driven for it still had a destination the run was aiming at -- but the
    config is not always where it comes from: a scenario may derive it in
    ``setup()`` from what it knows.  ``None`` therefore means "not named yet",
    and :meth:`BaseScenario.register_route_to_goal` is where an ego that still
    has none is refused.
    """

    #: Lanelet2 pose the ego is routed to, or ``None`` until something names it.
    goal_pose: Optional[Lanelet2Pose]

    #: Lanelet2 poses the route must pass through, in order.  Empty leaves the
    #: way to the goal to whatever plans the route, which is right when any way
    #: there will do and wrong when the run *is* a particular drive.
    waypoint_poses: List[Lanelet2Pose]

    def __init__(
        self,
        spawn_location: SpawnLocation,
        vehicle_type: str = "vehicle.mini.cooper",
        initial_speed_kmh: float = 0.0,
        spawn_retry_max_count: int = 0,
        spawn_retry_t_step: float = 0.1,
        spawn_retry_z_step: float = 0.5,
        *,
        goal_pose: Optional[Lanelet2Pose] = None,
        waypoint_poses: Optional[Sequence[Lanelet2Pose]] = None,
    ) -> None:
        super().__init__(
            role_name=EGO_ROLE_NAME,
            spawn_location=spawn_location,
            vehicle_type=vehicle_type,
            initial_speed_kmh=initial_speed_kmh,
            spawn_retry_max_count=spawn_retry_max_count,
            spawn_retry_t_step=spawn_retry_t_step,
            spawn_retry_z_step=spawn_retry_z_step,
        )
        self.goal_pose = goal_pose
        self.waypoint_poses = list(waypoint_poses or ())
