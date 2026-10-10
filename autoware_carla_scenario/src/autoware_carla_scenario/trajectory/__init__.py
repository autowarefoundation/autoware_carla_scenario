"""Trajectories for entities to follow (OpenSCENARIO `FollowTrajectoryAction`).

* :mod:`.model` -- the OpenSCENARIO-style trajectory (polyline, vertices,
  timing, following mode) and its interpolation;
* :mod:`.resolve` -- placing one in CARLA world coordinates;
* :mod:`.authoring` -- building one from what a scenario document states.

Reading a T4 driving scene into trajectories is the separate
``autoware_carla_scenario_t4`` package.

The action that follows a trajectory is
:class:`~autoware_carla_scenario.actions.FollowTrajectoryAction`.
"""

from .model import (
    MapPose,
    ReferenceContext,
    ResolvedTrajectory,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryPosition,
    TrajectorySample,
    TrajectoryTiming,
    TrajectoryVertex,
)

__all__ = [
    "MapPose",
    "ReferenceContext",
    "ResolvedTrajectory",
    "Trajectory",
    "TrajectoryFollowingMode",
    "TrajectoryPosition",
    "TrajectorySample",
    "TrajectoryTiming",
    "TrajectoryVertex",
]
