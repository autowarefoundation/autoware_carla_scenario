"""Trajectories for entities to follow (OpenSCENARIO `FollowTrajectoryAction`).

* :mod:`.model` -- the OpenSCENARIO-style trajectory (polyline, vertices,
  timing, following mode) and its interpolation;
* :mod:`.resolve` -- placing one in CARLA world coordinates;
* :mod:`.authoring` -- building one from what a scenario document states.

Reading a recorded scene (a T4 dataset) into trajectories is left to tools
outside the framework, such as ``scene_to_scenario_transpiler``.

The action that follows a trajectory is
:class:`~autoware_carla_scenario.actions.FollowTrajectoryAction`.
"""

from .model import (
    MapPose,
    ReferenceContext,
    RelativeLanePose,
    ResolvedTrajectory,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
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
    "RelativeLanePose",
    "ResolvedTrajectory",
    "RouteCrossingPose",
    "RouteCrosswalkPose",
    "RouteLanePose",
    "RouteOppositePose",
    "RouteRoadsidePose",
    "Trajectory",
    "TrajectoryFollowingMode",
    "TrajectoryPosition",
    "TrajectorySample",
    "TrajectoryTiming",
    "TrajectoryVertex",
]
