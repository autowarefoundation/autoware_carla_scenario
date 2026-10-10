"""Trajectories for entities to follow, and T4 scenes transcribed into them.

* :mod:`.model` -- the OpenSCENARIO-style trajectory (polyline, vertices,
  timing, following mode) and its interpolation;
* :mod:`.resolve` -- placing one in CARLA world coordinates;
* :mod:`.t4` -- reading a T4 driving scene into trajectories;
* :mod:`.replay` -- spawning a transcribed scene's road users and replaying it
  as a scenario (imported on its own: it needs the scenario machinery).

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
from .t4 import (
    T4_FRAME_RATE_HZ,
    T4Category,
    T4Detection,
    T4ObjectTrack,
    T4SceneTranscription,
    associate_tracks,
    read_t4_scene,
)

__all__ = [
    "MapPose",
    "ReferenceContext",
    "ResolvedTrajectory",
    "T4Category",
    "T4Detection",
    "T4ObjectTrack",
    "T4SceneTranscription",
    "T4_FRAME_RATE_HZ",
    "Trajectory",
    "TrajectoryFollowingMode",
    "TrajectoryPosition",
    "TrajectorySample",
    "TrajectoryTiming",
    "TrajectoryVertex",
    "associate_tracks",
    "read_t4_scene",
]
