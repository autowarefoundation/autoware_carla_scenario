"""adversity – TeraSim's adversarial behaviour generation, ported (#385).

The layers, bottom-up:

* :mod:`.geometry`, :mod:`.collision`, :mod:`.prediction`, :mod:`.idm`: the
  simulator-agnostic math -- footprints, trajectory conflicts, kinematic
  prediction and the car-following model the triggers compare against.
  Angles wrap with :func:`..kinematics.normalize_angle_rad`.

Everything here works in the Lanelet2 map frame: ENU, right-handed, yaw in
radians counter-clockwise from +x, positions at the agent's centre.  TeraSim
works in SUMO's frame (degrees clockwise from north, positions at the front
bumper); the port converts once, where an actor's state is read, and nowhere
else.
"""

from .collision import collision_time, segments_intersect, trajectories_intersect
from .geometry import (
    AgentFootprint,
    PredictedState,
    circle_centers,
    circle_offsets,
    circle_radius,
)
from .idm import (
    DEFAULT_IDM,
    IdmParameters,
    car_following_acceleration,
    free_flow_acceleration,
)
from .prediction import (
    interpolate_trajectory,
    lane_change_longitudinal_speed,
    predict_distance_velocity,
)

__all__ = [
    "DEFAULT_IDM",
    "AgentFootprint",
    "IdmParameters",
    "PredictedState",
    "car_following_acceleration",
    "circle_centers",
    "circle_offsets",
    "circle_radius",
    "collision_time",
    "free_flow_acceleration",
    "interpolate_trajectory",
    "lane_change_longitudinal_speed",
    "predict_distance_velocity",
    "segments_intersect",
    "trajectories_intersect",
]
