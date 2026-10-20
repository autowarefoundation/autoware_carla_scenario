"""adversity – TeraSim's adversarial behaviour generation, ported (#385).

The layers, bottom-up:

* :mod:`.geometry`, :mod:`.collision`, :mod:`.prediction`, :mod:`.idm`: the
  simulator-agnostic math -- footprints, trajectory conflicts, kinematic
  prediction and the car-following model the triggers compare against.
  Angles wrap with :func:`..kinematics.normalize_angle_rad`.
* :mod:`.location`: where an agent is on the Lanelet2 map (intersection,
  highway, roundabout), for each adversity's location gate.
* :mod:`.observation`: what an agent sees -- leader, follower, the lanes
  beside, the next signal, the road near a pedestrian -- from a snapshot of
  the agents (:mod:`.snapshot` takes it of a CARLA world).
* :mod:`.triggers`: when each phase-1 adversity fires, behind its
  location gate (:func:`.triggers.adversity_command`).
* :mod:`.command`: what a trigger answers with -- accelerate, change lanes,
  follow a path -- and the timed path it drives (``actions`` drives it).

Everything here works in the Lanelet2 map frame: ENU, right-handed, yaw in
radians counter-clockwise from +x, positions at the agent's centre.  TeraSim
works in SUMO's frame (degrees clockwise from north, positions at the front
bumper); the port converts once, where an actor's state is read, and nowhere
else.
"""

from .collision import collision_time, segments_intersect, trajectories_intersect
from .command import AdversarialCommand, CommandKind, plan_command, predict_command
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
from .location import AdversityLocation, location_matches, location_of
from .prediction import (
    interpolate_trajectory,
    lane_change_longitudinal_speed,
    predict_distance_velocity,
)
from .triggers import AdversityKind, TriggerThresholds, adversity_command

__all__ = [
    "DEFAULT_IDM",
    "AdversarialCommand",
    "AdversityKind",
    "AdversityLocation",
    "AgentFootprint",
    "CommandKind",
    "IdmParameters",
    "PredictedState",
    "TriggerThresholds",
    "adversity_command",
    "car_following_acceleration",
    "circle_centers",
    "circle_offsets",
    "circle_radius",
    "collision_time",
    "free_flow_acceleration",
    "interpolate_trajectory",
    "lane_change_longitudinal_speed",
    "location_matches",
    "location_of",
    "plan_command",
    "predict_command",
    "predict_distance_velocity",
    "segments_intersect",
    "trajectories_intersect",
]
