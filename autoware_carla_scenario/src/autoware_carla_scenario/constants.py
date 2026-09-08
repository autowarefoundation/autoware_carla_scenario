"""Package-wide constants for autoware_carla_scenario."""

from .entity_role import EntityRole

# CARLA role_name attribute value assigned to the ego vehicle actor.
EGO_ROLE_NAME: EntityRole = EntityRole.ego()

# Default port for the CARLA TrafficManager.
# The CARLA default (8000) often conflicts with other services (e.g. VS Code),
# so we use a different port to avoid ``RuntimeError: std::exception``.
# This value is used as a fallback when no Hydra config is available.
DEFAULT_TM_PORT: int = 8100

# When a forced lane change counts as completed.
#
# ``force_lane_change`` returns as soon as the command is queued, so the action
# watches the vehicle afterwards: the manoeuvre is over once the vehicle is on a
# different lane *and* has settled onto it.  Straddling the boundary is not
# finishing, which is why a lane id change alone is not enough -- a reaction
# triggered on "the cut-in completed" would otherwise fire while the car is
# still diagonal.
LANE_CHANGE_CENTER_TOLERANCE_M: float = 0.5
LANE_CHANGE_HEADING_TOLERANCE_DEG: float = 10.0

# There is deliberately no action-level timeout: a lane change that never
# happens stays in runningState and never reaches completeState, so a condition
# waiting on completion cannot be satisfied by a manoeuvre that did not occur.
# Ending such a run is the scenario timeout's job, which is already a FAIL
# condition.
