"""Package-wide constants for autoware_carla_scenario."""

from .entity_role import EntityRole

# CARLA role_name attribute value assigned to the ego vehicle actor.
EGO_ROLE_NAME: EntityRole = EntityRole.ego()

# CARLA role_name of the vehicle placement.vehicle_bumpers() spawns high above
# the map to measure a blueprint, and the vehicle-ahead measures skip.
BUMPER_PROBE_ROLE_NAME: str = "bumper_probe"

# Default port for the CARLA TrafficManager.
# The CARLA default (8000) often conflicts with other services (e.g. VS Code),
# so we use a different port to avoid ``RuntimeError: std::exception``.
# This value is used as a fallback when no Hydra config is available.
DEFAULT_TM_PORT: int = 8100

# The ``role_name`` values that make an actor the ego of a trajectory log: the
# recorder logs their control inputs, and a reader of the log finds the ego by
# them (t4_scene).
EGO_ROLES = ("ego", "hero")

# The simulation step the runner drives the world at (20 Hz).
#
# Named here because three places have to agree on it: the runner applies it to
# the world, an entity that converts ticks to simulation time reads it, and a
# traffic backend stepping a second simulator matches its step length to it.
# Three private copies that must agree is exactly the coupling this avoids.
FIXED_DELTA_SECONDS: float = 0.05


def tick_multiple(seconds: float, key: str) -> int:
    """The ticks in *seconds*, which must be a positive multiple of the tick.

    Every setting the run can only honour on a tick boundary -- the driver
    policy's step, a LiDAR's sweep, the T4 dataset's and the T4Scene's step --
    goes through this one check, so a value between two ticks is refused when
    the config is read instead of being rounded to the next tick at run time.
    A value within 1e-9 s of a multiple counts as one (0.15 is not exact in
    binary).

    Args:
        seconds: The interval.
        key: The setting's name, which the error names (``driver.policy_timestep_s``).

    Raises:
        ValueError: If *seconds* is not a positive multiple of the tick (and
            OverflowError for an infinite *seconds*, as ``round`` raises).
    """
    ticks = int(round(seconds / FIXED_DELTA_SECONDS))
    if ticks < 1 or abs(ticks * FIXED_DELTA_SECONDS - seconds) > 1e-9:
        raise ValueError(
            f"{key}={seconds} is not a positive multiple of the "
            f"{FIXED_DELTA_SECONDS} s tick"
        )
    return ticks


# Microseconds per second: simulation time as an integer, the clock the
# trajectory follower, the ego observation and the ROS 2 command arbiter keep.
US_PER_S: int = 1_000_000

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

# Background traffic, as every backend's spawn_background agrees on it (#967).
#
# The least speed a background vehicle may be asked to drive at, km/h.  A
# smaller one -- 0, a negative one, NaN or infinity -- is refused where it
# enters (TrafficSourceAction, background_traffic.source.speed_kmh, the
# editor's validator, the backends), not raised to this: ``speed_kmh`` is the
# most a vehicle drives at, so a cap above the one asked for would break that,
# and 0 would be a vehicle that never moves, which is no traffic.
MIN_BACKGROUND_SPEED_KMH: float = 1.0
# How far from a driving lane a background vehicle may be asked for and still
# be put on it, m (horizontally).  Farther, there is no lane there, and
# spawn_background places nothing, on every backend.
BACKGROUND_MAX_LANE_OFFSET_M: float = 5.0
