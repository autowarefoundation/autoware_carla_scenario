"""Turn a planned trajectory into CARLA vehicle controls.

alpasim's contract stops at the plan: the driver policy returns *where the vehicle should
be*, not throttle and steering.  Upstream that gap is filled by alpasim's vehicle dynamics
service; here it is filled the way Autoware fills it behind an end-to-end planner:

* **Lateral**: pure pursuit (Autoware also ships ``autoware_pure_pursuit``), trimmed by
  feedback on the measured yaw rate.
* **Longitudinal**: a port of Autoware's ``autoware_pid_longitudinal_controller`` -- its
  DRIVE / STOPPING / STOPPED / EMERGENCY states, its stop distance, delay compensation,
  velocity feedback, smooth stop and acceleration and jerk limits, with the defaults of
  its ``autoware_pid_longitudinal_controller.param.yaml``.  The plan carries poses and
  times only, so the speeds and accelerations Autoware's controller reads off a
  trajectory are put on it first, as ``autoware_tensorrt_e2e`` puts them on its own
  (``autoware_diffusion_planner``'s ``get_trajectory_from_poses``,
  :func:`plan_velocities`).
* **Vehicle interface**: the controller's acceleration is turned into pedals by the
  vehicle's powertrain model (:class:`~..utils.powertrain.ChaosPowertrain`), as
  Autoware's ``raw_vehicle_cmd_converter`` turns it with its accel and brake maps.

``ControlConfig.longitudinal_controller = "time_indexed"`` keeps the tracker this module
had before: the plan read on its own clock (where the plan has the vehicle *now*), which
:class:`~..actions.routing.RoutingAction` (``follow`` mode) drives OpenSCENARIO's timed
trajectories with.

All geometry in this module is in the **rig frame**: x forward, y left, z up,
right-handed.  Steering angles follow the same convention (positive = left), and the flip
to CARLA's left-handed steering (positive = right) happens exactly once, in
:meth:`VehicleCommand.to_carla_control`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

import numpy as np

from ..utils.powertrain import ChaosPowertrain
from .geometry import Pose, Trajectory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


__all__ = [
    "CONTROL_STATE_NAMES",
    "ControlConfig",
    "DRIVE",
    "EMERGENCY",
    "LONGITUDINAL_CONTROLLERS",
    "STOPPED",
    "STOPPING",
    "TrajectoryFollower",
    "VehicleCommand",
    "plan_velocities",
    "steer_angle",
    "steer_command",
]

logger = logging.getLogger(__name__)

#: Denominator guard for divisions by elapsed time or lookahead distance.
_EPSILON: float = 1e-6

#: The longitudinal controllers :attr:`ControlConfig.longitudinal_controller` names.
LONGITUDINAL_CONTROLLERS: Tuple[str, str] = ("pid", "time_indexed")

# Autoware's ``ControlState`` (pid_longitudinal_controller.hpp), in its order.
#: Following the plan's speed.
DRIVE: int = 0
#: Braking onto the plan's stop point.
STOPPING: int = 1
#: Standing, held on the brake.
STOPPED: int = 2
#: Overshot the stop point: braking hard.
EMERGENCY: int = 3
#: The states' names, by value.
CONTROL_STATE_NAMES: Tuple[str, str, str, str] = (
    "DRIVE",
    "STOPPING",
    "STOPPED",
    "EMERGENCY",
)

#: ``autoware::motion_utils::searchZeroVelocityIndex``'s threshold, m/s.
_ZERO_VELOCITY: float = 1e-3
#: ``removeOverlapPoints``' threshold, m, per coordinate (x and y: Autoware's is 2D).
_OVERLAP: float = 1e-8
#: ``getControlData``'s least distance worth compensating the delay for, m.
_MIN_RUNNING_DISTANCE: float = 0.01
#: ``SmoothStop::calculate``: how long the weak deceleration is kept after stopping, s.
_SMOOTH_STOP_WEAK_HOLD_S: float = 0.5
#: ``std::numeric_limits<double>::epsilon()``, where Autoware compares with it.
_DOUBLE_EPSILON: float = 2.220446049250313e-16

#: A non-finite plan or measurement is logged the first time, then once in this many
#: (as the HILS entity logs the non-finite command fields it drops).
_NON_FINITE_LOG_EVERY: int = 100

#: The fields only the ``time_indexed`` controller reads, which a ``pid`` config
#: setting away from their defaults is warned about (:meth:`ControlConfig.from_mapping`).
_TIME_INDEXED_FIELDS: Tuple[str, ...] = (
    "speed_kp",
    "speed_ki",
    "integral_limit",
    "position_gain",
    "max_position_correction_mps",
    "speed_preview_s",
    "stop_distance_m",
)


@dataclass(frozen=True)
class ControlConfig:
    """Gains and limits for :class:`TrajectoryFollower`.

    The longitudinal fields of the ``"pid"`` controller carry the names and the
    defaults of Autoware's ``autoware_pid_longitudinal_controller.param.yaml``; see
    that package for their meaning in full.
    """

    # -- Lateral (pure pursuit) ------------------------------------------------
    lookahead_gain_s: float = 0.6
    """Lookahead distance per unit speed, in seconds."""

    min_lookahead_m: float = 4.0
    """Lower bound on the lookahead distance.

    Also the least distance pure pursuit divides by: a plan shorter than this
    is pursued at its farthest point, and a point a few decimetres away, a few
    centimetres off the axis, would otherwise ask for full lock.
    """

    max_lookahead_m: float = 20.0
    """Upper bound on the lookahead distance."""

    wheelbase_m: float = 2.8
    """Distance between front and rear axles, used by the pure pursuit law."""

    max_steer_angle_rad: float = math.radians(56.0)
    """Wheel angle at full lock in CARLA's steering response (see :attr:`steer_exponent`)."""

    steer_exponent: float = 2.0
    """How CARLA's ``[-1, 1]`` steer turns the wheels:
    ``angle = max_steer_angle * |steer| ** steer_exponent``.

    CARLA 0.10's Chaos vehicles answer quadratically, whatever ``max_steer_angle`` the
    wheel physics reports (70 degrees): measured on the MKZ at 2 and 8 m/s, steer 0.1
    turns like 0.56 degrees, 0.3 like 5.0, 0.5 like 13.9 -- ``56 * steer**2`` within a
    few percent up to 0.5, at every speed.  A linear map asks for a tenth of the needed
    angle on a gentle curve.  ``1.0`` is a linear map.
    """

    max_steer_rate: float = 4.0
    """Maximum change in normalised steering per second."""

    yaw_rate_ki: float = 3.0
    """Integral feedback on the yaw rate.

    Pure pursuit asks for the yaw rate ``speed * curvature``; this trims the steering
    angle until the vehicle delivers it, absorbing what the steering map and the bicycle
    model miss (understeer, another vehicle's steering response).  The error is taken as
    the steering angle the bicycle model would need for the missing yaw rate, so the gain
    is unitless and the same at every speed.  Integral only: the vehicle answers a tick
    later, and a proportional term on top of that delay oscillates.  ``0`` steers
    open-loop.
    """

    yaw_rate_trim_limit_rad: float = math.radians(25.0)
    """Clamp on the integrated trim, in radians of steering."""

    yaw_rate_min_speed_mps: float = 1.0
    """Below this speed the yaw rate says little about the steering, so the trim is
    held rather than integrated."""

    # -- Longitudinal: which controller -------------------------------------------
    longitudinal_controller: str = "pid"
    """``"pid"``: the port of Autoware's ``autoware_pid_longitudinal_controller``,
    which reads the plan spatially -- the speed at the point the vehicle is at -- and
    stops at the plan's first zero-speed point (or its end).  ``"time_indexed"``: the
    plan read on its own clock, a PI on the speed it has the vehicle at *now* plus a
    pull towards where it has the vehicle now (the ``speed_*``, ``position_*`` and
    ``stop_distance_m`` fields); what
    :class:`~..actions.routing.RoutingAction` (``follow`` mode) drives timed
    trajectories with."""

    # -- Longitudinal "pid": the speeds put on the plan -----------------------------
    # autoware_tensorrt_e2e's ``postprocess`` parameters (e2e_planner.param.yaml).
    velocity_smoothing_window: int = 8
    """How many points, from each on, its speed is averaged over (forward moving
    average).  Capped at one less than the plan's points, where Autoware's node refuses
    a plan no longer than its window."""

    stopping_threshold: float = 0.3
    """m/s: a plan whose smoothed speed falls below this, from above, stops there
    (every later point is put at the point before, at zero speed) -- only when the
    vehicle was moving as the plan was made, as the node enables it."""

    # -- Longitudinal "pid": autoware_pid_longitudinal_controller.param.yaml -------
    delay_compensation_time: float = 0.17
    """s: how far ahead the vehicle's state is predicted to pick the target point."""

    enable_smooth_stop: bool = True
    """Brake onto a stop point through STOPPING; off, DRIVE goes straight to STOPPED."""

    enable_overshoot_emergency: bool = True
    """Brake hard (EMERGENCY) on overshooting the stop point by
    :attr:`emergency_state_overshoot_stop_dist`."""

    enable_keep_stopped_until_steer_convergence: bool = True
    """Keep STOPPED until the steering has converged.  The follower has no steering
    angle feedback and takes it as converged, so this holds only the first tick of
    a departure, as Autoware's first check does."""

    drive_state_stop_dist: float = 0.5
    """m: STOPPED departs once the stop point is farther than this."""

    drive_state_offset_stop_dist: float = 1.0
    """m: STOPPING returns to DRIVE once the stop point is farther than
    ``drive_state_stop_dist`` plus this."""

    stopping_state_stop_dist: float = 0.5
    """m: DRIVE starts stopping once the stop point is nearer than this."""

    stopped_state_entry_duration_time: float = 0.1
    """s the vehicle must not have been running for to count as stopped."""

    stopped_state_entry_vel: float = 0.01
    """m/s below which the vehicle is not running."""

    stopped_state_entry_acc: float = 0.1
    """m/s²: below ``stopped_state_entry_vel``, a lookahead point accelerating less
    than this is a stop point (velocity lookahead feedback)."""

    emergency_state_overshoot_stop_dist: float = 1.5
    """m past the stop point that is an overshoot (EMERGENCY)."""

    kp: float = 1.0
    """Proportional gain on the speed error (the PID, when not
    :attr:`enable_velocity_lookahead_feedback`)."""

    ki: float = 0.1
    """Integral gain on the speed error."""

    kd: float = 0.0
    """Derivative gain on the speed error."""

    max_out: float = 1.0
    """m/s²: upper bound of the PID's output."""

    min_out: float = -1.0
    """m/s²: lower bound of the PID's output."""

    max_p_effort: float = 1.0
    """m/s²: upper bound of the proportional term."""

    min_p_effort: float = -1.0
    """m/s²: lower bound of the proportional term."""

    max_i_effort: float = 0.3
    """m/s²: upper bound of the integral term."""

    min_i_effort: float = -0.3
    """m/s²: lower bound of the integral term."""

    max_d_effort: float = 0.0
    """m/s²: upper bound of the derivative term."""

    min_d_effort: float = 0.0
    """m/s²: lower bound of the derivative term."""

    lpf_vel_error_gain: float = 0.9
    """Low-pass gain on the speed error, per control step (``x = g x + (1 - g) u``)."""

    enable_integration_at_low_speed: bool = False
    """Integrate the speed error of a vehicle stuck below
    :attr:`current_vel_threshold_pid_integration`."""

    current_vel_threshold_pid_integration: float = 0.5
    """m/s above which the speed error is integrated."""

    time_threshold_before_pid_integration: float = 2.0
    """s under control before a stuck vehicle's error is integrated."""

    ff_scale_min: float = 0.5
    """Lower bound of the feedforward's scale (current over target speed)."""

    ff_scale_max: float = 2.0
    """Upper bound of the feedforward's scale."""

    enable_velocity_lookahead_feedback: bool = True
    """Close the gap to the plan's speed a lookahead ahead, blended with its
    acceleration there, instead of the PID (Autoware's default)."""

    velocity_lookahead_time: float = 1.0
    """s: the lookahead is the current speed (at least 0.1 m/s) times this, and the
    speed gap is closed over it."""

    velocity_lookahead_blend_weight: float = 0.5
    """Weight of the gap-closing acceleration against the plan's own."""

    enable_brake_keeping_before_stop: bool = False
    """Keep braking at least :attr:`brake_keeping_acc` from the plan's last
    deceleration before its stop."""

    brake_keeping_acc: float = -0.2
    """m/s² kept before a stop (see :attr:`enable_brake_keeping_before_stop`)."""

    smooth_stop_max_strong_acc: float = -0.5
    """m/s²: weakest strong deceleration a smooth stop starts with."""

    smooth_stop_min_strong_acc: float = -0.8
    """m/s²: strongest strong deceleration a smooth stop starts with."""

    smooth_stop_weak_acc: float = -0.3
    """m/s²: the weak deceleration of a smooth stop's last part."""

    smooth_stop_weak_stop_acc: float = -0.8
    """m/s² once past the stop point by ``smooth_stop_weak_stop_dist``."""

    smooth_stop_strong_stop_acc: float = -3.4
    """m/s² once stopped, or past the stop point by ``smooth_stop_strong_stop_dist``."""

    smooth_stop_max_fast_vel: float = 0.5
    """m/s above which a vehicle with no predicted stop brakes strongly."""

    smooth_stop_min_running_vel: float = 0.01
    """m/s above which the vehicle is running, for the smooth stop."""

    smooth_stop_min_running_acc: float = 0.01
    """m/s² above which the vehicle is running, for the smooth stop."""

    smooth_stop_weak_stop_time: float = 0.8
    """s: a vehicle predicted to stop later than this (plus the delay) brakes strongly."""

    smooth_stop_weak_stop_dist: float = -0.3
    """m: stop distance below which the weak stop deceleration is applied."""

    smooth_stop_strong_stop_dist: float = -0.5
    """m: stop distance below which the strong stop deceleration is applied."""

    stopped_vel: float = 0.0
    """m/s commanded in STOPPED."""

    stopped_acc: float = -3.4
    """m/s² commanded in STOPPED: through the powertrain model, the brake that holds
    the vehicle."""

    emergency_vel: float = 0.0
    """m/s commanded in EMERGENCY."""

    emergency_acc: float = -5.0
    """m/s² commanded in EMERGENCY."""

    emergency_jerk: float = -3.0
    """m/s³ the EMERGENCY deceleration is reached at."""

    lpf_acc_error_gain: float = 0.98
    """Low-pass gain on the acceleration error."""

    acc_feedback_gain: float = 0.0
    """Feedback gain on the filtered acceleration error."""

    max_acc: float = 3.0
    """m/s²: upper limit of the commanded acceleration."""

    min_acc: float = -5.0
    """m/s²: lower limit of the commanded acceleration."""

    max_jerk: float = 2.0
    """m/s³: upper limit of the commanded acceleration's rate."""

    min_jerk: float = -5.0
    """m/s³: lower limit of the commanded acceleration's rate."""

    max_acc_cmd_diff: float = 50.0
    """m/s³: limit of the output acceleration's rate, both ways."""

    ego_nearest_dist_threshold: float = 3.0
    """m: a plan point farther than this is not the vehicle's nearest unless none is
    nearer (autoware_launch's ``nearest_search.param.yaml``)."""

    ego_nearest_yaw_threshold: float = 1.046
    """rad: a plan point heading further off the vehicle's is not its nearest unless
    none is within it."""

    accel_lowpass_gain: float = 0.9
    """Low-pass gain on the vehicle's acceleration, estimated from its speed, per
    control step (``autoware_twist2accel``'s ``accel_lowpass_gain``)."""

    # -- Longitudinal "time_indexed" ----------------------------------------------
    # The plan is tracked in time: where the plan has the vehicle now, how fast and
    # how hard it is speeding up there, with a PI on what the vehicle misses of that,
    # all in m/s²; the vehicle's powertrain model turns the sum into pedals.
    speed_kp: float = 1.0
    """``time_indexed``: m/s² asked per m/s the vehicle is slower (faster) than the plan."""

    speed_ki: float = 0.1
    """``time_indexed``: m/s² per m·s of accumulated speed error."""

    integral_limit: float = 3.0
    """``time_indexed``: anti-windup bound on the accumulated speed error, m."""

    position_gain: float = 0.5
    """``time_indexed``: m/s of extra target speed per metre the vehicle is behind the
    plan (less when ahead).  With :attr:`speed_kp` it makes the gap close like a spring
    of ``sqrt(speed_kp * position_gain)`` rad/s, damped
    ``sqrt(speed_kp / position_gain) / 2``."""

    max_position_correction_mps: float = 6.0
    """``time_indexed``: bound on that correction, m/s."""

    speed_preview_s: float = 1.0
    """``time_indexed``: over how much of the plan, from now, its acceleration is read,
    in seconds.  Not its first segment alone: a plan pulling away from a standstill
    starts at a crawl."""

    stop_distance_m: float = 0.5
    """``time_indexed``: a plan none of whose points is this far from the vehicle is a
    request to stand still, and the vehicle is held on :attr:`stop_brake`."""

    # -- Vehicle interface ----------------------------------------------------------
    throttle_deadband: float = 0.3
    """The least throttle a standing vehicle (below :attr:`standstill_speed_mps`) is
    given when asked to speed up; ``0`` sets no floor.

    Part of the vehicle interface, not of the controller: the powertrain model, like
    an accel map, has a standing car move at any throttle, while a standing Lincoln
    MKZ (CARLA 0.10) stays put at up to 0.245 and pulls away from 0.26 (measured on
    Town10's red lights turning green).  Autoware's velocity lookahead feedback asks a
    standing car for as much as the plan's first metre accelerates, and a plan that
    pulls away at a crawl asks for a throttle near 0.23: without a floor the car
    never moves, so neither does the plan.  A real car creeps at any throttle, so a
    plan asking to move does move, a plan asking to move on a red light included.
    The ``time_indexed`` controller takes 0.25
    (:class:`~..actions.routing.RoutingAction` (``follow`` mode) sets it).
    """

    standstill_speed_mps: float = 0.5
    """Below this speed the vehicle counts as standing, for :attr:`throttle_deadband`."""

    stop_brake: float = 0.6
    """Brake applied with no plan to track (and, ``time_indexed``, when holding still)."""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ControlConfig":
        """Return gains built from a plain mapping (e.g. a Hydra node).

        Every ``*_rad`` field is also accepted as ``*_deg`` (``max_steer_angle_deg``,
        ``yaw_rate_trim_limit_deg``), since degrees read better in YAML.  A
        ``time_indexed`` field set away from its default for the ``pid`` controller,
        which does not read it, is warned about.

        Raises:
            ValueError: If *mapping* holds a key this config does not define, or
                names an unknown :attr:`longitudinal_controller`.
        """
        # Imported here: the Codon check does not compile this method, and Codon's
        # dataclasses has no fields() (docs/typecheck.md).
        from dataclasses import fields  # noqa: PLC0415

        known = {field.name for field in fields(cls)}
        aliases = {
            name[: -len("_rad")] + "_deg": name
            for name in known
            if name.endswith("_rad")
        }
        values = dict(mapping)
        for alias, name in aliases.items():
            degrees = values.pop(alias, None)
            if degrees is not None:
                values[name] = math.radians(float(degrees))

        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown ControlConfig key(s): {unknown}. "
                f"Known keys: {sorted(known | set(aliases))}"
            )
        config = cls(**values)
        if config.longitudinal_controller not in LONGITUDINAL_CONTROLLERS:
            raise ValueError(
                f"Unknown longitudinal_controller {config.longitudinal_controller!r}; "
                f"expected one of {list(LONGITUDINAL_CONTROLLERS)}"
            )
        if config.longitudinal_controller == "pid":
            default = cls()
            unread = [
                name
                for name in _TIME_INDEXED_FIELDS
                if name in values and getattr(config, name) != getattr(default, name)
            ]
            if unread:
                logger.warning(
                    "ControlConfig: %s only tune the time_indexed longitudinal "
                    "controller; the pid controller (Autoware's) ignores them",
                    ", ".join(unread),
                )
        return config


def steer_command(angle_rad: float, config: ControlConfig) -> float:
    """Return the ``[-1, 1]`` steer that turns the wheels by *angle_rad*, same sign."""
    ratio = min(1.0, abs(angle_rad) / config.max_steer_angle_rad)
    return math.copysign(ratio ** (1.0 / config.steer_exponent), angle_rad)


def steer_angle(command: float, config: ControlConfig) -> float:
    """Inverse of :func:`steer_command`: the wheel angle a steer gives."""
    ratio = min(1.0, abs(command)) ** config.steer_exponent
    return math.copysign(ratio * config.max_steer_angle_rad, command)


@dataclass(frozen=True)
class VehicleCommand:
    """An actuation command, in CARLA's convention.

    Attributes:
        throttle: Throttle in ``[0, 1]``.
        steer: Normalised steering in ``[-1, 1]``, **positive = right**.
        brake: Brake in ``[0, 1]``.
        hand_brake: Whether the hand brake is engaged.
        reverse: Whether reverse gear is engaged.
        target_speed_mps: Speed the longitudinal controller was aiming for.
        lookahead_lateral_offset_m: Lateral offset of the pursued point, for diagnostics.
        acceleration_mps2: Acceleration the longitudinal controller asked the vehicle
            interface for.
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    target_speed_mps: float = 0.0
    lookahead_lateral_offset_m: float = 0.0
    acceleration_mps2: float = 0.0

    def to_carla_control(self) -> "carla.VehicleControl":
        """Return the equivalent :class:`carla.VehicleControl`."""
        import typesafe_carla.carla as _carla  # noqa: PLC0415

        return _carla.VehicleControl(
            throttle=float(self.throttle),
            steer=float(self.steer),
            brake=float(self.brake),
            hand_brake=self.hand_brake,
            reverse=self.reverse,
        )


# ---------------------------------------------------------------------------
# The plan's speeds: autoware_diffusion_planner's get_trajectory_from_poses
# ---------------------------------------------------------------------------


def plan_velocities(
    positions: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    timestamps_us: List[int],
    base: np.ndarray[tuple[int], np.dtype[np.float64]],
    base_time_us: int,
    velocity_smoothing_window: int,
    enable_force_stop: bool,
    stopping_threshold: float,
) -> Tuple[List[float], List[float], List[int]]:
    """Speeds and accelerations for a plan of poses, as Autoware's e2e node gives them.

    ``get_trajectory_from_poses`` (``autoware_diffusion_planner``'s
    ``postprocessing_utils.cpp``, which ``autoware_tensorrt_e2e`` calls): each point's
    speed is its distance from the previous point -- the first's from *base*, where
    the vehicle was as the plan was made -- over the time between them; then a forward
    moving average over *velocity_smoothing_window* points, the last points keeping
    the last average; with *enable_force_stop*, the first point whose average falls
    below *stopping_threshold* from above stops the plan: it and every later point
    stand on the point before it, at zero speed.  Each point's acceleration is the
    change of speed to the next point over the time between them, the last point's
    zero.

    Autoware's node spaces its points 0.1 s apart and the first 0.1 s after the plan's
    stamp; here the times come from *timestamps_us* and *base_time_us*, so a plan
    stamped every 0.1 s from its stamp gets Autoware's numbers.  Distances are in
    3D, as the node's.

    Args:
        positions: The plan's ``(N, 3)`` positions, ``N >= 1``.
        timestamps_us: Their times.
        base: Where the vehicle was as the plan was made, in the same frame.
        base_time_us: When it was there: the plan's stamp.
        velocity_smoothing_window: Points per average; capped at ``N - 1``.
        enable_force_stop: Whether a falling speed stops the plan (the node enables
            it while the vehicle moves).
        stopping_threshold: The speed, m/s, that stops it.

    Returns:
        ``(speeds, accelerations, pose_index)``: per point, its speed in m/s, its
        acceleration in m/s², and the index of the position it stands on (its own,
        or a stopped point's).

    Raises:
        ValueError: If the plan has no point.
    """
    count = int(positions.shape[0])
    if count < 1:
        raise ValueError("a plan needs a point to have speeds")
    steps: List[float] = [0.0] * count
    for index in range(count):
        previous_us = base_time_us if index == 0 else timestamps_us[index - 1]
        steps[index] = max((timestamps_us[index] - previous_us) * 1e-6, _EPSILON)

    velocities: List[float] = [0.0] * count
    previous_x = float(base[0])
    previous_y = float(base[1])
    previous_z = float(base[2])
    for index in range(count):
        x = float(positions[index, 0])
        y = float(positions[index, 1])
        z = float(positions[index, 2])
        distance = math.sqrt(
            (x - previous_x) ** 2 + (y - previous_y) ** 2 + (z - previous_z) ** 2
        )
        velocities[index] = distance / steps[index]
        previous_x, previous_y, previous_z = x, y, z

    window = max(1, min(velocity_smoothing_window, count - 1))
    pose_index: List[int] = list(range(count))
    force_stop = False
    for index in range(count - window + 1):
        total = 0.0
        for offset in range(window):
            total += velocities[index + offset]
        velocities[index] = total / window
        if (
            enable_force_stop
            and index > 0
            and abs(velocities[index - 1]) > stopping_threshold
            and abs(velocities[index]) < stopping_threshold
        ):
            force_stop = True
        if index > 0 and force_stop:
            velocities[index] = 0.0
            pose_index[index] = pose_index[index - 1]
    last = velocities[count - window]
    for index in range(count - window + 1, count):
        velocities[index] = last
        if force_stop:
            velocities[index] = 0.0
            pose_index[index] = pose_index[index - 1]

    accelerations: List[float] = [0.0] * count
    for index in range(count - 1):
        accelerations[index] = (velocities[index + 1] - velocities[index]) / steps[
            index + 1
        ]
    return velocities, accelerations, pose_index


def _padded(values: List[float], count: int) -> List[float]:
    """*values*, cut or padded with its last (or zero) to *count*."""
    padded: List[float] = []
    for index in range(count):
        if index < len(values):
            padded.append(float(values[index]))
        elif padded:
            padded.append(padded[-1])
        else:
            padded.append(0.0)
    return padded


def _given_velocities(
    given: List[float],
    given_accelerations: Optional[List[float]],
    timestamps_us: List[int],
) -> Tuple[List[float], List[float], List[int]]:
    """A plan's own speeds and accelerations (an Autoware trajectory's); without
    accelerations, the change of speed over time to the next point."""
    count = len(timestamps_us)
    velocities = _padded(given, count)
    if given_accelerations is not None:
        return velocities, _padded(given_accelerations, count), list(range(count))
    accelerations: List[float] = [0.0] * count
    for index in range(count - 1):
        step = max((timestamps_us[index + 1] - timestamps_us[index]) * 1e-6, _EPSILON)
        accelerations[index] = (velocities[index + 1] - velocities[index]) / step
    return velocities, accelerations, list(range(count))


# ---------------------------------------------------------------------------
# Trajectory geometry: what Autoware's controller takes from motion_utils
# ---------------------------------------------------------------------------


class _Point:
    """A trajectory point: where, heading, and the speed and acceleration there."""

    x: float
    y: float
    z: float
    yaw: float
    vel: float
    acc: float

    def __init__(
        self, x: float, y: float, z: float, yaw: float, vel: float, acc: float
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.vel = vel
        self.acc = acc


def _normalize_angle(angle: float) -> float:
    """*angle* in ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _first_nearest_index(
    points: List[_Point],
    x: float,
    y: float,
    yaw: float,
    dist_threshold: float,
    yaw_threshold: float,
) -> int:
    """``findFirstNearestIndexWithSoftConstraints``: the nearest point of the first run
    of points within the distance and heading thresholds; failing that, within the
    distance alone; failing that, within the heading alone; failing that, the
    nearest."""
    squared_threshold = dist_threshold * dist_threshold
    for use_dist, use_yaw in ((True, True), (True, False), (False, True)):
        within = False
        best = math.inf
        found = 0
        for index in range(len(points)):
            point = points[index]
            squared = (point.x - x) ** 2 + (point.y - y) ** 2
            off = (use_dist and squared > squared_threshold) or (
                use_yaw and abs(_normalize_angle(point.yaw - yaw)) > yaw_threshold
            )
            if off:
                if within:
                    break
                continue
            if best <= squared:
                continue
            best = squared
            found = index
            within = True
        if within:
            return found
    nearest = 0
    best = math.inf
    for index in range(len(points)):
        squared = (points[index].x - x) ** 2 + (points[index].y - y) ** 2
        if squared < best:
            best = squared
            nearest = index
    return nearest


def _offset_to_segment(points: List[_Point], segment: int, x: float, y: float) -> float:
    """``calcLongitudinalOffsetToSegment``: how far along *segment* ``(x, y)`` projects,
    from its start (NaN when no distinct point follows it)."""
    front = points[segment]
    for index in range(segment + 1, len(points)):
        back = points[index]
        if abs(back.x - front.x) < _OVERLAP and abs(back.y - front.y) < _OVERLAP:
            continue
        dx = back.x - front.x
        dy = back.y - front.y
        length = math.hypot(dx, dy)
        if length < _OVERLAP:
            return math.nan
        return ((x - front.x) * dx + (y - front.y) * dy) / length
    return math.nan


def _first_nearest_segment(
    points: List[_Point],
    x: float,
    y: float,
    yaw: float,
    dist_threshold: float,
    yaw_threshold: float,
) -> int:
    """``findFirstNearestSegmentIndexWithSoftConstraints``."""
    nearest = _first_nearest_index(points, x, y, yaw, dist_threshold, yaw_threshold)
    if nearest == 0:
        return 0
    if nearest == len(points) - 1:
        return len(points) - 2
    if _offset_to_segment(points, nearest, x, y) <= 0.0:
        return nearest - 1
    return nearest


def _arc_length(points: List[_Point], source: int, destination: int) -> float:
    """``calcSignedArcLength`` between two points, in 2D: negative backwards."""
    if source > destination:
        return -_arc_length(points, destination, source)
    total = 0.0
    for index in range(source, destination):
        total += math.hypot(
            points[index + 1].x - points[index].x, points[index + 1].y - points[index].y
        )
    return total


def _lerp_point(points: List[_Point], segment: int, x: float, y: float) -> _Point:
    """``lerpTrajectoryPoint``: the point of *segment* ``(x, y)`` projects onto."""
    if len(points) == 1:
        only = points[0]
        return _Point(only.x, only.y, only.z, only.yaw, only.vel, only.acc)
    along = _offset_to_segment(points, segment, x, y)
    length = _arc_length(points, segment, segment + 1)
    ratio = 0.0
    if math.isfinite(length) and abs(length) >= 1e-6:
        ratio = min(max(along / length, 0.0), 1.0)
    front = points[segment]
    back = points[segment + 1]
    return _Point(
        front.x + (back.x - front.x) * ratio,
        front.y + (back.y - front.y) * ratio,
        front.z + (back.z - front.z) * ratio,
        front.yaw + _normalize_angle(back.yaw - front.yaw) * ratio,
        front.vel + (back.vel - front.vel) * ratio,
        front.acc + (back.acc - front.acc) * ratio,
    )


def _remove_overlap(points: List[_Point]) -> List[_Point]:
    """``removeOverlapPoints``: each point that stands on the one kept before it, dropped."""
    kept: List[_Point] = []
    for point in points:
        if kept:
            last = kept[-1]
            if abs(last.x - point.x) < _OVERLAP and abs(last.y - point.y) < _OVERLAP:
                continue
        kept.append(point)
    return kept


def _pose_after_distance(points: List[_Point], source: int, distance: float) -> _Point:
    """``findTrajectoryPoseAfterDistance``: the pose *distance* (3D) along from *source*."""
    remaining = distance
    found = points[-1]
    for index in range(source, len(points) - 1):
        front = points[index]
        back = points[index + 1]
        length = math.sqrt(
            (back.x - front.x) ** 2 + (back.y - front.y) ** 2 + (back.z - front.z) ** 2
        )
        if remaining < length:
            if remaining <= 0.0:
                return front
            ratio = remaining / length
            return _Point(
                front.x + (back.x - front.x) * ratio,
                front.y + (back.y - front.y) * ratio,
                front.z + (back.z - front.z) * ratio,
                front.yaw + _normalize_angle(back.yaw - front.yaw) * ratio,
                front.vel,
                front.acc,
            )
        remaining -= length
    return found


def _zero_velocity_index(points: List[_Point]) -> int:
    """``searchZeroVelocityIndex``: the first point standing still, or ``-1``."""
    for index in range(len(points)):
        if abs(points[index].vel) < _ZERO_VELOCITY:
            return index
    return -1


def _stop_distance(
    points: List[_Point], dist_threshold: float, yaw_threshold: float
) -> float:
    """``calcStopDistance``: the arc length from the vehicle (the rig origin) to the
    first point standing still, or to the plan's end; negative once past it."""
    count = len(points)
    if count < 2:
        return 0.0
    zero = _zero_velocity_index(points)
    end = zero if zero >= 0 else count - 1
    segment = _first_nearest_segment(
        points, 0.0, 0.0, 0.0, dist_threshold, yaw_threshold
    )
    end_segment = min(end, count - 2)
    signed = (
        _arc_length(points, segment, end_segment)
        - _offset_to_segment(points, segment, 0.0, 0.0)
        + _offset_to_segment(points, end_segment, points[end].x, points[end].y)
    )
    if math.isnan(signed):
        return 0.0
    return signed


def _finite_or(value: float, fallback: float) -> float:
    """*value*, or *fallback* when it is NaN or infinite: what controller state is
    updated with, so one bad input cannot latch in it."""
    return value if math.isfinite(value) else fallback


def _all_finite(values: Optional[List[float]]) -> bool:
    """Whether every one of *values* (none counts) is finite."""
    if values is None:
        return True
    for value in values:
        if not math.isfinite(value):
            return False
    return True


def _plan_is_finite(
    plan: Trajectory,
    velocities: Optional[List[float]],
    accelerations: Optional[List[float]],
) -> bool:
    """Whether *plan*'s poses, and its own *velocities* and *accelerations* when
    given, are all finite: a policy's NaN must not reach the controllers."""
    for pose in plan.poses:
        if not (
            bool(np.isfinite(pose.position).all())
            and bool(np.isfinite(pose.quat_xyzw).all())
        ):
            return False
    return _all_finite(velocities) and _all_finite(accelerations)


def _diff_limit(
    value: float, previous: float, dt: float, upper: float, lower: float
) -> float:
    """``applyDiffLimitFilter``: *value*, its rate from *previous* held within bounds."""
    rate = min(max((value - previous) / dt, lower), upper)
    return previous + rate * dt


# ---------------------------------------------------------------------------
# The follower
# ---------------------------------------------------------------------------


class TrajectoryFollower:
    """Tracks a planned trajectory with pure pursuit and Autoware's longitudinal control.

    One instance drives one vehicle; call :meth:`reset` when the plan's provenance
    changes (e.g. a new session).

    Args:
        config: Gains and limits.  ``None`` uses the defaults.
        powertrain: The vehicle's powertrain model, which turns an acceleration into
            pedals.  May be set later (:attr:`powertrain`), before the first plan is
            tracked.
    """

    _config: ControlConfig
    powertrain: Optional[ChaosPowertrain]
    _integral: float
    _previous_steer: float
    _yaw_rate_trim: float
    _asked_curvature: Optional[float]

    # The "pid" controller's state: Autoware's PidLongitudinalController members.
    _state: int
    _now_s: float
    _pid_integral: float
    _pid_previous_error: float
    _pid_first: bool
    _lpf_vel_error: float
    _lpf_acc_error: float
    _previous_raw_vel: float
    _previous_raw_acc: float
    _previous_acc: float
    _ctrl_cmd_history: List[Tuple[float, float]]
    _vel_history: List[Tuple[float, float]]
    _last_running_s: Optional[float]
    _previous_keep_stopped: Optional[bool]
    _strong_acc: float
    _weak_acc_s: float
    _previous_speed: Optional[float]
    _acceleration: float
    _stop_distance: float

    # The plan the speeds were put on (told apart by its length, its first and last
    # times and its first point), and those speeds.
    _has_profile: bool
    _profile_key: Tuple[int, int, int, float, float]
    _velocities: List[float]
    _accelerations: List[float]
    _pose_index: List[int]
    # The last plan taken, tracked on while a plan is not (setTrajectory).
    _last_plan: Optional[Trajectory]
    _last_given: Optional[List[float]]
    _last_given_accelerations: Optional[List[float]]
    _started: bool
    # How many non-finite plans, measurements and commands were refused (the log's
    # rate limit); not cleared by reset().
    _non_finite: int

    def __init__(
        self,
        config: Optional[ControlConfig] = None,
        powertrain: Optional[ChaosPowertrain] = None,
    ) -> None:
        # A ControlConfig is always true, so this is `config or ControlConfig()`,
        # which Codon cannot type (one side Optional, the other not).
        if config is None:
            config = ControlConfig()
        if config.longitudinal_controller not in LONGITUDINAL_CONTROLLERS:
            raise ValueError(
                "Unknown longitudinal_controller '"
                + config.longitudinal_controller
                + "'; expected 'pid' or 'time_indexed'"
            )
        self._config = config
        self.powertrain = powertrain
        self._non_finite = 0
        self.reset()

    @property
    def config(self) -> ControlConfig:
        """Return the controller configuration."""
        return self._config

    @property
    def control_state(self) -> int:
        """The ``pid`` controller's state: :data:`DRIVE`, :data:`STOPPING`,
        :data:`STOPPED` or :data:`EMERGENCY`."""
        return self._state

    @property
    def stop_distance_m(self) -> float:
        """The ``pid`` controller's last stop distance, m (``calcStopDistance``)."""
        return self._stop_distance

    def reset(self) -> None:
        """Clear the controllers' memory: the integral terms, the steering state, and
        the ``pid`` controller back to STOPPED."""
        self._integral = 0.0
        self._previous_steer = 0.0
        self._yaw_rate_trim = 0.0
        self._asked_curvature = None
        self._state = STOPPED
        self._now_s = 0.0
        self._pid_integral = 0.0
        self._pid_previous_error = 0.0
        self._pid_first = True
        self._lpf_vel_error = 0.0
        self._lpf_acc_error = 0.0
        self._previous_raw_vel = 0.0
        self._previous_raw_acc = 0.0
        self._previous_acc = 0.0
        self._ctrl_cmd_history = []
        self._vel_history = []
        self._last_running_s = None
        self._previous_keep_stopped = None
        self._strong_acc = 0.0
        self._weak_acc_s = 0.0
        self._previous_speed = None
        self._acceleration = 0.0
        self._stop_distance = 0.0
        self._has_profile = False
        self._profile_key = (0, 0, 0, 0.0, 0.0)
        self._velocities = []
        self._accelerations = []
        self._pose_index = []
        self._last_plan = None
        self._last_given = None
        self._last_given_accelerations = None
        self._started = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def step(
        self,
        plan_in_local: Trajectory,
        pose_local_to_rig: Pose,
        current_speed_mps: float,
        dt_s: float,
        yaw_rate_rps: Optional[float] = None,
        now_us: Optional[int] = None,
        gear: int = 0,
        plan_velocities_mps: Optional[List[float]] = None,
        plan_accelerations_mps2: Optional[List[float]] = None,
    ) -> VehicleCommand:
        """Return the command that tracks *plan_in_local* from the current pose.

        Args:
            plan_in_local: The driver's plan, in the local frame.
            pose_local_to_rig: The ego's current pose in the local frame.
            current_speed_mps: Measured ground speed.
            dt_s: Time since the previous call, in seconds.
            yaw_rate_rps: The ego's measured yaw rate, positive to the left.  ``None``
                steers on pure pursuit alone.
            now_us: The current time, on the plan's clock.  ``time_indexed``: ``None``
                takes the plan's first instant as now.  ``pid``: when the plan
                arrived, which tells a leading anchor point at the planning time
                from a planned point.
            gear: The gear the vehicle is in, as ``carla.VehicleControl.gear``
                reports it (``0``, neutral, is taken as the first gear a throttle
                selects).
            plan_velocities_mps: The plan's own speeds, one per point (``pid``
                only), as an Autoware trajectory carries them.  ``None`` puts speeds
                on the plan from its poses and times (:func:`plan_velocities`).
            plan_accelerations_mps2: With *plan_velocities_mps*, the plan's own
                accelerations; ``None`` takes the change of speed between points.

        Raises:
            RuntimeError: If no :attr:`powertrain` has been set.

        Returns:
            The actuation command.  ``time_indexed``: an empty plan yields a full
            stop.  ``pid``: a plan of fewer than two points is ignored, as
            Autoware's ``setTrajectory`` ignores it, and the last plan is tracked
            on; with none yet, a full stop.

            A plan that is not all finite (a pose, or ``pid``, a given speed or
            acceleration that is NaN or infinite) is refused the same way: the last
            plan is tracked on, and with none yet, a full stop.  A non-finite yaw
            rate is not fed back; a non-finite speed or time step, or a command
            that comes out non-finite, gives :meth:`_fallback` (no throttle, the
            stop brake, the steering held), the controllers' state untouched.
            Each is logged, the first time and then once in a hundred.
        """
        time_indexed = self._config.longitudinal_controller == "time_indexed"
        given_checked: Optional[List[float]] = None
        accelerations_checked: Optional[List[float]] = None
        if not time_indexed:
            # time_indexed does not read the plan's own speeds.
            given_checked = plan_velocities_mps
            accelerations_checked = plan_accelerations_mps2
        if not _plan_is_finite(plan_in_local, given_checked, accelerations_checked):
            self._refuse_non_finite("plan")
            last = self._last_plan
            if last is None:
                return self._hold_still()
            plan_in_local = last
            plan_velocities_mps = self._last_given
            plan_accelerations_mps2 = self._last_given_accelerations
        if not (math.isfinite(current_speed_mps) and math.isfinite(dt_s)):
            self._refuse_non_finite("speed or time step")
            return self._fallback()
        if yaw_rate_rps is not None and not math.isfinite(yaw_rate_rps):
            self._refuse_non_finite("yaw rate")
            yaw_rate_rps = None
        if time_indexed:
            self._last_plan = plan_in_local
            command = self._step_time_indexed(
                plan_in_local,
                pose_local_to_rig,
                current_speed_mps,
                dt_s,
                yaw_rate_rps,
                now_us,
                gear,
            )
        else:
            command = self._step_pid(
                plan_in_local,
                pose_local_to_rig,
                current_speed_mps,
                dt_s,
                yaw_rate_rps,
                now_us,
                gear,
                plan_velocities_mps,
                plan_accelerations_mps2,
            )
        if not (
            math.isfinite(command.throttle)
            and math.isfinite(command.steer)
            and math.isfinite(command.brake)
        ):
            self._refuse_non_finite("command")
            return self._fallback()
        return command

    def _step_pid(
        self,
        plan_in_local: Trajectory,
        pose_local_to_rig: Pose,
        current_speed_mps: float,
        dt_s: float,
        yaw_rate_rps: Optional[float],
        now_us: Optional[int],
        gear: int,
        plan_velocities_mps: Optional[List[float]],
        plan_accelerations_mps2: Optional[List[float]],
    ) -> VehicleCommand:
        """:meth:`step` with the ``pid`` longitudinal controller."""
        given = plan_velocities_mps
        given_accelerations = plan_accelerations_mps2
        if self._refresh_profile(
            plan_in_local,
            pose_local_to_rig,
            current_speed_mps,
            now_us,
            given,
            given_accelerations,
        ):
            self._last_plan = plan_in_local
            self._last_given = given
            self._last_given_accelerations = given_accelerations
        else:
            last = self._last_plan
            if last is None:
                return self._hold_still()
            plan_in_local = last
            given = self._last_given
            given_accelerations = self._last_given_accelerations
            self._refresh_profile(
                plan_in_local,
                pose_local_to_rig,
                current_speed_mps,
                now_us,
                given,
                given_accelerations,
            )
        plan_in_rig = plan_in_local.transform(pose_local_to_rig.inverse())
        points = plan_in_rig.positions
        target = self._lookahead_point(points, self._lookahead_m(current_speed_mps))
        steer = self._previous_steer
        lateral_offset = 0.0
        if target is not None:
            steer = self._lateral(target, current_speed_mps, dt_s, yaw_rate_rps)
            lateral_offset = float(target[1])
        acceleration, target_speed = self._pid_longitudinal(
            plan_in_rig, current_speed_mps, dt_s
        )
        throttle, brake = self._pedals(acceleration, current_speed_mps, gear)
        return VehicleCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            target_speed_mps=target_speed,
            lookahead_lateral_offset_m=lateral_offset,
            acceleration_mps2=acceleration,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hold_still(self) -> VehicleCommand:
        """Return a braking command and reset the controller state."""
        self.reset()
        return VehicleCommand(throttle=0.0, steer=0.0, brake=self._config.stop_brake)

    def _fallback(self) -> VehicleCommand:
        """The command for a step that could not be computed: no throttle, the stop
        brake, and the steering held where it was (not snapped to straight at
        speed); the controllers' state is kept for the next good step."""
        return VehicleCommand(
            throttle=0.0, steer=self._previous_steer, brake=self._config.stop_brake
        )

    def _refuse_non_finite(self, what: str) -> None:
        """Log a refused non-finite *what*: the first time, then once in a hundred."""
        if self._non_finite % _NON_FINITE_LOG_EVERY == 0:
            logger.warning(
                "TrajectoryFollower: refused a non-finite %s (NaN or infinite; "
                "%d refused so far)",
                what,
                self._non_finite + 1,
            )
        self._non_finite += 1

    def _lookahead_m(self, speed_mps: float) -> float:
        """Pure pursuit's lookahead distance at *speed_mps*."""
        return float(
            np.clip(
                self._config.lookahead_gain_s * max(speed_mps, 0.0),
                self._config.min_lookahead_m,
                self._config.max_lookahead_m,
            )
        )

    @staticmethod
    def _lookahead_point(
        points: np.ndarray[tuple[int, int], np.dtype[np.float64]], lookahead_m: float
    ) -> Optional[np.ndarray[tuple[int], np.dtype[np.float64]]]:
        """Return the first plan point at least *lookahead_m* ahead of the rig origin.

        Falls back to the farthest point when the plan is shorter than the lookahead,
        and ignores points behind the vehicle so a plan that starts slightly behind the
        rig origin does not fold the steering back on itself.

        Returns:
            A ``(3,)`` point in the rig frame, or ``None`` when no usable point exists.
        """
        if points.shape[0] == 0:
            return None
        ahead = points[points[:, 0] > 0.0]
        candidates = ahead if ahead.shape[0] else points
        distances = np.linalg.norm(candidates[:, :2], axis=1)
        beyond = np.flatnonzero(distances >= lookahead_m)
        index = int(beyond[0]) if beyond.size else int(np.argmax(distances))
        if distances[index] < _EPSILON:
            return None
        return candidates[index]

    def _lateral(
        self,
        target: np.ndarray[tuple[int], np.dtype[np.float64]],
        speed_mps: float,
        dt_s: float,
        yaw_rate_rps: Optional[float],
    ) -> float:
        """Return the normalised steering command for *target*, rate limited."""
        # A target nearer than the least lookahead (a short plan's end) is pursued
        # as if it were that far: its direction alone, not its nearness, sets the turn.
        distance = max(
            float(np.linalg.norm(target[:2])), self._config.min_lookahead_m, _EPSILON
        )
        alpha = math.atan2(float(target[1]), float(target[0]))
        curvature = 2.0 * math.sin(alpha) / distance
        angle = math.atan(self._config.wheelbase_m * curvature)
        if yaw_rate_rps is not None:
            angle += self._yaw_rate_feedback(speed_mps, yaw_rate_rps, curvature, dt_s)
        if not math.isfinite(angle):
            # steer_command would read NaN as full lock (min(1.0, nan) is 1.0).
            return self._previous_steer

        # Rig frame is right-handed (positive angle = left); CARLA steers positive right.
        command = -steer_command(angle, self._config)

        max_delta = self._config.max_steer_rate * max(dt_s, 0.0)
        if max_delta > 0.0:
            lower, upper = (
                self._previous_steer - max_delta,
                self._previous_steer + max_delta,
            )
            command = float(np.clip(command, lower, upper))
        self._previous_steer = command
        return command

    def _yaw_rate_feedback(
        self, speed_mps: float, yaw_rate_rps: float, curvature: float, dt_s: float
    ) -> float:
        """Return the steering trim, in radians, that closes the gap to the asked yaw rate."""
        config = self._config
        # The measured yaw rate answers the previous step's command, so it is held
        # against the curvature asked then; against this step's, every change in the
        # plan would read as a steering error for one step.
        asked = self._asked_curvature
        if math.isfinite(curvature):
            self._asked_curvature = curvature
        if asked is None or speed_mps < config.yaw_rate_min_speed_mps:
            return self._yaw_rate_trim
        # The yaw-rate error as a steering angle: what the bicycle model says it would
        # take to turn by the missing yaw rate.
        error = config.wheelbase_m * (asked - yaw_rate_rps / speed_mps)
        limit = config.yaw_rate_trim_limit_rad
        self._yaw_rate_trim = _finite_or(
            float(
                np.clip(
                    self._yaw_rate_trim + config.yaw_rate_ki * error * dt_s,
                    -limit,
                    limit,
                )
            ),
            self._yaw_rate_trim,
        )
        return self._yaw_rate_trim

    def _pedals(
        self, acceleration: float, current_speed_mps: float, gear: int
    ) -> Tuple[float, float]:
        """The vehicle interface: ``(throttle, brake)`` for *acceleration*."""
        if self.powertrain is None:
            raise RuntimeError(
                "TrajectoryFollower has no powertrain to turn accelerations into pedals"
            )
        throttle, brake = self.powertrain.pedals(acceleration, current_speed_mps, gear)
        if current_speed_mps < self._config.standstill_speed_mps and acceleration > 0.0:
            throttle = max(throttle, self._config.throttle_deadband)
        return throttle, brake

    # ------------------------------------------------------------------
    # Longitudinal "pid": autoware_pid_longitudinal_controller
    # ------------------------------------------------------------------

    def _refresh_profile(
        self,
        plan_in_local: Trajectory,
        pose_local_to_rig: Pose,
        current_speed_mps: float,
        now_us: Optional[int],
        given: Optional[List[float]],
        given_accelerations: Optional[List[float]],
    ) -> bool:
        """Put speeds on a plan the follower has not seen yet; whether it is one
        Autoware's controller takes (two points or more, its anchor aside).

        A plan led by a point stamped at or before now -- carla-driver-interface
        leads its plan with the ego's pose at the planning time -- has that point
        as its base, where the vehicle was as the plan was made, and the controller
        tracks the points after it, as the e2e node's trajectory starts a step
        after its stamp.  Otherwise the vehicle's pose as the follower first sees
        the plan is its base, at *now_us*.  Either is latched with the plan.  A
        plan that is not taken leaves the last one's speeds in place.
        """
        count = len(plan_in_local)
        if count == 0:
            return False
        first = plan_in_local.positions[0]
        key = (
            count,
            int(plan_in_local.timestamps_us[0]),
            int(plan_in_local.timestamps_us[-1]),
            float(first[0]),
            float(first[1]),
        )
        # A plan's own speeds are taken as given every time: nothing to keep.
        if given is None and self._has_profile and self._profile_key == key:
            return True
        stamps = [int(stamp) for stamp in plan_in_local.timestamps_us]
        start = 0
        if given is None and now_us is not None and stamps[0] <= now_us:
            start = 1
        if count - start < 2:
            return False
        if given is not None:
            velocities, accelerations, pose_index = _given_velocities(
                given, given_accelerations, stamps
            )
        else:
            config = self._config
            if start == 1:
                base = plan_in_local.positions[0]
                base_time_us = stamps[0]
            else:
                base = pose_local_to_rig.position
                base_time_us = stamps[0] - (stamps[1] - stamps[0])
                if now_us is not None and now_us < stamps[0]:
                    base_time_us = now_us
            velocities, accelerations, relative = plan_velocities(
                plan_in_local.positions[start:],
                stamps[start:],
                base,
                base_time_us,
                config.velocity_smoothing_window,
                # The node's ``ego.odometry.twist.twist.linear.x > epsilon``.
                current_speed_mps > _DOUBLE_EPSILON,
                config.stopping_threshold,
            )
            pose_index = [index + start for index in relative]
        self._has_profile = given is None
        self._profile_key = key
        self._velocities = velocities
        self._accelerations = accelerations
        self._pose_index = pose_index
        return True

    def _rig_points(self, plan_in_rig: Trajectory) -> List[_Point]:
        """The plan in the rig frame, with its speeds."""
        positions = plan_in_rig.positions
        poses = plan_in_rig.poses
        points: List[_Point] = []
        for index in range(len(self._velocities)):
            source = self._pose_index[index]
            points.append(
                _Point(
                    float(positions[source, 0]),
                    float(positions[source, 1]),
                    float(positions[source, 2]),
                    float(poses[source].yaw),
                    self._velocities[index],
                    self._accelerations[index],
                )
            )
        return points

    def _estimate_acceleration(self, speed_mps: float, dt_s: float) -> float:
        """The vehicle's acceleration, from its speed, low-pass filtered."""
        if self._previous_speed is not None:
            raw = (speed_mps - self._previous_speed) / dt_s
            gain = self._config.accel_lowpass_gain
            self._acceleration = gain * self._acceleration + (1.0 - gain) * raw
        self._previous_speed = speed_mps
        return self._acceleration

    def _pid_longitudinal(
        self, plan_in_rig: Trajectory, speed_mps: float, dt_s: float
    ) -> Tuple[float, float]:
        """``PidLongitudinalController::run``: ``(acceleration, target speed)``."""
        config = self._config
        dt = max(dt_s, _EPSILON)
        self._now_s += dt
        if not self._started:
            # Autoware's controller starts STOPPED, with the vehicle standing; one
            # taken over moving (a run-up, a spawn at speed, a new command
            # source) starts in DRIVE, or STOPPED's brake would pulse for a tick.
            self._started = True
            if abs(speed_mps) > config.stopped_state_entry_vel:
                self._state = DRIVE
        current_acc = self._estimate_acceleration(speed_mps, dt)
        dist_threshold = config.ego_nearest_dist_threshold
        yaw_threshold = config.ego_nearest_yaw_threshold

        # getControlData: the point the vehicle (the rig origin) is at, inserted.
        points = self._rig_points(plan_in_rig)
        segment = _first_nearest_segment(
            points, 0.0, 0.0, 0.0, dist_threshold, yaw_threshold
        )
        nearest = _lerp_point(points, segment, 0.0, 0.0)
        trajectory = points[: segment + 1] + [nearest] + points[segment + 1 :]
        predicted_vel, running_distance = self._predicted_state_after_delay(
            speed_mps, current_acc
        )
        target = nearest
        if running_distance > _MIN_RUNNING_DISTANCE:
            trajectory = _remove_overlap(trajectory)
            # Autoware walks from the index the nearest point was inserted at,
            # before the overlaps were removed.
            nearest_index = min(segment + 1, len(trajectory) - 1)
            pose = _pose_after_distance(trajectory, nearest_index, running_distance)
            target_segment = _first_nearest_segment(
                trajectory, pose.x, pose.y, pose.yaw, dist_threshold, yaw_threshold
            )
            target = _lerp_point(trajectory, target_segment, pose.x, pose.y)
            trajectory.insert(target_segment + 1, target)
        trajectory = _remove_overlap(trajectory)
        nearest_index = _first_nearest_index(
            trajectory, nearest.x, nearest.y, nearest.yaw, dist_threshold, yaw_threshold
        )
        target_index = _first_nearest_index(
            trajectory, target.x, target.y, target.yaw, dist_threshold, yaw_threshold
        )
        stop_dist = _stop_distance(trajectory, dist_threshold, yaw_threshold)
        self._stop_distance = stop_dist

        self._update_control_state(
            trajectory, nearest_index, speed_mps, stop_dist, predicted_vel
        )
        acceleration, velocity = self._ctrl_cmd(
            trajectory, target_index, speed_mps, current_acc, stop_dist, dt
        )

        # createCtrlCmdMsg: the speed history the smooth stop fits, as many entries
        # as control steps in the delay: 3 at the simulation's 0.05 s step, where
        # Autoware's 0.03 s control period keeps 5.
        self._vel_history.append((self._now_s, speed_mps))
        kept = max(int(config.delay_compensation_time / dt), 0)
        while len(self._vel_history) > kept:
            self._vel_history.pop(0)
        self._previous_acc = _finite_or(acceleration, self._previous_acc)
        return acceleration, velocity

    def _predicted_state_after_delay(
        self, current_vel: float, current_acc: float
    ) -> Tuple[float, float]:
        """``predictedStateAfterDelay``: ``(speed, distance run)`` after the delay."""
        delay = self._config.delay_compensation_time
        if not self._ctrl_cmd_history:
            time_to_stop = -current_vel / current_acc if current_acc != 0.0 else -1.0
            horizon = time_to_stop if 0.0 < time_to_stop < delay else delay
            predicted = current_vel + current_acc * horizon
            running = abs(horizon * current_vel + 0.5 * current_acc * horizon * horizon)
            return predicted, running
        predicted = current_vel
        running = 0.0
        history = self._ctrl_cmd_history
        for index in range(len(history)):
            stamp, acc = history[index]
            if self._now_s - stamp < delay:
                if index == len(history) - 1:
                    span = min(self._now_s - history[-1][0], delay)
                else:
                    span = min(history[index + 1][0] - stamp, delay)
                running += abs(abs(predicted) * span + 0.5 * acc * span * span)
                predicted += -acc * span if predicted < 0.0 else acc * span
                if current_vel != 0.0 and predicted / current_vel < 0.0:
                    predicted = 0.0
                    break
        return predicted, running

    def _update_control_state(
        self,
        trajectory: List[_Point],
        nearest_index: int,
        current_vel: float,
        stop_dist: float,
        predicted_vel: float,
    ) -> None:
        """``updateControlState``, the vehicle always under Autoware's control."""
        config = self._config
        departure_from_stopping = (
            stop_dist
            > config.drive_state_stop_dist + config.drive_state_offset_stop_dist
        )
        departure_from_stopped = stop_dist > config.drive_state_stop_dist
        stopping_condition = stop_dist < config.stopping_state_stop_dist
        is_stopped = abs(current_vel) < config.stopped_state_entry_vel
        # Forward gear: stopped, or moving backwards.
        if not (is_stopped or current_vel < 0.0):
            self._last_running_s = self._now_s
        stopped_condition = (
            self._last_running_s is not None
            and self._now_s - self._last_running_s
            > config.stopped_state_entry_duration_time
        )
        current_vel_cmd = abs(trajectory[nearest_index].vel)
        emergency_condition = (
            config.enable_overshoot_emergency
            and stop_dist < -config.emergency_state_overshoot_stop_dist
            and current_vel_cmd < _ZERO_VELOCITY
        )
        if self._state != STOPPED:
            self._previous_keep_stopped = None

        if self._state == DRIVE:
            if emergency_condition:
                self._change_state(EMERGENCY, stop_dist)
                return
            if config.enable_smooth_stop:
                if stopping_condition:
                    predicted_stop_dist = (
                        stop_dist
                        - 0.5
                        * (predicted_vel + current_vel)
                        * config.delay_compensation_time
                    )
                    self._smooth_stop_init(predicted_vel, predicted_stop_dist)
                    self._change_state(STOPPING, stop_dist)
            elif stopped_condition and not departure_from_stopped:
                self._change_state(STOPPED, stop_dist)
            return

        if self._state == STOPPING:
            if emergency_condition:
                self._change_state(EMERGENCY, stop_dist)
                return
            if stopped_condition:
                self._change_state(STOPPED, stop_dist)
                return
            if departure_from_stopping:
                self._pid_reset()
                self._lpf_vel_error = 0.0
                # Autoware: "prevent the car from taking a long time to start to move".
                self._previous_raw_acc = max(0.0, self._previous_raw_acc)
                self._change_state(DRIVE, stop_dist)
            return

        if self._state == STOPPED:
            if departure_from_stopped:
                # The steering is taken as converged (no steering angle feedback).
                current_keep_stopped = False
                keep_stopped = self._previous_keep_stopped is None or (
                    current_keep_stopped or bool(self._previous_keep_stopped)
                )
                self._previous_keep_stopped = current_keep_stopped
                if config.enable_keep_stopped_until_steer_convergence and keep_stopped:
                    return
                self._pid_reset()
                self._lpf_vel_error = 0.0
                self._lpf_acc_error = 0.0
                self._change_state(DRIVE, stop_dist)
            return

        # EMERGENCY: under control, it is left only once stopped.
        if stopped_condition:
            self._change_state(STOPPED, stop_dist)

    def _change_state(self, state: int, stop_dist: float) -> None:
        """``changeControlState``, logged."""
        if state == self._state:
            return
        if state == EMERGENCY:
            logger.warning(
                "longitudinal control: emergency stop, %.2f m past the stop point",
                -stop_dist,
            )
        else:
            logger.debug(
                "longitudinal control state %s -> %s (stop distance %.2f m)",
                CONTROL_STATE_NAMES[self._state],
                CONTROL_STATE_NAMES[state],
                stop_dist,
            )
        self._state = state

    def _ctrl_cmd(
        self,
        trajectory: List[_Point],
        target_index: int,
        current_vel: float,
        current_acc: float,
        stop_dist: float,
        dt: float,
    ) -> Tuple[float, float]:
        """``calcCtrlCmd``: ``(acceleration, speed)`` to command, without slope
        compensation (the plan carries no pitch the follower could use)."""
        config = self._config
        target = trajectory[target_index]
        if self._state == STOPPED:
            velocity = config.stopped_vel
            acceleration = config.stopped_acc
            self._previous_raw_vel = 0.0
            self._previous_raw_acc = 0.0
        else:
            raw_vel = target.vel
            raw_acc = target.acc
            if self._state == EMERGENCY:
                emergency = abs(config.emergency_acc)
                raw_vel = _diff_limit(
                    config.emergency_vel,
                    self._previous_raw_vel,
                    dt,
                    emergency,
                    -emergency,
                )
                raw_acc = min(max(config.emergency_acc, config.min_acc), config.max_acc)
                jerk = abs(config.emergency_jerk)
                raw_acc = _diff_limit(raw_acc, self._previous_raw_acc, dt, jerk, -jerk)
            else:
                if self._state == DRIVE:
                    raw_vel = target.vel
                    raw_acc = self._velocity_feedback(
                        trajectory, target_index, current_vel, dt
                    )
                    raw_acc = self._keep_brake_before_stop(
                        trajectory, target_index, raw_acc
                    )
                else:
                    raw_acc = self._smooth_stop(stop_dist, current_vel, current_acc)
                    raw_vel = config.stopped_vel
                raw_acc = min(max(raw_acc, config.min_acc), config.max_acc)
                raw_acc = _diff_limit(
                    raw_acc,
                    self._previous_raw_acc,
                    dt,
                    config.max_jerk,
                    config.min_jerk,
                )
            self._previous_raw_vel = _finite_or(raw_vel, self._previous_raw_vel)
            self._previous_raw_acc = _finite_or(raw_acc, self._previous_raw_acc)
            gain = config.lpf_acc_error_gain
            self._lpf_acc_error = _finite_or(
                gain * self._lpf_acc_error + (1.0 - gain) * (current_acc - raw_acc),
                self._lpf_acc_error,
            )
            acceleration = raw_acc - self._lpf_acc_error * config.acc_feedback_gain
            velocity = raw_vel

        # storeAccelCmd: the commands the delay compensation replays.
        if self._state == DRIVE:
            self._ctrl_cmd_history.append((self._now_s, self._previous_raw_acc))
        else:
            self._ctrl_cmd_history = []
        if len(self._ctrl_cmd_history) > 2 and (
            self._now_s - self._ctrl_cmd_history[1][0] > config.delay_compensation_time
        ):
            self._ctrl_cmd_history.pop(0)

        limit = abs(config.max_acc_cmd_diff)
        acceleration = _diff_limit(acceleration, self._previous_acc, dt, limit, -limit)
        return acceleration, velocity

    def _velocity_feedback(
        self, trajectory: List[_Point], target_index: int, current_vel: float, dt: float
    ) -> float:
        """``applyVelocityFeedback`` (forward gear)."""
        config = self._config
        target = trajectory[target_index]
        if config.enable_velocity_lookahead_feedback:
            horizon = config.velocity_lookahead_time
            weight = config.velocity_lookahead_blend_weight
            lookahead = max(abs(current_vel), 0.1) * horizon
            future = target_index
            travelled = 0.0
            for index in range(target_index, len(trajectory) - 1):
                front = trajectory[index]
                back = trajectory[index + 1]
                travelled += math.sqrt(
                    (back.x - front.x) ** 2
                    + (back.y - front.y) ** 2
                    + (back.z - front.z) ** 2
                )
                future = index + 1
                if travelled >= lookahead:
                    break
            ahead = trajectory[future]
            gain = config.lpf_vel_error_gain
            self._lpf_vel_error = _finite_or(
                gain * self._lpf_vel_error + (1.0 - gain) * (ahead.vel - current_vel),
                self._lpf_vel_error,
            )
            connect = self._lpf_vel_error / horizon
            target_acc = ahead.acc
            # A stop point accelerates by nothing, yet must be braked into.
            if (
                abs(ahead.vel) < config.stopped_state_entry_vel
                and ahead.acc < config.stopped_state_entry_acc
            ):
                target_acc = target.acc
            return weight * connect + (1.0 - weight) * target_acc

        moving = abs(current_vel) > config.current_vel_threshold_pid_integration
        stuck = (
            not moving and self._now_s > config.time_threshold_before_pid_integration
        )
        integrate = moving or (config.enable_integration_at_low_speed and stuck)
        gain = config.lpf_vel_error_gain
        self._lpf_vel_error = _finite_or(
            gain * self._lpf_vel_error + (1.0 - gain) * (target.vel - current_vel),
            self._lpf_vel_error,
        )
        feedback = self._pid_calculate(self._lpf_vel_error, dt, integrate)
        # Feedforward from time to arc length: the plan's acceleration, scaled.
        scale = min(
            max(abs(current_vel) / max(abs(target.vel), 0.1), config.ff_scale_min),
            config.ff_scale_max,
        )
        return target.acc * scale + feedback

    def _pid_calculate(self, error: float, dt: float, integrate: bool) -> float:
        """``PIDController::calculate``."""
        config = self._config
        proportional = min(
            max(config.kp * error, config.min_p_effort), config.max_p_effort
        )
        if integrate:
            integral = self._pid_integral + error * dt
            if config.ki != 0.0:
                integral = min(
                    max(integral, config.min_i_effort / config.ki),
                    config.max_i_effort / config.ki,
                )
            self._pid_integral = _finite_or(integral, self._pid_integral)
        integral = config.ki * self._pid_integral
        differential = 0.0
        if self._pid_first:
            self._pid_first = False
        else:
            differential = (error - self._pid_previous_error) / dt
        derivative = min(
            max(config.kd * differential, config.min_d_effort), config.max_d_effort
        )
        self._pid_previous_error = _finite_or(error, self._pid_previous_error)
        return min(
            max(proportional + integral + derivative, config.min_out), config.max_out
        )

    def _pid_reset(self) -> None:
        """``PIDController::reset``."""
        self._pid_integral = 0.0
        self._pid_previous_error = 0.0
        self._pid_first = True

    def _keep_brake_before_stop(
        self, trajectory: List[_Point], target_index: int, acceleration: float
    ) -> float:
        """``keepBrakeBeforeStop``."""
        config = self._config
        if not config.enable_brake_keeping_before_stop:
            return acceleration
        stop = _zero_velocity_index(trajectory)
        if stop < 0:
            return acceleration
        least = math.inf
        least_index = len(trajectory)
        for index in range(stop, -1, -1):
            if trajectory[index].acc > least:
                break
            least = trajectory[index].acc
            least_index = index
        keep = max(config.brake_keeping_acc, least)
        if target_index >= least_index and acceleration > keep:
            return keep
        return acceleration

    def _smooth_stop_init(
        self, predicted_vel: float, predicted_stop_dist: float
    ) -> None:
        """``SmoothStop::init``."""
        config = self._config
        self._weak_acc_s = self._now_s
        if predicted_stop_dist < _DOUBLE_EPSILON:
            self._strong_acc = config.smooth_stop_min_strong_acc
            return
        strong = -(predicted_vel**2) / (2.0 * predicted_stop_dist)
        self._strong_acc = max(
            min(strong, config.smooth_stop_max_strong_acc),
            config.smooth_stop_min_strong_acc,
        )

    def _time_to_stop(self) -> Optional[float]:
        """``SmoothStop::calcTimeToStop``: when a line through the speed history
        reaches zero, if ahead."""
        history = self._vel_history
        size = float(len(history))
        if size == 0.0:
            return None
        mean_t = 0.0
        mean_v = 0.0
        sum_tv = 0.0
        sum_tt = 0.0
        for stamp, speed in history:
            t = stamp - self._now_s
            mean_t += t / size
            mean_v += speed / size
            sum_tv += t * speed
            sum_tt += t * t
        denominator = size * mean_t * mean_t - sum_tt
        if abs(denominator) < _DOUBLE_EPSILON:
            return None
        slope = (size * mean_t * mean_v - sum_tv) / denominator
        intercept = mean_v - slope * mean_t
        if abs(slope) < _DOUBLE_EPSILON:
            return None
        time_to_stop = -intercept / slope
        if time_to_stop > 0.0:
            return time_to_stop
        return None

    def _smooth_stop(
        self, stop_dist: float, current_vel: float, current_acc: float
    ) -> float:
        """``SmoothStop::calculate``."""
        config = self._config
        time_to_stop = self._time_to_stop()
        is_fast = abs(current_vel) > config.smooth_stop_max_fast_vel
        is_running = (
            abs(current_vel) > config.smooth_stop_min_running_vel
            or abs(current_acc) > config.smooth_stop_min_running_acc
        )
        if stop_dist < config.smooth_stop_strong_stop_dist:
            return config.smooth_stop_strong_stop_acc
        if stop_dist < config.smooth_stop_weak_stop_dist:
            return config.smooth_stop_weak_stop_acc
        if is_running:
            late = (
                time_to_stop is not None
                and time_to_stop
                > config.smooth_stop_weak_stop_time + config.delay_compensation_time
            )
            if late or (time_to_stop is None and is_fast):
                return self._strong_acc
            self._weak_acc_s = self._now_s
            return config.smooth_stop_weak_acc
        if self._now_s - self._weak_acc_s < _SMOOTH_STOP_WEAK_HOLD_S:
            return config.smooth_stop_weak_acc
        return config.smooth_stop_strong_stop_acc

    # ------------------------------------------------------------------
    # Longitudinal "time_indexed"
    # ------------------------------------------------------------------

    def _step_time_indexed(
        self,
        plan_in_local: Trajectory,
        pose_local_to_rig: Pose,
        current_speed_mps: float,
        dt_s: float,
        yaw_rate_rps: Optional[float],
        now_us: Optional[int],
        gear: int,
    ) -> VehicleCommand:
        """:meth:`step` with the ``time_indexed`` longitudinal controller."""
        if not plan_in_local:
            return self._hold_still()

        plan_in_rig = plan_in_local.transform(pose_local_to_rig.inverse())
        points = plan_in_rig.positions

        if _reach_m(points) < self._config.stop_distance_m:
            return self._hold_still()
        target = self._lookahead_point(points, self._lookahead_m(current_speed_mps))
        if target is None:
            return self._hold_still()

        steer = self._lateral(target, current_speed_mps, dt_s, yaw_rate_rps)
        asked, target_speed = self._time_indexed_longitudinal(
            plan_in_rig, now_us, current_speed_mps, dt_s
        )
        throttle, brake = self._pedals(asked, current_speed_mps, gear)

        return VehicleCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            target_speed_mps=target_speed,
            lookahead_lateral_offset_m=float(target[1]),
            acceleration_mps2=asked,
        )

    def _time_indexed_longitudinal(
        self,
        plan_in_rig: Trajectory,
        now_us: Optional[int],
        current_speed_mps: float,
        dt_s: float,
    ) -> Tuple[float, float]:
        """Return ``(asked acceleration, target speed)`` that keep the vehicle on the
        plan in time."""
        config = self._config
        speed, acceleration, gap = _reference(
            plan_in_rig, now_us, config.speed_preview_s
        )
        standing = current_speed_mps < config.standstill_speed_mps
        if standing:
            # The brake has nothing left to take off a standing car: what the
            # integral wound up slowing it down must not hold it once the plan goes.
            self._integral = max(self._integral, 0.0)
        correction = float(
            np.clip(
                config.position_gain * gap,
                -config.max_position_correction_mps,
                config.max_position_correction_mps,
            )
        )
        target = max(0.0, speed + correction)
        error = target - max(0.0, current_speed_mps)
        self._integral = _finite_or(
            float(
                np.clip(
                    self._integral + error * dt_s,
                    -config.integral_limit,
                    config.integral_limit,
                )
            ),
            self._integral,
        )
        asked = (
            acceleration + config.speed_kp * error + config.speed_ki * self._integral
        )
        return asked, target


def _reference(
    plan_in_rig: Trajectory, now_us: Optional[int], preview_s: float
) -> tuple[float, float, float]:
    """Where the plan has the vehicle at *now_us*: ``(speed, acceleration, gap)``.

    The speed is the plan's at that instant, the acceleration its mean over the
    next *preview_s*, and the gap how far along the plan that point is ahead of
    the rig origin (negative behind), in metres.
    """
    times_s = (
        np.asarray(plan_in_rig.timestamps_us, dtype=np.float64)
        - plan_in_rig.timestamps_us[0]
    ) * 1e-6
    points = plan_in_rig.positions[:, :2]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    durations = np.diff(times_s)
    moving = durations > _EPSILON
    if not moving.any():
        return 0.0, 0.0, 0.0
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    middles = (times_s[:-1] + durations / 2.0)[moving]
    speeds = lengths[moving] / durations[moving]

    now_s = 0.0 if now_us is None else (now_us - plan_in_rig.timestamps_us[0]) * 1e-6
    now_s = float(np.clip(now_s, 0.0, times_s[-1]))
    ahead_s = min(now_s + preview_s, float(times_s[-1]))
    speed = float(np.interp(now_s, middles, speeds))
    acceleration = (
        (float(np.interp(ahead_s, middles, speeds)) - speed) / (ahead_s - now_s)
        if ahead_s - now_s > _EPSILON
        else 0.0
    )

    # The rig origin's projection onto the plan, as a distance along it.
    starts, steps = points[:-1], np.diff(points, axis=0)
    squared = np.maximum((steps**2).sum(axis=1), _EPSILON)
    fractions = np.clip((-starts * steps).sum(axis=1) / squared, 0.0, 1.0)
    nearest = starts + fractions[:, None] * steps
    closest = int(np.argmin((nearest**2).sum(axis=1)))
    along = arc[closest] + fractions[closest] * lengths[closest]
    return speed, acceleration, float(np.interp(now_s, times_s, arc)) - along


def _reach_m(points: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> float:
    """How far a plan's ``(N, 3)`` rig-frame points get from the vehicle."""
    if len(points) == 0:
        return 0.0
    return float(np.linalg.norm(points[:, :2], axis=1).max())
