"""Turn a planned trajectory into CARLA vehicle controls.

alpasim's contract stops at the plan: the driver policy returns *where the vehicle should
be*, not throttle and steering.  Upstream that gap is filled by alpasim's vehicle dynamics
service; here -- as in ``carla_driver_interface``'s runtime -- it is filled by a pure
pursuit lateral controller, trimmed by feedback on the measured yaw rate, and a speed
controller tracking the plan in time, feeding ``carla.VehicleControl``.

The speed controller works in accelerations, as Autoware's longitudinal controller does:
the plan's own acceleration plus a PI on the speed it misses, turned into pedals by the
vehicle's powertrain model (:class:`~..utils.powertrain.ChaosPowertrain`), as Autoware's
vehicle interface turns them with its accel and brake maps.

All geometry in this module is in the **rig frame**: x forward, y left, z up,
right-handed.  Steering angles follow the same convention (positive = left), and the flip
to CARLA's left-handed steering (positive = right) happens exactly once, in
:meth:`VehicleCommand.to_carla_control`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from ..utils.powertrain import ChaosPowertrain
from .geometry import Pose, Trajectory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


__all__ = [
    "ControlConfig",
    "TrajectoryFollower",
    "VehicleCommand",
    "steer_angle",
    "steer_command",
]

#: Denominator guard for divisions by elapsed time or lookahead distance.
_EPSILON: float = 1e-6


@dataclass(frozen=True)
class ControlConfig:
    """Gains and limits for :class:`TrajectoryFollower`.

    Defaults mirror ``carla_driver_interface``'s ``ControlConfig`` so a policy tuned
    against that runtime behaves the same here.
    """

    # -- Lateral (pure pursuit) ------------------------------------------------
    lookahead_gain_s: float = 0.6
    """Lookahead distance per unit speed, in seconds."""

    min_lookahead_m: float = 4.0
    """Lower bound on the lookahead distance."""

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

    # -- Longitudinal -----------------------------------------------------------
    # The plan is tracked in time: where the plan has the vehicle now, how fast and
    # how hard it is speeding up there, with a PI on what the vehicle misses of that,
    # all in m/s²; the vehicle's powertrain model turns the sum into pedals.  The
    # policy reads the vehicle's speed and acceleration back, so a follower that over-
    # or undershoots the plan is planned onward -- a runaway or a crawl.
    speed_kp: float = 1.0
    """m/s² asked per m/s the vehicle is slower (faster) than the plan."""

    speed_ki: float = 0.1
    """m/s² per m·s of accumulated speed error."""

    integral_limit: float = 3.0
    """Anti-windup bound on the accumulated speed error, m (``speed_ki`` times it
    bounds the integral's share of the asked acceleration)."""

    position_gain: float = 0.5
    """m/s of extra target speed per metre the vehicle is behind the plan (less
    when ahead).  With :attr:`speed_kp` it makes the gap close like a spring of
    ``sqrt(speed_kp * position_gain)`` rad/s, damped ``sqrt(speed_kp / position_gain) / 2``."""

    max_position_correction_mps: float = 6.0
    """Bound on that correction, m/s."""

    speed_preview_s: float = 1.0
    """Over how much of the plan, from now, its acceleration is read, in seconds.

    Not its first segment alone: a plan pulling away from a standstill starts at a
    crawl, a few millimetres in its first 0.1 s.
    """

    throttle_deadband: float = 0.25
    """The least throttle a standing vehicle (below :attr:`standstill_speed_mps`) is
    asked to speed up with.

    A standing Lincoln MKZ on flat road (CARLA 0.10) stays put at 0.2 and pulls
    away at 0.25, where its powertrain model has it already pulling away at 0.2.
    A car with an automatic gearbox creeps off the brake, and the policy's plans
    from a standstill assume it: a few cm/s in the first second, some metres in
    six.  Tracked as asked, that is a throttle short of moving the car, so it would
    never leave.  ``0`` sets no floor.
    """

    standstill_speed_mps: float = 0.5
    """Below this speed the vehicle counts as standing, for :attr:`throttle_deadband`."""

    stop_distance_m: float = 0.5
    """A plan travelling less than this over its whole horizon is a request to stand
    still, and the vehicle is held on the brake.

    Decided by distance, not by the target speed, as Autoware's longitudinal
    controller decides its stop state: a plan pulling away from a standstill asks
    for almost no speed in its first second yet goes metres over its horizon, and
    holding the vehicle there would keep it on the line for good.
    """

    stop_brake: float = 0.6
    """Brake applied when holding still."""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ControlConfig":
        """Return gains built from a plain mapping (e.g. a Hydra node).

        Every ``*_rad`` field is also accepted as ``*_deg`` (``max_steer_angle_deg``,
        ``yaw_rate_trim_limit_deg``), since degrees read better in YAML.

        Raises:
            ValueError: If *mapping* holds a key this config does not define.
        """
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
        return cls(**values)


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
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    target_speed_mps: float = 0.0
    lookahead_lateral_offset_m: float = 0.0

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


class TrajectoryFollower:
    """Tracks a planned trajectory with pure pursuit plus a speed controller.

    One instance drives one vehicle; call :meth:`reset` when the plan's provenance
    changes (e.g. a new session).

    Args:
        config: Gains and limits.  ``None`` uses the defaults.
        powertrain: The vehicle's powertrain model, which turns an acceleration into
            pedals.  May be set later (:attr:`powertrain`), before the first plan is
            tracked.
    """

    def __init__(
        self,
        config: Optional[ControlConfig] = None,
        powertrain: Optional[ChaosPowertrain] = None,
    ) -> None:
        self._config = config or ControlConfig()
        self.powertrain = powertrain
        self.reset()

    @property
    def config(self) -> ControlConfig:
        """Return the controller configuration."""
        return self._config

    def reset(self) -> None:
        """Clear the integral term and the steering state."""
        self._integral = 0.0
        self._previous_steer = 0.0
        self._yaw_rate_trim = 0.0
        self._asked_curvature: Optional[float] = None

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
    ) -> VehicleCommand:
        """Return the command that tracks *plan_in_local* from the current pose.

        Args:
            plan_in_local: The driver's plan, in the local frame.
            pose_local_to_rig: The ego's current pose in the local frame.
            current_speed_mps: Measured ground speed.
            dt_s: Time since the previous call, in seconds.
            yaw_rate_rps: The ego's measured yaw rate, positive to the left.  ``None``
                steers on pure pursuit alone.
            now_us: The current time, on the plan's clock.  ``None`` takes the
                plan's first instant as now.
            gear: The gear the vehicle is in, as ``carla.VehicleControl.gear``
                reports it (``0``, neutral, is taken as the first gear a throttle
                selects).

        Raises:
            RuntimeError: If no :attr:`powertrain` has been set.

        Returns:
            The actuation command.  An empty plan yields a full stop.
        """
        if not plan_in_local:
            return self._hold_still()

        plan_in_rig = plan_in_local.transform(pose_local_to_rig.inverse())
        points = plan_in_rig.positions

        if _travel_m(points) < self._config.stop_distance_m:
            return self._hold_still()
        lookahead = float(
            np.clip(
                self._config.lookahead_gain_s * max(current_speed_mps, 0.0),
                self._config.min_lookahead_m,
                self._config.max_lookahead_m,
            )
        )
        target = self._lookahead_point(points, lookahead)
        if target is None:
            return self._hold_still()

        steer = self._lateral(target, current_speed_mps, dt_s, yaw_rate_rps)
        throttle, brake, target_speed = self._longitudinal(
            plan_in_rig, now_us, current_speed_mps, dt_s, gear
        )

        return VehicleCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            target_speed_mps=target_speed,
            lookahead_lateral_offset_m=float(target[1]),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hold_still(self) -> VehicleCommand:
        """Return a braking command and reset the controller state."""
        self.reset()
        return VehicleCommand(throttle=0.0, steer=0.0, brake=self._config.stop_brake)

    @staticmethod
    def _lookahead_point(
        points: np.ndarray, lookahead_m: float
    ) -> Optional[np.ndarray]:
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
        target: np.ndarray,
        speed_mps: float,
        dt_s: float,
        yaw_rate_rps: Optional[float],
    ) -> float:
        """Return the normalised steering command for *target*, rate limited."""
        distance = max(float(np.linalg.norm(target[:2])), _EPSILON)
        alpha = math.atan2(float(target[1]), float(target[0]))
        curvature = 2.0 * math.sin(alpha) / distance
        angle = math.atan(self._config.wheelbase_m * curvature)
        if yaw_rate_rps is not None:
            angle += self._yaw_rate_feedback(speed_mps, yaw_rate_rps, curvature, dt_s)

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
        asked, self._asked_curvature = self._asked_curvature, curvature
        if asked is None or speed_mps < config.yaw_rate_min_speed_mps:
            return self._yaw_rate_trim
        # The yaw-rate error as a steering angle: what the bicycle model says it would
        # take to turn by the missing yaw rate.
        error = config.wheelbase_m * (asked - yaw_rate_rps / speed_mps)
        limit = config.yaw_rate_trim_limit_rad
        self._yaw_rate_trim = float(
            np.clip(
                self._yaw_rate_trim + config.yaw_rate_ki * error * dt_s, -limit, limit
            )
        )
        return self._yaw_rate_trim

    def _longitudinal(
        self,
        plan_in_rig: Trajectory,
        now_us: Optional[int],
        current_speed_mps: float,
        dt_s: float,
        gear: int,
    ) -> tuple[float, float, float]:
        """Return ``(throttle, brake, target speed)`` that keep the vehicle on the plan."""
        if self.powertrain is None:
            raise RuntimeError(
                "TrajectoryFollower has no powertrain to turn accelerations into pedals"
            )
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
        self._integral = float(
            np.clip(
                self._integral + error * dt_s,
                -config.integral_limit,
                config.integral_limit,
            )
        )
        asked = (
            acceleration + config.speed_kp * error + config.speed_ki * self._integral
        )
        throttle, brake = self.powertrain.pedals(asked, current_speed_mps, gear)
        if standing and asked > 0.0:
            throttle = max(throttle, config.throttle_deadband)
        return throttle, brake, target


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


def _travel_m(points: np.ndarray) -> float:
    """How far a plan's ``(N, 3)`` points travel, end to end along the path."""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())
