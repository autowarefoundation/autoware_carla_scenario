"""Turn a planned trajectory into CARLA vehicle controls.

alpasim's contract stops at the plan: the driver policy returns *where the vehicle should
be*, not throttle and steering.  Upstream that gap is filled by alpasim's vehicle dynamics
service; here -- as in ``carla_driver_interface``'s runtime -- it is filled by a pure
pursuit lateral controller, trimmed by feedback on the measured yaw rate, and a PID
longitudinal controller feeding ``carla.VehicleControl``.

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

    # -- Longitudinal (PID) ----------------------------------------------------
    speed_kp: float = 0.16
    """Proportional gain on speed error, in pedal per m/s.

    The target speed is read :attr:`speed_preview_s` ahead, so the error is the
    acceleration the plan asks for times that preview; a full pedal moves a
    Lincoln MKZ at 30 km/h by about 6 m/s² either way (CARLA 0.10), so reaching the
    previewed speed in time takes about ``1 / (6 * speed_preview_s)``.  A larger
    gain answers a plan easing off by 2 m/s with a full brake: the car stops
    hard, and a policy reading its own acceleration back plans the stop onward.
    """

    speed_ki: float = 0.15
    """Integral gain on speed error."""

    speed_kd: float = 0.05
    """Derivative gain, on the measured speed rather than the error: a new plan
    moving the target is a step in the error, not something to brake against."""

    integral_limit: float = 1.0
    """Clamp on the integral term, preventing wind-up while braking."""

    speed_preview_s: float = 1.0
    """How far into the plan the target speed is read, in seconds.

    The plan's speed at this instant, not at its start: a plan pulling away from
    a standstill starts at a crawl, and read at its first segment it says "stay".
    """

    throttle_deadband: float = 0.2
    """Throttle below which a standing vehicle does not move at all; a positive
    command to a standing vehicle is mapped onto ``[throttle_deadband, 1]``.

    The PID asks for acceleration-like effort, but CARLA's throttle has a dead band:
    a standing Lincoln MKZ on flat road (CARLA 0.10) stays put at 0.2 and pulls away
    at 0.25.  Without the offset a gentle pull-away -- a target of a few cm/s --
    commands a throttle that moves nothing, and the car never leaves.  Only from a
    standstill (below :attr:`standstill_speed_mps`): a rolling MKZ holds 30 km/h at
    about 0.1 and gains speed at 0.2, so offsetting every command would turn the
    smallest correction into a surge.  ``0`` passes the PID output through.
    """

    standstill_speed_mps: float = 0.5
    """Below this speed the vehicle counts as standing, for :attr:`throttle_deadband`."""

    brake_deadband: float = 0.05
    """PID effort below which a slowing command coasts rather than brakes; past it,
    the brake is the effort less this.

    CARLA's brake is strong (0.3 for one tick takes about 0.1 m/s off 30 km/h)
    while a coasting MKZ loses only about 0.2 m/s each second, so braking for every
    few tenths of a m/s over the target jolts the car -- and a policy that reads
    its own acceleration back plans the jolt onward.
    """

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
    """Tracks a planned trajectory with pure pursuit plus a speed PID.

    One instance drives one vehicle; call :meth:`reset` when the plan's provenance
    changes (e.g. a new session).

    Args:
        config: Gains and limits.  ``None`` uses the defaults.
    """

    def __init__(self, config: Optional[ControlConfig] = None) -> None:
        self._config = config or ControlConfig()
        self.reset()

    @property
    def config(self) -> ControlConfig:
        """Return the controller configuration."""
        return self._config

    def reset(self) -> None:
        """Clear the integral term, the derivative memory, and the steering state."""
        self._integral = 0.0
        self._previous_speed: Optional[float] = None
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
    ) -> VehicleCommand:
        """Return the command that tracks *plan_in_local* from the current pose.

        Args:
            plan_in_local: The driver's plan, in the local frame.
            pose_local_to_rig: The ego's current pose in the local frame.
            current_speed_mps: Measured ground speed.
            dt_s: Time since the previous call, in seconds.
            yaw_rate_rps: The ego's measured yaw rate, positive to the left.  ``None``
                steers on pure pursuit alone.

        Returns:
            The actuation command.  An empty plan yields a full stop.
        """
        if not plan_in_local:
            return self._hold_still()

        plan_in_rig = plan_in_local.transform(pose_local_to_rig.inverse())
        points = plan_in_rig.positions

        if _travel_m(points) < self._config.stop_distance_m:
            return self._hold_still()
        target_speed = self._target_speed(plan_in_rig)
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
        throttle, brake = self._longitudinal(target_speed, current_speed_mps, dt_s)

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

    def _target_speed(self, plan_in_rig: Trajectory) -> float:
        """Return the speed the plan asks for :attr:`ControlConfig.speed_preview_s` ahead.

        A plan carries positions and timestamps but no explicit speed, so the speed is
        read off the segment spanning the preview instant.  Not the first segment: a
        plan pulling away from a standstill starts at a crawl (a few millimetres in
        its first 0.1 s).  Looking ahead also starts braking for a stop the plan
        reaches within the preview, as a driver would.
        """
        if len(plan_in_rig) < 2:
            return 0.0
        times_s = (
            np.asarray(plan_in_rig.timestamps_us, dtype=np.float64)
            - plan_in_rig.timestamps_us[0]
        ) * 1e-6
        if times_s[-1] < _EPSILON:
            return 0.0
        preview = min(self._config.speed_preview_s, float(times_s[-1]))
        index = int(np.clip(np.searchsorted(times_s, preview), 1, len(times_s) - 1))
        duration_s = times_s[index] - times_s[index - 1]
        if duration_s < _EPSILON:
            return 0.0
        positions = plan_in_rig.positions
        return (
            float(np.linalg.norm(positions[index, :2] - positions[index - 1, :2]))
            / duration_s
        )

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
        self, target_speed_mps: float, current_speed_mps: float, dt_s: float
    ) -> tuple[float, float]:
        """Return ``(throttle, brake)`` for the requested speed."""
        error = target_speed_mps - current_speed_mps
        if dt_s > _EPSILON:
            self._integral = float(
                np.clip(
                    self._integral + error * dt_s,
                    -self._config.integral_limit,
                    self._config.integral_limit,
                )
            )
            # On the measured speed: the error's own derivative would kick the
            # output whenever a new plan moves the target.  None on the first step.
            derivative = (
                0.0
                if self._previous_speed is None
                else (self._previous_speed - current_speed_mps) / dt_s
            )
        else:
            derivative = 0.0
        self._previous_speed = current_speed_mps

        output = (
            self._config.speed_kp * error
            + self._config.speed_ki * self._integral
            + self._config.speed_kd * derivative
        )
        if output > 0.0:
            if current_speed_mps >= self._config.standstill_speed_mps:
                return float(min(output, 1.0)), 0.0
            deadband = self._config.throttle_deadband
            return deadband + (1.0 - deadband) * float(min(output, 1.0)), 0.0
        return 0.0, float(np.clip(-output - self._config.brake_deadband, 0.0, 1.0))


def _travel_m(points: np.ndarray) -> float:
    """How far a plan's ``(N, 3)`` points travel, end to end along the path."""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())
