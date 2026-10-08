"""Unit tests for the pure-pursuit + PID trajectory follower."""

from __future__ import annotations

import math

import pytest

from autoware_carla_scenario.driver.control import (
    ControlConfig,
    TrajectoryFollower,
    VehicleCommand,
    steer_angle,
    steer_command,
)
from autoware_carla_scenario.driver.geometry import Pose, Trajectory


_DT_S = 0.05
_STEP_US = 100_000


def _plan(points, *, speed_mps: float = 8.0) -> Trajectory:
    """Return a local-frame plan through *points* at a constant *speed_mps*.

    Timestamps are derived from the requested speed so that the follower's target speed
    comes out as *speed_mps*.
    """
    plan = Trajectory.empty()
    previous = None
    timestamp = 0
    for x, y in points:
        if previous is not None:
            distance = math.hypot(x - previous[0], y - previous[1])
            timestamp += int(round(distance / speed_mps * 1e6))
        plan.append(timestamp, Pose.from_xyz_yaw(x, y, 0.0, 0.0))
        previous = (x, y)
    return plan


def _straight(speed_mps: float = 8.0) -> Trajectory:
    return _plan([(step * 2.0, 0.0) for step in range(21)], speed_mps=speed_mps)


# ---------------------------------------------------------------------------
# Lateral control
# ---------------------------------------------------------------------------


def test_straight_plan_steers_straight() -> None:
    follower = TrajectoryFollower()
    command = follower.step(_straight(), Pose.identity(), 8.0, _DT_S)
    assert command.steer == pytest.approx(0.0, abs=1e-9)


def test_plan_to_the_left_steers_left() -> None:
    """The rig frame is right-handed, CARLA steers positive to the right."""
    follower = TrajectoryFollower()
    plan = _plan([(step * 2.0, step * 0.6) for step in range(21)])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer < 0.0


def test_plan_to_the_right_steers_right() -> None:
    follower = TrajectoryFollower()
    plan = _plan([(step * 2.0, -step * 0.6) for step in range(21)])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer > 0.0


def test_steering_is_rate_limited() -> None:
    """A hard turn cannot exceed max_steer_rate * dt in a single tick."""
    config = ControlConfig(max_steer_rate=1.0)
    follower = TrajectoryFollower(config)
    plan = _plan([(step * 1.0, -step * 3.0) for step in range(21)])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer == pytest.approx(config.max_steer_rate * _DT_S, abs=1e-9)


def test_steering_accounts_for_the_ego_pose() -> None:
    """A plan that is straight in local coordinates curves once the ego is yawed."""
    follower = TrajectoryFollower()
    yawed = Pose.from_xyz_yaw(0.0, 0.0, 0.0, math.radians(-20.0))
    command = follower.step(_straight(), yawed, 8.0, _DT_S)
    # The path now runs off to the ego's left, so the command steers left.
    assert command.steer < 0.0


def test_points_behind_the_vehicle_are_ignored() -> None:
    """A plan starting behind the rig origin must not fold the steering backwards."""
    follower = TrajectoryFollower()
    plan = _plan([(-4.0, 0.0), *[(step * 2.0, 0.0) for step in range(1, 21)]])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer == pytest.approx(0.0, abs=1e-9)


def _arc(radius_m: float, speed_mps: float) -> Trajectory:
    """A left-hand arc of *radius_m* starting at the rig origin, heading +x."""
    angles = [step * 2.0 / radius_m for step in range(21)]
    return _plan(
        [(radius_m * math.sin(a), radius_m * (1.0 - math.cos(a))) for a in angles],
        speed_mps=speed_mps,
    )


def test_steer_map_inverts_carla_quadratic_response() -> None:
    """CARLA 0.10 turns the wheels by about 56 deg * steer**2."""
    config = ControlConfig()
    # 5 degrees needs steer 0.3 on the MKZ; a linear map over 70 would send 0.07.
    assert steer_command(math.radians(5.0), config) == pytest.approx(0.3, abs=0.01)
    for angle in (-0.6, -0.1, 0.0, 0.02, 0.4):
        assert steer_angle(steer_command(angle, config), config) == pytest.approx(
            angle, abs=1e-9
        )
    assert steer_command(math.radians(90.0), config) == 1.0
    linear = ControlConfig(max_steer_angle_rad=math.radians(70.0), steer_exponent=1.0)
    assert steer_command(math.radians(7.0), linear) == pytest.approx(0.1)


def test_pure_pursuit_recovers_the_geometric_steering_angle() -> None:
    radius = 40.0
    config = ControlConfig(
        min_lookahead_m=4.0, max_lookahead_m=4.0, max_steer_rate=100.0
    )
    command = TrajectoryFollower(config).step(
        _arc(radius, 8.0), Pose.identity(), 8.0, _DT_S
    )
    expected = math.atan(config.wheelbase_m / radius)
    assert -steer_angle(command.steer, config) == pytest.approx(expected, rel=0.1)


def _yaw_rate_trace(
    plant, radius_m: float = 60.0, speed_mps: float = 13.0, steps: int = 80
):
    """Yaw rate (rad/s) per tick, closing the loop through *plant*.

    *plant* maps the steering angle the follower commands to the one the vehicle
    actually turns on; the measured yaw rate answers one tick later, as in CARLA.
    """
    config = ControlConfig()
    follower = TrajectoryFollower(config)
    plan = _arc(radius_m, speed_mps)
    yaw_rate, trace = 0.0, []
    for _ in range(steps):
        command = follower.step(
            plan, Pose.identity(), speed_mps, _DT_S, yaw_rate_rps=yaw_rate
        )
        wheel_angle = plant(-steer_angle(command.steer, config))
        yaw_rate = speed_mps * math.tan(wheel_angle) / config.wheelbase_m
        trace.append(yaw_rate)
    return trace


def test_yaw_rate_feedback_makes_a_weakly_steering_vehicle_turn() -> None:
    """A vehicle turning half as much as the map promises still gets its yaw rate."""
    radius, speed = 60.0, 13.0
    asked = _yaw_rate_trace(lambda angle: angle, radius, speed)[-1]
    open_loop = TrajectoryFollower(ControlConfig(yaw_rate_ki=0.0, max_steer_rate=100.0))
    command = open_loop.step(_arc(radius, speed), Pose.identity(), speed, _DT_S)
    weak = 0.5 * -steer_angle(command.steer, open_loop.config)
    assert speed * math.tan(weak) / open_loop.config.wheelbase_m < 0.6 * asked

    trace = _yaw_rate_trace(lambda angle: 0.5 * angle, radius, speed)
    assert trace[-1] == pytest.approx(asked, rel=0.02)


def test_yaw_rate_feedback_does_not_disturb_a_vehicle_that_steers_as_modelled() -> None:
    """The yaw rate is held against the curvature it answers, so no trim builds up."""
    follower = TrajectoryFollower()
    plan = _arc(60.0, 13.0)
    follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=0.0)
    curvature = follower._asked_curvature  # noqa: SLF001
    assert curvature is not None
    asked = 13.0 * curvature
    follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=asked)
    assert follower._yaw_rate_trim == pytest.approx(0.0, abs=1e-12)  # noqa: SLF001


def test_yaw_rate_trim_is_held_at_a_standstill() -> None:
    """At walking pace the yaw rate says nothing about the steering."""
    follower = TrajectoryFollower()
    plan = _arc(30.0, 8.0)
    follower.step(plan, Pose.identity(), 8.0, _DT_S, yaw_rate_rps=0.0)
    follower.step(plan, Pose.identity(), 8.0, _DT_S, yaw_rate_rps=0.0)
    trim = follower._yaw_rate_trim  # noqa: SLF001
    assert trim != 0.0
    follower.step(plan, Pose.identity(), 0.2, _DT_S, yaw_rate_rps=0.0)
    assert follower._yaw_rate_trim == trim  # noqa: SLF001


# ---------------------------------------------------------------------------
# Longitudinal control
# ---------------------------------------------------------------------------


def test_below_target_speed_opens_the_throttle() -> None:
    follower = TrajectoryFollower()
    command = follower.step(_straight(speed_mps=10.0), Pose.identity(), 1.0, _DT_S)
    assert command.throttle > 0.0
    assert command.brake == pytest.approx(0.0)
    assert command.target_speed_mps == pytest.approx(10.0, rel=1e-3)


def test_above_target_speed_brakes() -> None:
    follower = TrajectoryFollower()
    command = follower.step(_straight(speed_mps=2.0), Pose.identity(), 15.0, _DT_S)
    assert command.brake > 0.0
    assert command.throttle == pytest.approx(0.0)


def test_a_plan_pulling_away_from_a_standstill_is_not_a_stop() -> None:
    """A plan that crawls for its first tenth of a second still asks to move.

    Read at its first segment (a few millimetres in 0.1 s) it is below the stop
    speed, and the vehicle would be held on the brake forever.
    """
    plan = Trajectory.empty()
    for index in range(31):
        t_s = index * 0.1
        plan.append(
            index * _STEP_US, Pose.from_xyz_yaw(0.5 * 0.4 * t_s * t_s, 0.0, 0.0, 0.0)
        )
    command = TrajectoryFollower().step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.target_speed_mps == pytest.approx(0.4 * 0.95, rel=0.05)
    assert command.throttle > 0.0 and command.brake == pytest.approx(0.0)


def test_a_plan_that_crawls_then_goes_is_driven_not_held() -> None:
    """Standing still is decided by how far the plan goes, not by its first second.

    OnePlanner's plan from a standstill on an open road: 4 cm in the first second,
    8 m in six.  Its speed a second ahead is a crawl, but it is a plan to go.
    """
    plan = Trajectory.empty()
    for index in range(61):
        t_s = index * 0.1
        plan.append(
            index * _STEP_US, Pose.from_xyz_yaw(8.4 * (t_s / 6.0) ** 3, 0.0, 0.0, 0.0)
        )
    command = TrajectoryFollower().step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.target_speed_mps < 0.2
    assert command.throttle > 0.0 and command.brake == pytest.approx(0.0)


def test_any_throttle_starts_past_the_dead_band() -> None:
    """A gentle pull-away must still command a throttle that moves the car."""
    config = ControlConfig(throttle_deadband=0.2)
    command = TrajectoryFollower(config).step(
        _straight(speed_mps=0.1), Pose.identity(), 0.0, _DT_S
    )
    assert 0.2 < command.throttle < 0.4
    full = TrajectoryFollower(config).step(
        _straight(speed_mps=30.0), Pose.identity(), 0.0, _DT_S
    )
    assert full.throttle == pytest.approx(1.0)


def test_a_rolling_car_gets_no_dead_band_offset() -> None:
    """Rolling, a small correction is a small throttle: the offset is for pull-aways."""
    command = TrajectoryFollower().step(
        _straight(speed_mps=8.4), Pose.identity(), 8.33, _DT_S
    )
    assert 0.0 < command.throttle < 0.1


def test_a_little_over_the_target_coasts() -> None:
    """A few tenths of a m/s over the plan lets the car roll off, not brake."""
    command = TrajectoryFollower().step(
        _straight(speed_mps=8.2), Pose.identity(), 8.33, _DT_S
    )
    assert command.throttle == pytest.approx(0.0)
    assert command.brake == pytest.approx(0.0)


def test_a_moved_target_is_no_derivative_kick() -> None:
    """The derivative is on the measured speed: neither the first step nor a new
    plan asking for less, at an unchanged speed, moves the output through it."""
    config = ControlConfig(speed_kp=0.0, speed_ki=0.0, speed_kd=1.0, brake_deadband=0.0)
    follower = TrajectoryFollower(config)
    for target_mps in (8.0, 6.0):
        command = follower.step(_straight(speed_mps=target_mps), Pose.identity(), 8.0, _DT_S)
        assert command.brake == pytest.approx(0.0)
        assert command.throttle == pytest.approx(0.0)


def test_a_car_losing_speed_is_pushed_back_by_the_derivative() -> None:
    """At an unchanged target, the derivative answers the speed falling with throttle."""
    config = ControlConfig(speed_kp=0.0, speed_ki=0.0, speed_kd=1.0, throttle_deadband=0.0)
    follower = TrajectoryFollower(config)
    follower.step(_straight(speed_mps=8.0), Pose.identity(), 8.0, _DT_S)
    command = follower.step(_straight(speed_mps=8.0), Pose.identity(), 7.9, _DT_S)
    assert command.throttle > 0.0


def test_a_plan_easing_off_brakes_gently() -> None:
    """A plan 2 m/s slower a second ahead asks for about 2 m/s², not a full brake."""
    follower = TrajectoryFollower()
    command = follower.step(_straight(speed_mps=6.0), Pose.identity(), 8.0, _DT_S)
    assert 0.0 < command.brake < 0.5


def test_a_plan_stopping_within_the_preview_brakes_early() -> None:
    """A plan reaching its stop within a second asks for (almost) nothing now."""
    plan = _plan([(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)], speed_mps=5.0)
    plan.append(
        plan.timestamps_us[-1] + 2_000_000, Pose.from_xyz_yaw(1.5, 0.0, 0.0, 0.0)
    )
    command = TrajectoryFollower().step(plan, Pose.identity(), 5.0, _DT_S)
    assert command.target_speed_mps == pytest.approx(0.0)
    assert command.brake > 0.0


def test_stationary_plan_holds_the_vehicle() -> None:
    """A plan that does not advance is a request to stand still."""
    config = ControlConfig()
    follower = TrajectoryFollower(config)
    plan = Trajectory.empty()
    for index in range(5):
        plan.append(index * _STEP_US, Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0))
    command = follower.step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.throttle == pytest.approx(0.0)
    assert command.brake == pytest.approx(config.stop_brake)


def test_empty_plan_brakes_and_resets() -> None:
    config = ControlConfig()
    follower = TrajectoryFollower(config)
    follower.step(_straight(), Pose.identity(), 1.0, _DT_S)
    command = follower.step(Trajectory.empty(), Pose.identity(), 5.0, _DT_S)
    assert command.brake == pytest.approx(config.stop_brake)
    assert command.steer == pytest.approx(0.0)


def test_integral_term_is_clamped() -> None:
    """Sustained error must not wind the integral term past its limit."""
    config = ControlConfig(
        speed_kp=0.0,
        speed_kd=0.0,
        speed_ki=1.0,
        integral_limit=0.5,
        throttle_deadband=0.0,
    )
    follower = TrajectoryFollower(config)
    plan = _straight(speed_mps=10.0)
    for _ in range(200):
        command = follower.step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.throttle == pytest.approx(0.5, abs=1e-6)


def test_reset_clears_controller_state() -> None:
    config = ControlConfig(
        speed_kp=0.0, speed_kd=0.0, speed_ki=1.0, throttle_deadband=0.0
    )
    follower = TrajectoryFollower(config)
    plan = _straight(speed_mps=10.0)
    for _ in range(10):
        follower.step(plan, Pose.identity(), 0.0, _DT_S)
    follower.reset()
    command = follower.step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.throttle == pytest.approx(1.0 * 10.0 * _DT_S, abs=1e-6)


# ---------------------------------------------------------------------------
# Command conversion
# ---------------------------------------------------------------------------


def test_command_defaults_are_inert() -> None:
    command = VehicleCommand()
    assert command.throttle == 0.0
    assert command.brake == 0.0
    assert command.hand_brake is False
    assert command.reverse is False
