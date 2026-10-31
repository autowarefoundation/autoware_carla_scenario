"""Unit tests for the trajectory follower: pure pursuit, and the longitudinal
controllers (Autoware's pid, and time_indexed)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from autoware_carla_scenario.driver.control import (
    DRIVE,
    EMERGENCY,
    STOPPED,
    STOPPING,
    ControlConfig,
    TrajectoryFollower,
    VehicleCommand,
    plan_velocities,
    steer_angle,
    steer_command,
)
from autoware_carla_scenario.driver.geometry import Pose, Trajectory
from autoware_carla_scenario.utils.powertrain import CREEP_THROTTLE

from ._vehicle_physics import mkz_powertrain


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


def _follower(config: ControlConfig | None = None) -> TrajectoryFollower:
    """A follower driving a Lincoln MKZ."""
    return TrajectoryFollower(config, mkz_powertrain())


def _asked(command: VehicleCommand, speed_mps: float) -> float:
    """The acceleration *command*'s pedals settle at, in first."""
    return mkz_powertrain().acceleration(speed_mps, command.throttle, command.brake, 1)


def _straight(speed_mps: float = 8.0) -> Trajectory:
    return _plan([(step * 2.0, 0.0) for step in range(21)], speed_mps=speed_mps)


# ---------------------------------------------------------------------------
# Lateral control
# ---------------------------------------------------------------------------


def test_straight_plan_steers_straight() -> None:
    follower = _follower()
    command = follower.step(_straight(), Pose.identity(), 8.0, _DT_S)
    assert command.steer == pytest.approx(0.0, abs=1e-9)


def test_plan_to_the_left_steers_left() -> None:
    """The rig frame is right-handed, CARLA steers positive to the right."""
    follower = _follower()
    plan = _plan([(step * 2.0, step * 0.6) for step in range(21)])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer < 0.0


def test_plan_to_the_right_steers_right() -> None:
    follower = _follower()
    plan = _plan([(step * 2.0, -step * 0.6) for step in range(21)])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer > 0.0


def test_steering_is_rate_limited() -> None:
    """A hard turn cannot exceed max_steer_rate * dt in a single tick."""
    config = ControlConfig(max_steer_rate=1.0)
    follower = _follower(config)
    plan = _plan([(step * 1.0, -step * 3.0) for step in range(21)])
    command = follower.step(plan, Pose.identity(), 8.0, _DT_S)
    assert command.steer == pytest.approx(config.max_steer_rate * _DT_S, abs=1e-9)


def test_steering_accounts_for_the_ego_pose() -> None:
    """A plan that is straight in local coordinates curves once the ego is yawed."""
    follower = _follower()
    yawed = Pose.from_xyz_yaw(0.0, 0.0, 0.0, math.radians(-20.0))
    command = follower.step(_straight(), yawed, 8.0, _DT_S)
    # The path now runs off to the ego's left, so the command steers left.
    assert command.steer < 0.0


def test_points_behind_the_vehicle_are_ignored() -> None:
    """A plan starting behind the rig origin must not fold the steering backwards."""
    follower = _follower()
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
    command = _follower(config).step(_arc(radius, 8.0), Pose.identity(), 8.0, _DT_S)
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
    follower = _follower(config)
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
    open_loop = _follower(ControlConfig(yaw_rate_ki=0.0, max_steer_rate=100.0))
    command = open_loop.step(_arc(radius, speed), Pose.identity(), speed, _DT_S)
    weak = 0.5 * -steer_angle(command.steer, open_loop.config)
    assert speed * math.tan(weak) / open_loop.config.wheelbase_m < 0.6 * asked

    trace = _yaw_rate_trace(lambda angle: 0.5 * angle, radius, speed)
    assert trace[-1] == pytest.approx(asked, rel=0.02)


def test_yaw_rate_feedback_does_not_disturb_a_vehicle_that_steers_as_modelled() -> None:
    """The yaw rate is held against the curvature it answers, so no trim builds up."""
    follower = _follower()
    plan = _arc(60.0, 13.0)
    follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=0.0)
    curvature = follower._asked_curvature  # noqa: SLF001
    assert curvature is not None
    asked = 13.0 * curvature
    follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=asked)
    assert follower._yaw_rate_trim == pytest.approx(0.0, abs=1e-12)  # noqa: SLF001


def test_yaw_rate_trim_is_held_at_a_standstill() -> None:
    """At walking pace the yaw rate says nothing about the steering."""
    follower = _follower()
    plan = _arc(30.0, 8.0)
    follower.step(plan, Pose.identity(), 8.0, _DT_S, yaw_rate_rps=0.0)
    follower.step(plan, Pose.identity(), 8.0, _DT_S, yaw_rate_rps=0.0)
    trim = follower._yaw_rate_trim  # noqa: SLF001
    assert trim != 0.0
    follower.step(plan, Pose.identity(), 0.2, _DT_S, yaw_rate_rps=0.0)
    assert follower._yaw_rate_trim == trim  # noqa: SLF001


# ---------------------------------------------------------------------------
# Longitudinal control, time_indexed: the plan tracked on its own clock
# ---------------------------------------------------------------------------

_TIMED = ControlConfig(longitudinal_controller="time_indexed", throttle_deadband=0.25)


def _timed(config: ControlConfig | None = None) -> TrajectoryFollower:
    """A follower with the time_indexed longitudinal controller, driving an MKZ."""
    return TrajectoryFollower(_TIMED if config is None else config, mkz_powertrain())


def test_below_target_speed_opens_the_throttle() -> None:
    follower = _timed()
    command = follower.step(_straight(speed_mps=10.0), Pose.identity(), 1.0, _DT_S)
    assert command.throttle > 0.0
    assert command.brake == pytest.approx(0.0)
    assert command.target_speed_mps == pytest.approx(10.0, rel=1e-3)


def test_above_target_speed_brakes() -> None:
    follower = _timed()
    command = follower.step(_straight(speed_mps=2.0), Pose.identity(), 15.0, _DT_S)
    assert command.brake > 0.0
    assert command.throttle == CREEP_THROTTLE


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
    command = _timed().step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.target_speed_mps < 0.1
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
    command = _timed().step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.target_speed_mps < 0.2
    assert command.throttle > 0.0 and command.brake == pytest.approx(0.0)


def test_any_throttle_starts_past_the_dead_band() -> None:
    """A gentle pull-away must still command a throttle that moves the car."""
    config = ControlConfig(
        longitudinal_controller="time_indexed", throttle_deadband=0.2
    )
    command = _timed(config).step(_straight(speed_mps=0.1), Pose.identity(), 0.0, _DT_S)
    assert 0.2 <= command.throttle < 0.4
    full = _timed(config).step(_straight(speed_mps=30.0), Pose.identity(), 0.0, _DT_S)
    assert full.throttle == pytest.approx(1.0)


def test_a_rolling_car_gets_no_dead_band_offset() -> None:
    """Rolling, a small correction is a small throttle: the offset is for pull-aways."""
    command = _timed().step(_straight(speed_mps=8.4), Pose.identity(), 8.33, _DT_S)
    assert 0.0 < command.throttle < 0.1


def _accelerating(speed_mps: float, acceleration_mps2: float) -> Trajectory:
    """A straight 6 s plan from *speed_mps* at a constant *acceleration_mps2*."""
    plan = Trajectory.empty()
    for index in range(61):
        t_s = index * 0.1
        x = speed_mps * t_s + 0.5 * acceleration_mps2 * t_s * t_s
        plan.append(index * _STEP_US, Pose.from_xyz_yaw(x, 0.0, 0.0, 0.0))
    return plan


def test_the_plans_acceleration_is_fed_forward() -> None:
    """On the plan's speed, the pedals are the ones that settle at its acceleration."""
    command = _timed().step(_accelerating(8.0, 1.0), Pose.identity(), 8.0, _DT_S)
    assert _asked(command, 8.0) == pytest.approx(1.0, abs=0.05)


def test_the_gear_the_vehicle_is_in_sets_the_throttle() -> None:
    """The same push takes more throttle in second than in first."""
    plan = _accelerating(8.0, 1.0)
    first = _timed().step(plan, Pose.identity(), 8.0, _DT_S, gear=1)
    second = _timed().step(plan, Pose.identity(), 8.0, _DT_S, gear=2)
    assert second.throttle > first.throttle


def test_a_follower_without_a_powertrain_refuses_to_drive() -> None:
    with pytest.raises(RuntimeError):
        TrajectoryFollower().step(_straight(), Pose.identity(), 8.0, _DT_S)


def test_a_plan_easing_off_brakes_gently() -> None:
    """A plan slowing at 2 m/s² asks for a part of the brake, not all of it."""
    command = _timed().step(_accelerating(8.0, -2.0), Pose.identity(), 8.0, _DT_S)
    assert 0.0 < command.brake < 0.5


def test_a_vehicle_behind_the_plan_catches_up() -> None:
    """Where the plan has the vehicle now is read on the plan's clock: 2 m short
    of it, the vehicle aims faster than the plan."""
    plan = _straight(speed_mps=8.0)
    command = _timed().step(
        plan, Pose.identity(), 8.0, _DT_S, now_us=plan.timestamps_us[0] + 250_000
    )
    assert command.target_speed_mps == pytest.approx(
        8.0 + _TIMED.position_gain * 2.0, abs=0.05
    )
    assert command.throttle > 0.0


def test_a_car_braked_to_a_stop_pulls_away_when_the_plan_goes() -> None:
    """The integral wound up braking does not hold the car once it stands."""
    follower = _timed()
    for _ in range(40):
        follower.step(_straight(speed_mps=2.0), Pose.identity(), 8.0, _DT_S)
    command = follower.step(_accelerating(0.0, 0.5), Pose.identity(), 0.0, _DT_S)
    assert command.throttle >= follower.config.throttle_deadband
    assert command.brake == pytest.approx(0.0)


def test_a_plan_stopping_within_the_preview_brakes_early() -> None:
    """A plan reaching its stop within a second is braked for now."""
    plan = _plan([(0.0, 0.0), (1.0, 0.0), (1.5, 0.0)], speed_mps=5.0)
    plan.append(
        plan.timestamps_us[-1] + 2_000_000, Pose.from_xyz_yaw(1.5, 0.0, 0.0, 0.0)
    )
    command = _timed().step(plan, Pose.identity(), 5.0, _DT_S)
    assert command.brake > 0.0


def test_stationary_plan_holds_the_vehicle() -> None:
    """A plan that does not advance is a request to stand still."""
    config = _TIMED
    follower = _timed(config)
    plan = Trajectory.empty()
    for index in range(5):
        plan.append(index * _STEP_US, Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0))
    command = follower.step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.throttle == pytest.approx(0.0)
    assert command.brake == pytest.approx(config.stop_brake)


def test_a_jittering_plan_to_stand_still_holds_the_vehicle() -> None:
    """A learned policy's plan to stand still: 80 points jittering by centimetres.

    Its path adds up to over a metre while no point gets 0.4 m from the vehicle.
    Measured by path length it was tracked, and the follower pursued its farthest
    point, a few decimetres away and a few centimetres to the right, at full lock
    while the dead band crept the car round it.
    """
    config = _TIMED
    plan = Trajectory.empty()
    for index in range(80):
        x = 0.15 + 0.1 * math.sin(1.7 * index)
        y = -0.04 + 0.03 * math.cos(2.3 * index)
        plan.append((index + 1) * _STEP_US, Pose.from_xyz_yaw(x, y, 0.0, 0.0))
    points = plan.positions
    path_m = sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:]))
    assert path_m > 1.0  # what the old, path-length test read as a plan to move
    command = _timed(config).step(plan, Pose.identity(), 0.2, _DT_S)
    assert command.throttle == pytest.approx(0.0)
    assert command.steer == pytest.approx(0.0)
    assert command.brake == pytest.approx(config.stop_brake)


def test_a_short_plan_is_pursued_as_if_at_the_least_lookahead() -> None:
    """A plan ending 1.5 m ahead and 5 cm to the right turns as one 4 m long would.

    Pure pursuit's curvature grows as its target nears: divided by 1.5 m rather
    than 4 m, this one would steer about twice as hard, and a few decimetres away
    it reaches full lock.
    """
    config = ControlConfig(min_lookahead_m=4.0, max_steer_rate=100.0)
    short = _plan([(0.1 * i, -0.05 * i / 15) for i in range(16)], speed_mps=0.5)
    long = _plan([(0.4 * i, -0.4 * 0.05 / 1.5 * i) for i in range(11)], speed_mps=0.5)
    short_steer = _timed(config).step(short, Pose.identity(), 0.2, _DT_S).steer
    long_steer = _timed(config).step(long, Pose.identity(), 0.2, _DT_S).steer
    assert short_steer > 0.0  # to the right
    # The long plan's target is 4.002 m away, a hair past the lookahead.
    assert short_steer == pytest.approx(long_steer, rel=1e-2)


def test_empty_plan_brakes_and_resets_time_indexed() -> None:
    config = _TIMED
    follower = _timed(config)
    follower.step(_straight(), Pose.identity(), 1.0, _DT_S)
    command = follower.step(Trajectory.empty(), Pose.identity(), 5.0, _DT_S)
    assert command.brake == pytest.approx(config.stop_brake)
    assert command.steer == pytest.approx(0.0)


def test_integral_term_is_clamped() -> None:
    """Sustained error must not wind the integral term past its limit."""
    config = ControlConfig(
        longitudinal_controller="time_indexed",
        speed_kp=0.0,
        speed_ki=1.0,
        integral_limit=0.5,
        throttle_deadband=0.0,
    )
    follower = _timed(config)
    plan = _straight(speed_mps=10.0)
    for _ in range(200):
        command = follower.step(plan, Pose.identity(), 0.0, _DT_S)
    assert _asked(command, 0.0) == pytest.approx(0.5, abs=1e-6)


def test_reset_clears_controller_state() -> None:
    config = ControlConfig(
        longitudinal_controller="time_indexed",
        speed_kp=0.0,
        speed_ki=1.0,
        throttle_deadband=0.0,
    )
    follower = _timed(config)
    plan = _straight(speed_mps=10.0)
    for _ in range(10):
        follower.step(plan, Pose.identity(), 0.0, _DT_S)
    follower.reset()
    command = follower.step(plan, Pose.identity(), 0.0, _DT_S)
    assert _asked(command, 0.0) == pytest.approx(1.0 * 10.0 * _DT_S, abs=1e-6)


# ---------------------------------------------------------------------------
# Longitudinal control, pid: Autoware's autoware_pid_longitudinal_controller
# ---------------------------------------------------------------------------


def _timed_plan(xs, *, step_us: int = _STEP_US, first_us: int = _STEP_US) -> Trajectory:
    """A straight local-frame plan through *xs*, a point every *step_us*, the first
    *first_us* after now (0) -- as a policy stamps its plan from the time it plans."""
    plan = Trajectory.empty()
    for index, x in enumerate(xs):
        plan.append(first_us + index * step_us, Pose.from_xyz_yaw(x, 0.0, 0.0, 0.0))
    return plan


def _run(
    follower: TrajectoryFollower,
    plan: Trajectory,
    speed_mps: float,
    ticks: int,
    *,
    x_m: float = 0.0,
    velocities: list[float] | None = None,
) -> VehicleCommand:
    """Track *plan* for *ticks* ticks from *x_m* at a speed held at *speed_mps*."""
    command = VehicleCommand()
    for _ in range(ticks):
        command = follower.step(
            plan,
            Pose.from_xyz_yaw(x_m, 0.0, 0.0, 0.0),
            speed_mps,
            _DT_S,
            plan_velocities_mps=velocities,
        )
    return command


def _accelerating_from_rest(
    acceleration_mps2: float, seconds: float = 8.0
) -> Trajectory:
    count = int(round(seconds / 0.1))
    return _timed_plan(
        [0.5 * acceleration_mps2 * ((index + 1) * 0.1) ** 2 for index in range(count)]
    )


def test_speeds_are_put_on_the_plan_as_autowares_e2e_node_puts_them() -> None:
    """get_trajectory_from_poses on a known pose list, by hand.

    Raw speeds, distance per 0.1 s from the base: 10, 10, 10, 1, 1.  A forward
    average over 2 points: 10, 10, 5.5, 1, then the tail keeps the last average, 1.
    Accelerations: the change to the next point over 0.1 s, the last 0.
    """
    positions = np.array([[x, 0.0, 0.0] for x in (1.0, 2.0, 3.0, 3.1, 3.2)])
    stamps = [100_000 * (index + 1) for index in range(5)]
    base = np.zeros(3)
    velocities, accelerations, pose_index = plan_velocities(
        positions, stamps, base, 0, 2, False, 3.0
    )
    assert velocities == pytest.approx([10.0, 10.0, 5.5, 1.0, 1.0])
    assert accelerations == pytest.approx([0.0, -45.0, -45.0, 0.0, 0.0])
    assert pose_index == [0, 1, 2, 3, 4]

    # The vehicle moving as the plan was made: the average falling below 3 m/s
    # (at the fourth point) stops the plan on the point before it.
    velocities, accelerations, pose_index = plan_velocities(
        positions, stamps, base, 0, 2, True, 3.0
    )
    assert velocities == pytest.approx([10.0, 10.0, 5.5, 0.0, 0.0])
    assert accelerations == pytest.approx([0.0, -45.0, -55.0, 0.0, 0.0])
    assert pose_index == [0, 1, 2, 2, 2]


def test_speeds_follow_the_plans_own_step() -> None:
    """A plan stamped every 0.2 s goes half as fast over the same poses, and the
    window is capped at one less than the plan's points."""
    positions = np.array([[x, 0.0, 0.0] for x in (1.0, 2.0, 3.0)])
    velocities, _, _ = plan_velocities(
        positions, [200_000, 400_000, 600_000], np.zeros(3), 0, 8, False, 0.3
    )
    assert velocities == pytest.approx([5.0, 5.0, 5.0])


def test_autowares_default_window_on_a_policy_plan() -> None:
    """Eight-point average of a 0.1 s plan, as the node computes it."""
    xs = [0.02 * (index + 1) ** 2 for index in range(20)]
    positions = np.array([[x, 0.0, 0.0] for x in xs])
    stamps = [100_000 * (index + 1) for index in range(20)]
    velocities, accelerations, _ = plan_velocities(
        positions, stamps, np.zeros(3), 0, 8, False, 0.3
    )
    raw = [(xs[0] - 0.0) / 0.1] + [(b - a) / 0.1 for a, b in zip(xs, xs[1:])]
    expected = [sum(raw[i : i + 8]) / 8 for i in range(13)]
    expected += [expected[-1]] * 7
    assert velocities == pytest.approx(expected)
    assert accelerations[:-1] == pytest.approx(
        [(b - a) / 0.1 for a, b in zip(expected, expected[1:])]
    )
    assert accelerations[-1] == 0.0


def test_a_plan_that_dwells_then_goes_keeps_the_vehicle_stopped() -> None:
    """A plan standing on the vehicle's spot for 2.7 s, then pulling away (4.9 m at
    6 s): its first eight speeds average to zero, so it stops at its first point,
    and STOPPED departs only once the stop point is 0.5 m away."""
    follower = _follower()
    dwell = [0.0] * 27
    go = [4.9 * ((index + 1) / 33.0) ** 2 for index in range(33)]
    command = _run(follower, _timed_plan(dwell + go), 0.0, 20)
    assert follower.control_state == STOPPED
    assert follower.stop_distance_m == pytest.approx(0.0, abs=1e-6)
    assert command.throttle == pytest.approx(CREEP_THROTTLE)
    assert command.brake > 0.3
    assert _asked(command, 0.0) == pytest.approx(0.0)  # held


def test_a_plan_whose_stop_point_is_far_enough_departs() -> None:
    """Pulling away at 0.5 m/s² for 8 s: no zero speed, so the stop point is the
    plan's end, 16 m on.  STOPPED keeps the first tick (Autoware's steering check)
    and departs on the second."""
    follower = _follower()
    plan = _accelerating_from_rest(0.5)
    first = _run(follower, plan, 0.0, 1)
    assert follower.control_state == STOPPED
    assert first.brake > 0.0
    assert follower.stop_distance_m == pytest.approx(16.0, abs=0.1)
    _run(follower, plan, 0.0, 1)
    assert follower.control_state == DRIVE
    # The first DRIVE output rises from STOPPED's brake by max_acc_cmd_diff.
    third = _run(follower, plan, 0.0, 1)
    assert third.acceleration_mps2 > 0.0
    assert third.throttle > 0.0 and third.brake == pytest.approx(0.0)


def test_a_stop_point_nearer_than_half_a_metre_does_not_depart() -> None:
    """A plan that goes 0.4 m and stops: STOPPED holds."""
    follower = _follower()
    xs = [min(0.4, 0.05 * (index + 1)) for index in range(40)]
    _run(follower, _timed_plan(xs), 0.0, 10)
    assert follower.control_state == STOPPED


def test_a_crawling_pull_away_gets_the_throttle_that_moves_the_car() -> None:
    """Pulling away at 0.1 m/s², the lookahead feedback asks a standing car for less
    than CARLA's MKZ moves at (0.25): the throttle is floored, or the car and the
    plan stand for good."""
    plan = _accelerating_from_rest(0.1)
    floored = _follower()
    command = _run(floored, plan, 0.0, 20)
    assert floored.control_state == DRIVE
    assert command.throttle == pytest.approx(floored.config.throttle_deadband)
    assert command.brake == pytest.approx(0.0)
    bare = _follower(ControlConfig(throttle_deadband=0.0))
    assert _run(bare, plan, 0.0, 20).throttle < 0.25
    # Rolling, the floor is off.
    assert _run(_follower(), plan, 1.0, 20).throttle < floored.config.throttle_deadband


def _plan_to_stop(
    x_m: float, speed_mps: float, stop_m: float, now_us: int
) -> Trajectory:
    """A policy's plan from *x_m* at *speed_mps*: braking evenly onto *stop_m*, then
    standing there; standing where it is once (nearly) stopped."""
    remaining = stop_m - x_m
    deceleration, stop_s = 0.0, 0.0
    if speed_mps > 0.05 and remaining > 0.01:
        deceleration = speed_mps * speed_mps / (2.0 * remaining)
        stop_s = speed_mps / deceleration
    xs = []
    for index in range(80):
        t_s = min((index + 1) * 0.1, stop_s)
        xs.append(x_m + speed_mps * t_s - 0.5 * deceleration * t_s * t_s)
    return _timed_plan(xs, first_us=now_us + _STEP_US)


def test_the_vehicle_stops_at_the_plans_stop_point() -> None:
    """From 6 m/s, plans braking evenly onto a stop 12 m on, made every 0.1 s from
    where the vehicle is: DRIVE, STOPPING once the stop point is within 0.5 m, then
    STOPPED, at rest short of the stop -- by the metre the e2e node's forward
    average and force stop take off the plan.  The vehicle answers through the
    MKZ's powertrain model, its pedals settling at once."""
    powertrain = mkz_powertrain()
    follower = TrajectoryFollower(ControlConfig(), powertrain)
    stop_m = 12.0
    x_m, speed, states = 0.0, 6.0, []
    plan = Trajectory.empty()
    for tick in range(300):
        if tick % 2 == 0:
            plan = _plan_to_stop(x_m, speed, stop_m, tick * 50_000)
        command = follower.step(
            plan, Pose.from_xyz_yaw(x_m, 0.0, 0.0, 0.0), speed, _DT_S
        )
        states.append(follower.control_state)
        gear = powertrain.gear_for(speed)
        acceleration = powertrain.acceleration(
            speed, command.throttle, command.brake, gear
        )
        speed = max(0.0, speed + acceleration * _DT_S)
        x_m += speed * _DT_S
    assert states[0] == DRIVE  # taken over moving
    assert STOPPING in states
    assert states[-1] == STOPPED
    assert speed == 0.0
    assert stop_m - 1.5 < x_m <= stop_m
    assert command.brake == pytest.approx(
        powertrain.pedals(ControlConfig().stopped_acc, 0.0, 1)[1]
    )


def test_overshooting_the_stop_point_is_an_emergency() -> None:
    """In DRIVE, a plan standing 2 m and more behind the vehicle (past the 1.5 m
    overshoot) brakes at the emergency deceleration, reached at the emergency jerk."""
    follower = _follower()
    _run(follower, _accelerating_from_rest(0.5), 0.0, 2)
    assert follower.control_state == DRIVE
    behind = _timed_plan([-6.0 + 0.5 * index for index in range(9)])
    command = _run(follower, behind, 3.0, 1, velocities=[0.0] * 9)
    assert follower.control_state == EMERGENCY
    assert follower.stop_distance_m == pytest.approx(-6.0)
    assert command.acceleration_mps2 < 0.0
    command = _run(follower, behind, 3.0, 60, velocities=[0.0] * 9)
    assert command.acceleration_mps2 == pytest.approx(-5.0)


def test_drive_tracks_the_plans_speed_through_the_lookahead_feedback() -> None:
    """At the plan's speed and acceleration, Autoware's default feedback asks for
    half the plan's acceleration plus half the (filtered) gap a second ahead."""
    follower = _follower()
    plan = _timed_plan([0.1 * (index + 1) * 8.0 for index in range(80)])
    command = _run(follower, plan, 8.0, 2)
    assert follower.control_state == DRIVE
    assert command.target_speed_mps == pytest.approx(8.0)
    command = _run(follower, plan, 8.0, 40)
    assert command.acceleration_mps2 == pytest.approx(0.0, abs=1e-6)
    slower = _run(follower, plan, 6.0, 60)
    # Gap 2 m/s closed over 1 s, weighted 0.5: 1 m/s², under the 2 m/s³ jerk limit.
    assert slower.acceleration_mps2 == pytest.approx(1.0, abs=0.02)
    assert slower.throttle > 0.0


def test_the_acceleration_is_jerk_limited() -> None:
    """Departing, the asked acceleration rises from zero by at most max_jerk (2 m/s³)."""
    follower = _follower()
    plan = _timed_plan([0.1 * (index + 1) * 10.0 for index in range(80)])
    _run(follower, plan, 0.0, 1)
    for tick in range(1, 11):
        command = _run(follower, plan, 0.0, 1)
        assert follower.control_state == DRIVE
        assert command.acceleration_mps2 <= 2.0 * _DT_S * tick + 1e-9
    assert command.acceleration_mps2 == pytest.approx(2.0 * _DT_S * 10)


def test_the_plans_own_speeds_are_used_when_given() -> None:
    """An Autoware trajectory's speeds, not ones from its poses: here zero, a stop
    at its first point."""
    follower = _follower()
    plan = _timed_plan([0.2 + index for index in range(20)])
    _run(follower, plan, 0.0, 5, velocities=[0.0] * 20)
    assert follower.control_state == STOPPED
    _run(follower, plan, 0.0, 5)
    assert follower.control_state == DRIVE


def test_a_single_point_plan_holds_the_vehicle() -> None:
    """Autoware's controller ignores a trajectory of fewer than two points: with
    none before it, there is nothing to track."""
    plan = _timed_plan([3.0])
    command = _follower().step(plan, Pose.identity(), 0.0, _DT_S)
    assert command.brake == pytest.approx(ControlConfig().stop_brake)


def test_a_plan_too_short_is_ignored_and_the_last_one_tracked() -> None:
    """setTrajectory keeps the last trajectory when it is handed a short (or
    empty) one: the controller's state carries on, not reset to a hold."""
    follower = _follower()
    plan = _timed_plan([0.1 * (index + 1) * 8.0 for index in range(80)])
    _run(follower, plan, 8.0, 3)
    assert follower.control_state == DRIVE
    for short in (_timed_plan([3.0]), Trajectory.empty()):
        command = _run(follower, short, 8.0, 1)
        assert follower.control_state == DRIVE
        assert command.target_speed_mps == pytest.approx(8.0)
        assert command.brake == pytest.approx(0.0)


def test_an_anchor_point_at_the_planning_time_is_the_plans_base() -> None:
    """carla-driver-interface leads a plan with the ego's pose at the planning
    time; the policy's poses follow every 0.1 s.  The anchor is where the speeds
    are measured from, not a point of the trajectory: no speed of zero at the
    plan's start, and Autoware's numbers for the poses after it."""
    now_us = 5_000_000
    xs = [0.5 * (index + 1) for index in range(20)]  # 5 m/s
    plan = Trajectory.empty()
    plan.append(now_us, Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0))
    for index, x in enumerate(xs):
        plan.append(
            now_us + (index + 1) * _STEP_US, Pose.from_xyz_yaw(x, 0.0, 0.0, 0.0)
        )
    follower = _follower()
    follower.step(plan, Pose.identity(), 5.0, _DT_S, now_us=now_us)
    assert len(follower._velocities) == 20  # noqa: SLF001
    assert follower._velocities == pytest.approx([5.0] * 20)  # noqa: SLF001
    assert follower._pose_index[0] == 1  # noqa: SLF001
    # Without the anchor taken as such, its own speed would read zero.
    unanchored = _follower()
    unanchored.step(plan, Pose.identity(), 5.0, _DT_S)
    assert unanchored._velocities[0] < 5.0  # noqa: SLF001


def test_the_plans_base_is_latched_with_the_plan() -> None:
    """A 10 Hz plan tracked over two 20 Hz ticks keeps the speeds put on it from
    where the vehicle was as it came, though the vehicle has moved on since."""
    follower = _follower()
    plan = _timed_plan([0.1 * (index + 1) * 8.0 for index in range(80)])
    follower.step(plan, Pose.identity(), 8.0, _DT_S, now_us=0)
    speeds = list(follower._velocities)  # noqa: SLF001
    follower.step(
        plan, Pose.from_xyz_yaw(0.4, 0.0, 0.0, 0.0), 8.0, _DT_S, now_us=50_000
    )
    assert follower._velocities == speeds  # noqa: SLF001
    assert speeds[0] == pytest.approx(8.0)


def test_a_moving_vehicle_taken_over_starts_in_drive() -> None:
    """A follower started (or reset) on a moving vehicle -- the run-up's hand-off,
    a spawn at speed, a new command source -- drives on: no STOPPED brake pulse."""
    follower = _follower()
    plan = _timed_plan([0.1 * (index + 1) * 8.0 for index in range(80)])
    command = _run(follower, plan, 8.0, 1)
    assert follower.control_state == DRIVE
    assert command.brake == pytest.approx(0.0)
    follower.reset()
    command = _run(follower, plan, 8.0, 1)
    assert follower.control_state == DRIVE
    assert command.brake == pytest.approx(0.0)
    # Standing, it starts STOPPED, as Autoware's does.
    standing = _follower()
    _run(standing, plan, 0.0, 1)
    assert standing.control_state == STOPPED


def test_stopping_departs_again_once_the_stop_point_moves_away() -> None:
    """STOPPING returns to DRIVE once the stop point is farther than
    drive_state_stop_dist + drive_state_offset_stop_dist (1.5 m)."""
    follower = _follower()
    xs = [0.1 * (index + 1) for index in range(40)]
    stopping = [2.0 if x < 0.3 - 1e-9 else 0.0 for x in xs]
    _run(follower, _timed_plan(xs), 2.0, 1, velocities=stopping)
    assert follower.control_state == STOPPING
    assert follower.stop_distance_m == pytest.approx(0.3)
    _run(follower, _timed_plan(xs), 2.0, 1, velocities=[2.0] * 40)
    assert follower.control_state == DRIVE


def test_an_unknown_longitudinal_controller_is_refused() -> None:
    with pytest.raises(ValueError, match="longitudinal_controller"):
        ControlConfig.from_mapping({"longitudinal_controller": "mpc"})
    with pytest.raises(ValueError, match="longitudinal_controller"):
        TrajectoryFollower(ControlConfig(longitudinal_controller="mpc"))


def test_a_time_indexed_gain_on_the_pid_controller_is_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        ControlConfig.from_mapping({"speed_kp": 2.0})
    assert any("speed_kp" in record.getMessage() for record in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING"):
        ControlConfig.from_mapping({"speed_kp": 1.0, "kp": 2.0})
        ControlConfig.from_mapping(
            {"longitudinal_controller": "time_indexed", "speed_kp": 2.0}
        )
    assert not caplog.records


# ---------------------------------------------------------------------------
# Command conversion
# ---------------------------------------------------------------------------


def test_command_defaults_are_inert() -> None:
    command = VehicleCommand()
    assert command.throttle == 0.0
    assert command.brake == 0.0
    assert command.hand_brake is False
    assert command.reverse is False


# ---------------------------------------------------------------------------
# Non-finite plans and measurements (#916)
# ---------------------------------------------------------------------------

_CONTROLLERS = ("pid", "time_indexed")


def _policy_plan(y_m: float = 0.0) -> Trajectory:
    """A policy's straight plan at 8 m/s, a point every 0.1 s from 0.1 s on, every
    point *y_m* to the side (NaN or an infinity: a bad inference)."""
    plan = Trajectory.empty()
    for index in range(40):
        plan.append(
            (index + 1) * _STEP_US,
            Pose(
                np.array([0.8 * (index + 1), y_m, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
            ),
        )
    return plan


def _drive(
    follower: TrajectoryFollower,
    plans: list[Trajectory],
    *,
    velocities: list[list[float] | None] | None = None,
    accelerations: list[list[float] | None] | None = None,
) -> list[VehicleCommand]:
    commands = []
    for index, plan in enumerate(plans):
        commands.append(
            follower.step(
                plan,
                Pose.identity(),
                8.0,
                _DT_S,
                yaw_rate_rps=0.0,
                now_us=0,
                plan_velocities_mps=None if velocities is None else velocities[index],
                plan_accelerations_mps2=(
                    None if accelerations is None else accelerations[index]
                ),
            )
        )
    return commands


@pytest.mark.parametrize("controller", _CONTROLLERS)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_plan_is_refused_and_the_last_one_tracked(
    controller: str, bad: float, caplog: pytest.LogCaptureFixture
) -> None:
    """The issue's repro: one bad plan among good straight ones used to latch NaN in
    the yaw-rate trim and steer at full lock for the rest of the run.  Refused, it
    leaves the follower exactly where tracking the last good plan again would."""
    config = ControlConfig(longitudinal_controller=controller)
    good = [_policy_plan() for _ in range(61)]
    with caplog.at_level("WARNING", logger="autoware_carla_scenario.driver.control"):
        hit = _drive(_follower(config), [good[0], _policy_plan(bad), *good[2:]])
    reference = _drive(_follower(config), good)
    assert hit == reference
    last = hit[-1]
    assert last.steer == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(last.throttle) and last.throttle > 0.0
    assert [record.getMessage() for record in caplog.records] == [
        "TrajectoryFollower: refused a non-finite plan (NaN or infinite; "
        "1 refused so far)"
    ]


@pytest.mark.parametrize("controller", _CONTROLLERS)
def test_a_non_finite_first_plan_holds_the_vehicle(controller: str) -> None:
    follower = _follower(ControlConfig(longitudinal_controller=controller))
    (command,) = _drive(follower, [_policy_plan(math.nan)])
    assert command == VehicleCommand(brake=ControlConfig().stop_brake)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
@pytest.mark.parametrize("field", ["velocities", "accelerations"])
def test_non_finite_given_speeds_are_refused(bad: float, field: str) -> None:
    """An Autoware trajectory's NaN speed used to latch in the pid controller's
    filters and integral: NaN throttle for the rest of the run."""
    plans = [_policy_plan() for _ in range(40)]
    good: list[list[float] | None] = [[8.0] * 40 for _ in plans]
    zeros: list[list[float] | None] = [[0.0] * 40 for _ in plans]
    bad_given = list(good)
    bad_zeros = list(zeros)
    target = bad_given if field == "velocities" else bad_zeros
    target[1] = [bad] * 40
    hit = _drive(_follower(), plans, velocities=bad_given, accelerations=bad_zeros)
    reference = _drive(_follower(), plans, velocities=good, accelerations=zeros)
    assert hit == reference
    assert all(math.isfinite(command.throttle) for command in hit)
    assert hit[-1].throttle > 0.0


def test_the_time_indexed_controller_ignores_the_speeds_it_does_not_read() -> None:
    config = ControlConfig(longitudinal_controller="time_indexed")
    plans = [_policy_plan() for _ in range(5)]
    hit = _drive(_follower(config), plans, velocities=[[math.nan] * 40] * 5)
    assert hit == _drive(_follower(config), plans)


@pytest.mark.parametrize("controller", _CONTROLLERS)
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_yaw_rate_is_not_fed_back(controller: str, bad: float) -> None:
    follower = _follower(ControlConfig(longitudinal_controller=controller))
    plan = _arc(60.0, 13.0)
    follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=0.0)
    follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=0.1)
    trim = follower._yaw_rate_trim  # noqa: SLF001
    command = follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=bad)
    assert follower._yaw_rate_trim == trim  # noqa: SLF001
    assert math.isfinite(command.steer) and abs(command.steer) < 0.5
    command = follower.step(plan, Pose.identity(), 13.0, _DT_S, yaw_rate_rps=0.1)
    assert math.isfinite(follower._yaw_rate_trim)  # noqa: SLF001
    assert math.isfinite(command.steer) and abs(command.steer) < 0.5


@pytest.mark.parametrize("controller", _CONTROLLERS)
@pytest.mark.parametrize("speed, dt", [(math.nan, _DT_S), (8.0, math.nan)])
def test_a_non_finite_speed_or_step_brakes_and_keeps_the_state(
    controller: str, speed: float, dt: float
) -> None:
    config = ControlConfig(longitudinal_controller=controller)
    follower = _follower(config)
    plan = _policy_plan(0.5)
    _drive(follower, [plan] * 3)
    steer = follower._previous_steer  # noqa: SLF001
    command = follower.step(
        plan, Pose.identity(), speed, dt, yaw_rate_rps=0.0, now_us=0
    )
    assert command == VehicleCommand(steer=steer, brake=config.stop_brake)
    # Nothing latched: the next good step is the one an unbroken run takes.
    after = _drive(follower, [plan] * 3)
    assert after == _drive(_follower(config), [plan] * 6)[3:]


@pytest.mark.parametrize("controller", _CONTROLLERS)
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_ego_pose_brakes_and_keeps_the_state(
    controller: str, bad: float
) -> None:
    """A NaN ego pose would put NaN speeds on the plan, kept with it while it lasts."""
    config = ControlConfig(longitudinal_controller=controller)
    follower = _follower(config)
    plan = _policy_plan(0.5)
    _drive(follower, [plan] * 3)
    steer = follower._previous_steer  # noqa: SLF001
    # (A Pose cannot be built with a NaN orientation.)
    command = follower.step(
        plan,
        Pose(np.array([bad, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
        8.0,
        _DT_S,
        yaw_rate_rps=0.0,
        now_us=0,
    )
    assert command == VehicleCommand(steer=steer, brake=config.stop_brake)
    after = _drive(follower, [plan] * 3)
    assert after == _drive(_follower(config), [plan] * 6)[3:]


@pytest.mark.parametrize("controller", _CONTROLLERS)
def test_a_refused_plan_past_the_last_ones_end_is_tracked_sanely(
    controller: str,
) -> None:
    """The last good plan, tracked on after a refusal, is in the past by then (as
    a cached plan is while a policy is slow): tracked as it is, to its end, with
    no non-finite command."""
    follower = _follower(ControlConfig(longitudinal_controller=controller))
    follower.step(_policy_plan(), Pose.identity(), 8.0, _DT_S, now_us=0)
    for x_m, now_us in ((10.0, 2_000_000), (31.0, 10_000_000), (31.9, 60_000_000)):
        command = follower.step(
            _policy_plan(math.nan),
            Pose.from_xyz_yaw(x_m, 0.0, 0.0, 0.0),
            8.0,
            _DT_S,
            yaw_rate_rps=0.0,
            now_us=now_us,
        )
        assert all(
            math.isfinite(value)
            for value in (command.throttle, command.steer, command.brake)
        )
        assert command.steer == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("controller", _CONTROLLERS)
def test_a_non_finite_command_is_replaced_by_the_brake(controller: str) -> None:
    config = ControlConfig(longitudinal_controller=controller)
    follower = _follower(config)
    _drive(follower, [_policy_plan()] * 3)
    powertrain = follower.powertrain
    assert powertrain is not None
    follower.powertrain = SimpleNamespace(  # type: ignore[assignment]
        pedals=lambda *_: (math.nan, 0.0)
    )
    (command,) = _drive(follower, [_policy_plan()])
    assert command == VehicleCommand(steer=command.steer, brake=config.stop_brake)
    assert math.isfinite(command.steer)


def test_controller_state_is_never_updated_with_a_non_finite_value() -> None:
    """Defence in depth, below the checks at the entry."""
    follower = _follower()
    _drive(follower, [_policy_plan()] * 3)
    follower._yaw_rate_feedback(8.0, math.nan, 0.0, _DT_S)  # noqa: SLF001
    follower._yaw_rate_feedback(8.0, 0.0, math.nan, _DT_S)  # noqa: SLF001
    follower._pid_calculate(math.nan, _DT_S, True)  # noqa: SLF001
    for name in (
        "_yaw_rate_trim",
        "_asked_curvature",
        "_pid_integral",
        "_pid_previous_error",
    ):
        value = getattr(follower, name)
        assert value is not None and math.isfinite(value), name
