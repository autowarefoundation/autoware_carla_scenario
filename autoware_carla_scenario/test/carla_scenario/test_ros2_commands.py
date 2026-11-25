"""Tests for the ROS 2 ego's command arbitration and plan building."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from autoware_carla_scenario.entity.ros2_entity import plan_from_trajectory
from autoware_carla_scenario.ros2 import (
    CommandArbiter,
    CommandSourceConfig,
    ControlCommand,
    SteerAccelCommand,
    TrajectoryCommand,
    TrajectoryCommandPoint,
)
from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.ros2.commands import (
    Command,
    SteerAccelPairer,
    command_is_applicable,
    command_is_finite,
    fits_float32,
)
from autoware_carla_scenario.ros2.config import FLOAT32_MAX
from autoware_carla_scenario.ros2.transport import Received

_VP = CommandSourceConfig(
    name="vp",
    kind="steer_accel",
    steering_topic="/s",
    acceleration_topic="/a",
    priority=0,
    stale_s=0.3,
)
_SI = CommandSourceConfig(
    name="si", kind="control", topic="/c", priority=10, stale_s=0.3
)


def _us(seconds: float) -> int:
    return int(round(seconds * 1_000_000))


def _control(velocity: float = 1.0) -> ControlCommand:
    return ControlCommand(0.0, velocity, -3.0)


# ---------------------------------------------------------------------------
# CommandArbiter
# ---------------------------------------------------------------------------


def test_nothing_is_selected_before_any_command() -> None:
    assert CommandArbiter([_VP, _SI]).select(_us(0.0)) is None


def test_a_fresh_higher_priority_source_wins() -> None:
    arbiter = CommandArbiter([_VP, _SI])
    arbiter.observe("vp", SteerAccelCommand(0.1, 0.5), _us(1.0))
    arbiter.observe("si", _control(), _us(1.0))
    selection = arbiter.select(_us(1.05))
    assert selection is not None
    assert selection.source.name == "si"
    assert selection.fresh


def test_a_stale_higher_priority_source_gives_way() -> None:
    arbiter = CommandArbiter([_VP, _SI])
    arbiter.observe("si", _control(), _us(1.0))
    arbiter.observe("vp", SteerAccelCommand(0.1, 0.5), _us(1.4))
    selection = arbiter.select(_us(1.4))
    assert selection is not None
    assert selection.source.name == "vp"


def test_a_command_exactly_stale_s_old_is_still_fresh() -> None:
    arbiter = CommandArbiter([_SI])
    arbiter.observe("si", _control(), _us(1.0))
    assert arbiter.select(_us(1.3)).fresh  # type: ignore[union-attr]
    assert not arbiter.select(_us(1.31)).fresh  # type: ignore[union-attr]


def test_with_every_source_stale_the_last_driver_is_returned_not_fresh() -> None:
    arbiter = CommandArbiter([_VP, _SI])
    arbiter.observe("vp", SteerAccelCommand(0.1, 0.5), _us(1.0))
    assert arbiter.select(_us(1.0)).source.name == "vp"  # type: ignore[union-attr]
    stale = arbiter.select(_us(5.0))
    assert stale is not None
    assert stale.source.name == "vp"
    assert not stale.fresh
    assert stale.command == SteerAccelCommand(0.1, 0.5)


def test_equal_priorities_take_the_most_recent() -> None:
    other = CommandSourceConfig(name="other", kind="control", topic="/o", priority=10)
    arbiter = CommandArbiter([_SI, other])
    arbiter.observe("si", _control(1.0), _us(1.0))
    arbiter.observe("other", _control(2.0), _us(1.1))
    assert arbiter.select(_us(1.1)).source.name == "other"  # type: ignore[union-attr]


def test_equal_priorities_take_the_most_recent_within_one_tick() -> None:
    """Every command polled on a tick arrives at the tick's time: the order
    they were observed in, not that time, says which is the most recent."""
    other = CommandSourceConfig(name="other", kind="control", topic="/o", priority=10)
    arbiter = CommandArbiter([_SI, other])
    arbiter.observe("si", _control(1.0), 100)
    assert arbiter.select(100).source.name == "si"  # type: ignore[union-attr]
    arbiter.observe("other", _control(2.0), 200)
    arbiter.observe("si", _control(3.0), 200)
    assert arbiter.select(200).source.name == "si"  # type: ignore[union-attr]
    arbiter.observe("si", _control(4.0), 300)
    arbiter.observe("other", _control(5.0), 300)
    selection = arbiter.select(300)
    assert selection is not None
    assert (selection.source.name, selection.command) == ("other", _control(5.0))
    # The arrival time is still what freshness is judged by.
    assert selection.received_us == 300


def test_a_newer_command_replaces_the_older_from_the_same_source() -> None:
    arbiter = CommandArbiter([_SI])
    arbiter.observe("si", _control(1.0), _us(1.0))
    arbiter.observe("si", _control(2.0), _us(1.1))
    assert arbiter.select(_us(1.1)).command == _control(2.0)  # type: ignore[union-attr]


def test_an_unknown_source_is_refused() -> None:
    with pytest.raises(KeyError):
        CommandArbiter([_SI]).observe("ghost", _control(), _us(0.0))


def _trajectory(
    x: float = 0.0, w: float = 1.0, velocity: float = 1.0, acceleration: float = 0.0
) -> TrajectoryCommand:
    return TrajectoryCommand(
        stamp_us=0,
        points=(
            TrajectoryCommandPoint(
                0,
                Pose(np.array([x, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, w])),
                velocity,
                acceleration,
            ),
        ),
    )


@pytest.mark.parametrize(
    "command",
    [
        SteerAccelCommand(0.1, 0.5),
        ControlCommand(0.1, -2.0, -3.0),
        ControlCommand(0.1, 2.0, -3.0, jerk_mps3=-1.0),
        _trajectory(),
    ],
)
def test_a_finite_command_is_finite(command: Command) -> None:
    assert command_is_finite(command)


@pytest.mark.parametrize(
    "command",
    [
        SteerAccelCommand(math.nan, 0.5),
        SteerAccelCommand(0.1, math.inf),
        ControlCommand(math.nan, 2.0, -3.0),
        ControlCommand(0.1, -math.inf, -3.0),
        ControlCommand(0.1, 2.0, math.nan),
        ControlCommand(0.1, 2.0, -3.0, jerk_mps3=math.nan),
        _trajectory(x=math.nan),
        _trajectory(w=math.inf),
        _trajectory(velocity=math.nan),
        _trajectory(acceleration=-math.inf),
    ],
)
def test_a_non_finite_command_is_not_finite(command: Command) -> None:
    assert not command_is_finite(command)


def test_a_non_finite_command_does_not_keep_its_source_fresh() -> None:
    """#2434: a source sending NaN every cycle used to count as fresh forever."""
    arbiter = CommandArbiter([_VP, _SI])
    assert arbiter.observe("si", _control(2.0), _us(1.0))
    assert arbiter.observe("vp", SteerAccelCommand(0.1, 0.5), _us(1.0))
    for seconds in (1.1, 1.2, 1.3, 1.4):
        assert not arbiter.observe("si", _control(math.nan), _us(seconds))
        assert arbiter.observe("vp", SteerAccelCommand(0.1, 0.5), _us(seconds))
    # Within stale_s, its last good command still drives...
    selection = arbiter.select(_us(1.3))
    assert selection is not None
    assert (selection.source.name, selection.command) == ("si", _control(2.0))
    # ...and then it ages out as a silent source does.
    selection = arbiter.select(_us(1.4))
    assert selection is not None
    assert selection.source.name == "vp"


# ---------------------------------------------------------------------------
# Float32 range (#2505)
# ---------------------------------------------------------------------------


def test_float32_max_is_the_largest_finite_float32() -> None:
    assert FLOAT32_MAX == float(np.finfo(np.float32).max)
    with np.errstate(over="ignore"):
        assert np.isinf(np.float32(3.5e38))


@pytest.mark.parametrize("value", [0.0, -1.5, FLOAT32_MAX, -FLOAT32_MAX, 1e38])
def test_a_value_within_float32_s_range_fits(value: float) -> None:
    assert fits_float32(value)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        1e39,
        -1e39,
        1e308,
        np.nextafter(FLOAT32_MAX, math.inf),
    ],
)
def test_a_value_beyond_float32_s_range_does_not_fit(value: float) -> None:
    assert not fits_float32(value)


@pytest.mark.parametrize(
    "command",
    [
        SteerAccelCommand(1e39, 0.5),
        SteerAccelCommand(0.1, 1e39),
        SteerAccelCommand(0.1, -1e39),
        ControlCommand(1e39, 2.0, -3.0),
        ControlCommand(0.1, 1e39, -3.0),
        ControlCommand(0.1, 2.0, -1e39),
        ControlCommand(0.1, 2.0, -3.0, jerk_mps3=1e39),
    ],
)
def test_a_command_beyond_float32_s_range_is_refused(command: Command) -> None:
    """#2505: finite as a double, but an infinity once CARLA's
    VehicleAckermannControl stores it as a float32."""
    assert not command_is_finite(command)
    assert not CommandArbiter([_VP, _SI]).observe(
        "vp" if isinstance(command, SteerAccelCommand) else "si", command, 0
    )


def test_a_trajectory_is_not_bounded_to_float32() -> None:
    """A trajectory is read by the framework's follower, in doubles."""
    assert command_is_finite(_trajectory(velocity=1e39))


@pytest.mark.parametrize(
    "acceleration",
    [
        # Each fits a float32 itself; the step, twice it, does not.
        2e38,
        FLOAT32_MAX,
        -FLOAT32_MAX,
    ],
)
def test_a_steer_accel_step_beyond_float32_s_range_is_refused(
    acceleration: float,
) -> None:
    """#2505: ``acceleration * speed_horizon_s`` (2 s) is the target speed's
    step; it must fit a float32 too, or CARLA gets an infinite target."""
    command = SteerAccelCommand(0.1, acceleration)
    assert command_is_finite(command)
    assert not command_is_applicable(command, _VP)
    assert not CommandArbiter([_VP]).observe("vp", command, 0)


def test_a_steer_accel_step_within_float32_s_range_is_applicable() -> None:
    command = SteerAccelCommand(0.1, FLOAT32_MAX / 2.0)
    assert command_is_applicable(command, _VP)  # the step is FLOAT32_MAX


def test_a_steer_accel_steering_scaled_beyond_float32_s_range_is_refused() -> None:
    source = CommandSourceConfig(
        name="vp",
        kind="steer_accel",
        steering_topic="/s",
        acceleration_topic="/a",
        steering_sign=1e10,
    )
    assert command_is_applicable(SteerAccelCommand(1e28, 0.5), source)
    assert not command_is_applicable(SteerAccelCommand(1e29, 0.5), source)


def test_an_overflowing_command_is_never_selected_or_held() -> None:
    """#2505: the last good command drives on, then the source goes stale."""
    arbiter = CommandArbiter([_VP])
    assert arbiter.observe("vp", SteerAccelCommand(0.1, 0.5), _us(1.0))
    for seconds in (1.1, 1.2, 1.3, 1.4):
        assert not arbiter.observe("vp", SteerAccelCommand(0.1, 1e308), _us(seconds))
    selection = arbiter.select(_us(1.3))
    assert selection is not None
    assert (selection.command, selection.fresh) == (SteerAccelCommand(0.1, 0.5), True)
    selection = arbiter.select(_us(1.4))
    assert selection is not None
    assert (selection.command, selection.fresh) == (SteerAccelCommand(0.1, 0.5), False)


def test_a_source_with_only_non_finite_commands_never_drives() -> None:
    arbiter = CommandArbiter([_SI])
    assert not arbiter.observe("si", _control(math.nan), _us(1.0))
    assert arbiter.select(_us(1.0)) is None


# ---------------------------------------------------------------------------
# SteerAccelPairer
# ---------------------------------------------------------------------------


_MS = 1_000_000


def test_a_command_needs_a_fresh_value_on_both_topics() -> None:
    pairer = SteerAccelPairer()
    assert pairer.steering(0.1, 0) is None
    assert pairer.acceleration(0.5, 1 * _MS) == SteerAccelCommand(0.1, 0.5)
    # Each value is used once.
    assert pairer.acceleration(0.6, 100 * _MS) is None
    assert pairer.steering(0.2, 101 * _MS) == SteerAccelCommand(0.2, 0.6)


def test_the_newest_value_of_a_topic_is_paired() -> None:
    pairer = SteerAccelPairer()
    pairer.steering(0.1, 0)
    pairer.steering(0.2, 100 * _MS)
    assert pairer.acceleration(0.5, 101 * _MS) == SteerAccelCommand(0.2, 0.5)


def test_halves_written_further_apart_than_the_window_do_not_pair() -> None:
    pairer = SteerAccelPairer(window_ns=20 * _MS)
    pairer.acceleration(0.5, 0)
    # The stale acceleration is dropped, not paired with this steering ...
    assert pairer.steering(0.1, 21 * _MS) is None
    # ... which waits for its own acceleration.
    assert pairer.acceleration(0.6, 22 * _MS) == SteerAccelCommand(0.1, 0.6)
    # Exactly the window apart still pairs.
    pairer.steering(0.2, 100 * _MS)
    assert pairer.acceleration(0.7, 120 * _MS) == SteerAccelCommand(0.2, 0.7)


def test_a_half_delivered_late_is_dropped_and_keeps_the_pending_one() -> None:
    pairer = SteerAccelPairer(window_ns=20 * _MS)
    pairer.steering(0.2, 100 * _MS)
    # An acceleration written long before the pending steering: a lost cycle's.
    assert pairer.acceleration(0.5, 0) is None
    assert pairer.acceleration(0.6, 101 * _MS) == SteerAccelCommand(0.2, 0.6)


def test_the_window_defaults_to_half_the_measured_cycle() -> None:
    pairer = SteerAccelPairer()
    # Unmeasured: the first two halves pair whatever their distance.
    assert pairer.window_ns is None
    pairer.steering(0.1, 0)
    assert pairer.acceleration(0.5, 45 * _MS) == SteerAccelCommand(0.1, 0.5)
    pairer.steering(0.2, 100 * _MS)
    pairer.acceleration(0.6, 145 * _MS)
    assert pairer.window_ns == 50 * _MS
    # The median: one interval doubled by a lost message does not move it.
    pairer.steering(0.4, 300 * _MS)
    assert pairer.window_ns == 50 * _MS
    assert pairer.dropped == 0


def test_a_lost_half_at_a_fast_rate_does_not_shift_the_pairing() -> None:
    # 50 Hz with the acceleration 8 ms after the steering, so the next cycle's
    # steering is 12 ms after it: a fixed 20 ms window would pair the orphaned
    # acceleration with it, and every later command would lag by a cycle.
    pairer = SteerAccelPairer()
    got: List[Tuple[float, float]] = []
    for k in range(1, 9):
        stamp = k * 20 * _MS
        commands = [pairer.steering(k / 10, stamp)] if k != 4 else []
        commands.append(pairer.acceleration(float(k), stamp + 8 * _MS))
        got += [
            (command.steering_rad, command.acceleration_mps2)
            for command in commands
            if command is not None
        ]
    assert got == [(k / 10, float(k)) for k in (1, 2, 3, 5, 6, 7, 8)]
    assert pairer.dropped == 1


def test_dropped_counts_every_half_left_unpaired() -> None:
    pairer = SteerAccelPairer(window_ns=20 * _MS)
    pairer.steering(0.1, 0)
    pairer.steering(0.2, 100 * _MS)  # supersedes 0.1
    pairer.acceleration(0.5, 0)  # late: dropped itself
    pairer.acceleration(0.6, 200 * _MS)  # 0.2 is stale
    assert pairer.dropped == 3
    assert pairer.window_ns == 20 * _MS


def test_a_negative_window_is_refused() -> None:
    with pytest.raises(ValueError):
        SteerAccelPairer(window_ns=-1)
    with pytest.raises(ValueError):
        CommandSourceConfig(
            name="vp",
            kind="steer_accel",
            steering_topic="/s",
            acceleration_topic="/a",
            pair_window_s=-0.1,
        )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def _point(t: float, x: float, v: float) -> TrajectoryCommandPoint:
    return TrajectoryCommandPoint(_us(t), Pose.from_xyz_yaw(x, 0.0, 0.0, 0.0), v)


def test_a_timed_trajectory_keeps_its_timing() -> None:
    command = TrajectoryCommand(
        stamp_us=_us(10.0), points=(_point(0.0, 0.0, 2.0), _point(0.5, 1.0, 2.0))
    )
    plan = plan_from_trajectory(command, anchor_us=_us(10.0))
    assert plan.timestamps_us == [10_000_000, 10_500_000]
    assert plan.positions[1][0] == pytest.approx(1.0)


def test_an_untimed_trajectory_is_timed_from_its_speeds() -> None:
    command = TrajectoryCommand(
        stamp_us=0,
        points=(_point(0.0, 0.0, 2.0), _point(0.0, 2.0, 2.0), _point(0.0, 6.0, 6.0)),
    )
    plan = plan_from_trajectory(command, anchor_us=_us(1.0))
    # 2 m at 2 m/s, then 4 m at a mean of 4 m/s.
    assert plan.timestamps_us == [1_000_000, 2_000_000, 3_000_000]


def test_an_untimed_segment_length_is_math_dist() -> None:
    # plan_from_trajectory spells math.dist as math.hypot of the differences
    # (Codon's math has no dist()): the two give the same number.
    start = Pose.from_xyz_yaw(0.1, 0.7, 5.0, 0.0)
    end = Pose.from_xyz_yaw(3.3, -2.9, -1.0, 0.0)
    command = TrajectoryCommand(
        stamp_us=0,
        points=(
            TrajectoryCommandPoint(0, start, 1.0),
            TrajectoryCommandPoint(0, end, 1.0),
        ),
    )
    plan = plan_from_trajectory(command, anchor_us=0)
    length = math.dist(start.position[:2], end.position[:2])
    assert plan.timestamps_us == [0, int(round(length * 1_000_000))]


def test_received_is_a_source_name_and_a_command() -> None:
    assert Received == Tuple[str, Command]


def test_an_untimed_trajectory_ends_where_it_stops() -> None:
    command = TrajectoryCommand(
        stamp_us=0,
        points=(_point(0.0, 0.0, 1.0), _point(0.0, 1.0, 0.0), _point(0.0, 2.0, 0.0)),
    )
    plan = plan_from_trajectory(command, anchor_us=0)
    # The second segment has no speed to be timed by: the plan ends before it.
    assert len(plan) == 2
