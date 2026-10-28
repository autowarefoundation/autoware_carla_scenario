"""Tests for the ROS 2 ego's command arbitration and plan building."""

from __future__ import annotations

import math
from typing import Tuple

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
from autoware_carla_scenario.ros2.commands import Command, SteerAccelPairer
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


def test_a_newer_command_replaces_the_older_from_the_same_source() -> None:
    arbiter = CommandArbiter([_SI])
    arbiter.observe("si", _control(1.0), _us(1.0))
    arbiter.observe("si", _control(2.0), _us(1.1))
    assert arbiter.select(_us(1.1)).command == _control(2.0)  # type: ignore[union-attr]


def test_an_unknown_source_is_refused() -> None:
    with pytest.raises(KeyError):
        CommandArbiter([_SI]).observe("ghost", _control(), _us(0.0))


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
