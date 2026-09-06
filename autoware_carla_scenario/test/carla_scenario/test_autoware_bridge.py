"""Unit tests for the minimal Autoware bridge contract.

CARLA- and ROS 2-free: exercises the abstract contract via
:class:`FakeAutowareBridge`.
"""

from __future__ import annotations

import math

import pytest

from autoware_carla_scenario.autoware_bridge import BridgePose, FakeAutowareBridge

_INITIAL = BridgePose.from_yaw(x=1.0, y=2.0, z=0.5, yaw=0.5)
_GOAL = BridgePose.from_yaw(x=10.0, y=20.0, z=3.0, yaw=1.5)


def test_not_ready_before_configure() -> None:
    bridge = FakeAutowareBridge()
    assert bridge.is_ready() is False


def test_ready_after_configure() -> None:
    bridge = FakeAutowareBridge()
    bridge.configure(_INITIAL, _GOAL)

    assert bridge.configured_initial_pose == _INITIAL
    assert bridge.configured_goal == _GOAL
    assert bridge.is_ready() is True


def test_ready_honours_delay() -> None:
    bridge = FakeAutowareBridge(ready_after=3)
    bridge.configure(_INITIAL, _GOAL)

    readies = [bridge.is_ready() for _ in range(4)]
    assert readies == [False, False, False, True]


def test_close_is_recorded() -> None:
    bridge = FakeAutowareBridge()
    assert bridge.closed is False
    bridge.close()
    assert bridge.closed is True
    assert "close" in bridge.calls


def test_configure_recorded_in_calls() -> None:
    bridge = FakeAutowareBridge()
    bridge.configure(_INITIAL, _GOAL)
    assert "configure" in bridge.calls


def test_from_rpy_matches_from_yaw_when_level() -> None:
    # A pose with no pitch or roll is the planar case, so the two constructors
    # have to agree -- otherwise the goal Autoware plans to would depend on
    # which one the caller happened to reach for.
    level = BridgePose.from_rpy(x=1.0, y=2.0, z=0.5, roll=0.0, pitch=0.0, yaw=0.5)
    assert level == _INITIAL


def test_from_rpy_carries_pitch_and_roll() -> None:
    # A quarter turn about X alone: w and x are cos/sin of the half angle, and
    # the other two components stay zero.
    quarter = math.sqrt(0.5)
    rolled = BridgePose.from_rpy(
        x=0.0, y=0.0, z=0.0, roll=math.pi / 2, pitch=0.0, yaw=0.0
    )
    assert rolled.rotation.w == pytest.approx(quarter)
    assert rolled.rotation.x == pytest.approx(quarter)
    assert rolled.rotation.y == pytest.approx(0.0)
    assert rolled.rotation.z == pytest.approx(0.0)

    pitched = BridgePose.from_rpy(
        x=0.0, y=0.0, z=0.0, roll=0.0, pitch=math.pi / 2, yaw=0.0
    )
    assert pitched.rotation.w == pytest.approx(quarter)
    assert pitched.rotation.y == pytest.approx(quarter)
    assert pitched.rotation.x == pytest.approx(0.0)
    assert pitched.rotation.z == pytest.approx(0.0)


def test_from_rpy_is_a_unit_quaternion() -> None:
    pose = BridgePose.from_rpy(x=0.0, y=0.0, z=0.0, roll=0.3, pitch=-0.2, yaw=1.1)
    r = pose.rotation
    assert r.w**2 + r.x**2 + r.y**2 + r.z**2 == pytest.approx(1.0)
