"""TeraSim's scheduled static adversities: a stalled object, collision objects (#401).

On the follow-trajectory action's fakes (:mod:`.test_follow_trajectory_action`):
a teleport puts the vehicle where it is told.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from autoware_carla_scenario import (
    CarlaWorldPose,
    Lanelet2Pose,
    MapPose,
    ReferenceContext,
    StalledObjectAction,
    TrajectoryFollowingMode,
    collision_objects,
)
from autoware_carla_scenario.actions.follow_trajectory import HIDDEN_DEPTH_M
from autoware_carla_scenario.actions.stalled_object import FOREVER_S
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

from .test_follow_trajectory_action import _Actor, _Entity, _run

AT = CarlaWorldPose(30.0, 4.0, 0.0, yaw=90.0)


@pytest.fixture
def stalled() -> Iterator[_Entity]:
    registered = _Entity(_Actor())
    register_entity("stalled1", registered)
    yield registered
    unregister_entity("stalled1")


def test_it_is_a_held_pose_out_of_the_world_outside_its_window() -> None:
    action = StalledObjectAction("stalled1", AT, 1.0, 3.0)
    assert action.trajectory.vertices[0].position == AT
    assert [v.time for v in action.trajectory.vertices] == [1.0, 3.0]
    assert (action.start_time, action.end_time, action.pose) == (1.0, 3.0, AT)
    # Absolute times, placed every tick, out of the world outside them.
    assert action._time_reference is not None
    assert action._time_reference.domain is ReferenceContext.ABSOLUTE
    assert action._following_mode is TrajectoryFollowingMode.POSITION
    assert action._hidden_outside


def test_without_an_end_it_stays_for_good() -> None:
    action = StalledObjectAction("stalled1", AT)
    assert action.end_time is None
    assert action.trajectory.vertices[-1].time == FOREVER_S


def test_it_appears_stands_still_and_is_removed(stalled: _Entity) -> None:
    actor = stalled.actor
    action = StalledObjectAction("stalled1", AT, 1.0, 2.0)
    _run(action, actor, 0.5)
    assert actor.z == pytest.approx(-HIDDEN_DEPTH_M) and actor.physics is False
    clock = _run(action, actor, 1.0, start=0.5)
    assert (actor.x, actor.y, actor.z) == pytest.approx((30.0, 4.0, 0.0))
    assert actor.yaw == pytest.approx(90.0) and actor.physics is True
    assert actor.speed == pytest.approx(0.0)
    _run(action, actor, 1.0, start=clock)
    assert action.finished
    assert actor.z == pytest.approx(-HIDDEN_DEPTH_M) and actor.physics is False


@pytest.mark.parametrize(
    ("pose", "start", "end"),
    [
        (MapPose(1.0, 2.0), 0.0, None),  # no yaw to stand at
        (AT, -1.0, None),
        (AT, 2.0, 2.0),
    ],
)
def test_what_it_refuses(pose: object, start: float, end: object) -> None:
    with pytest.raises(ValueError):
        StalledObjectAction("stalled1", pose, start, end)  # type: ignore[arg-type]


def test_collision_objects_are_two_stalled_vehicles_of_one_window() -> None:
    first, second = collision_objects(
        "npc1",
        "npc2",
        Lanelet2Pose(10, 20.0),
        Lanelet2Pose(10, 24.0, heading=0.5),
        5.0,
        30.0,
    )
    assert (first.label, second.label) == ("collision_1", "collision_2")
    assert first.pose == Lanelet2Pose(10, 20.0)
    assert second.pose == Lanelet2Pose(10, 24.0, heading=0.5)
    assert {(a.start_time, a.end_time) for a in (first, second)} == {(5.0, 30.0)}
