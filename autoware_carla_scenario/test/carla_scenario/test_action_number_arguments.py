"""Actions' numeric arguments refuse booleans and what never ends (#2532).

``True`` is an ``int`` in Python, so a boolean would otherwise read as 1 m,
1 m/s or 1 s (#2191, #2273 refuse it elsewhere).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest

from autoware_carla_scenario.actions._trajectory_run import TrajectoryRun
from autoware_carla_scenario.actions.communication_loss import (
    CommunicationLossAction,
)
from autoware_carla_scenario.actions.routing import RoutingAction
from autoware_carla_scenario.actions.spawn import SpawnAction
from autoware_carla_scenario.actions import walk_straight
from autoware_carla_scenario.actions.walk_straight import WalkStraightAction
from autoware_carla_scenario.conditions.always_true import AlwaysTrueCondition
from autoware_carla_scenario.conditions.metric import MetricCondition
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.entity.pedestrian_entity import DEFAULT_WALKING_SPEED_MPS
from autoware_carla_scenario.trajectory.model import Trajectory, TrajectoryVertex


def _line() -> Trajectory:
    return Trajectory(
        "line",
        [
            TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0, yaw=0.0)),
            TrajectoryVertex(CarlaWorldPose(20.0, 0.0, 0.0, yaw=0.0)),
        ],
    )


_GOAL = Lanelet2Pose(lanelet_id=1, s=10.0)

_BUILDS: dict[str, Callable[[Any], object]] = {
    "walk_straight.speed_ms": lambda v: WalkStraightAction("pedestrian1", v),
    "spawn.speed": lambda v: SpawnAction("npc1", speed=v),
    "spawn.shift_max": lambda v: SpawnAction("npc1", shift_max=v),
    "spawn.length": lambda v: SpawnAction("npc1", length=v),
    "spawn.width": lambda v: SpawnAction("npc1", width=v),
    "trajectory_run.speed": lambda v: TrajectoryRun("npc1", _line(), speed=v),
    "trajectory_run.initial_distance_offset": lambda v: TrajectoryRun(
        "npc1", _line(), initial_distance_offset=v
    ),
    "routing.walking_speed": lambda v: RoutingAction(
        "pedestrian1", _GOAL, walking_speed=v
    ),
    "routing.speed": lambda v: RoutingAction("npc1", trajectory=_line(), speed=v),
    "routing.initial_distance_offset": lambda v: RoutingAction(
        "npc1", trajectory=_line(), initial_distance_offset=v
    ),
    "communication_loss.duration": lambda v: CommunicationLossAction(duration=v),
}


@pytest.mark.parametrize("name", sorted(_BUILDS))
@pytest.mark.parametrize("boolean", [True, False])
def test_a_boolean_is_refused(name: str, boolean: bool) -> None:
    build = _BUILDS[name]
    build(1.0)  # the number True would read as is taken: the type is refused
    with pytest.raises(ValueError, match=repr(boolean)):
        build(boolean)


@pytest.mark.parametrize("duration", [math.inf, math.nan, -math.inf])
def test_a_communication_loss_without_end_is_refused(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be positive and finite"):
        CommunicationLossAction(duration=duration)


def test_a_communication_loss_refuses_a_metric_condition_as_until() -> None:
    metric = MetricCondition("min_distance_to_obstacle_m", label="m")
    with pytest.raises(ValueError, match="'m' is a MetricCondition"):
        CommunicationLossAction(until=metric)
    # Every tick's condition is still taken.
    action = CommunicationLossAction(until=AlwaysTrueCondition(), duration=2.0)
    assert action.duration == 2.0


def test_the_walking_speed_default_is_one_constant() -> None:
    assert walk_straight.DEFAULT_WALKING_SPEED_MS == DEFAULT_WALKING_SPEED_MPS
    assert WalkStraightAction("pedestrian1")._speed_ms == DEFAULT_WALKING_SPEED_MPS
