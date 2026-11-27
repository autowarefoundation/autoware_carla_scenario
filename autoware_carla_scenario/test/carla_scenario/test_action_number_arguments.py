"""Actions' numeric arguments refuse booleans and what never ends (#2532).

``True`` is an ``int`` in Python, so a boolean would otherwise read as 1 m,
1 m/s or 1 s (#2191, #2273 refuse it elsewhere).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import pytest

from autoware_carla_scenario.actions._trajectory_run import TrajectoryRun
from autoware_carla_scenario.actions.communication_loss import (
    CommunicationLossAction,
)
from autoware_carla_scenario.actions.play_motion import PlayMotionAction
from autoware_carla_scenario.actions.routing import RoutingAction
from autoware_carla_scenario.actions.smart_pedestrian import (
    SmartPedestrianControllerAction,
)
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
    # #2597: the siblings of the speeds above.
    "walk_straight.turn_back_ttc_s": lambda v: WalkStraightAction(
        "pedestrian1", 1.4, turn_back_ttc_s=v
    ),
    "walk_straight.turn_back_distance_m": lambda v: WalkStraightAction(
        "pedestrian1", 1.4, turn_back_distance_m=v
    ),
    "walk_straight.turn_back_within_m": lambda v: WalkStraightAction(
        "pedestrian1", 1.4, turn_back_within_m=v
    ),
    "routing.turn_back_ttc_s": lambda v: RoutingAction(
        "pedestrian1", _GOAL, turn_back_ttc_s=v
    ),
    "routing.turn_back_distance_m": lambda v: RoutingAction(
        "pedestrian1", _GOAL, turn_back_distance_m=v
    ),
    "routing.turn_back_within_m": lambda v: RoutingAction(
        "pedestrian1", _GOAL, turn_back_within_m=v
    ),
    "smart_pedestrian.speed_ms": lambda v: SmartPedestrianControllerAction(
        "pedestrian1", "walks on", speed_ms=v
    ),
    "smart_pedestrian.duration_s": lambda v: SmartPedestrianControllerAction(
        "pedestrian1", "walks on", duration_s=v
    ),
}


@pytest.mark.parametrize("name", sorted(_BUILDS))
@pytest.mark.parametrize("boolean", [True, False])
def test_a_boolean_is_refused(name: str, boolean: bool) -> None:
    build = _BUILDS[name]
    build(1.0)  # the number True would read as is taken: the type is refused
    with pytest.raises(ValueError, match=repr(boolean)):
        build(boolean)


@pytest.mark.parametrize("key", ["fps", "duration_s"])
@pytest.mark.parametrize("boolean", [True, False])
def test_a_played_motion_refuses_a_boolean(
    key: str, boolean: bool, tmp_path: Path
) -> None:
    """A boolean fps or duration would read as 0 or 1 (#2597)."""
    from .test_play_motion import RAISED, _write_motion

    motion = _write_motion(tmp_path / "wave.npz", [{}, RAISED])
    taken: dict[str, Any] = {key: 1.0}
    refused: dict[str, Any] = {key: boolean}
    PlayMotionAction("pedestrian1", motion, **taken)
    with pytest.raises(ValueError, match=repr(boolean)):
        PlayMotionAction("pedestrian1", motion, **refused)


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
