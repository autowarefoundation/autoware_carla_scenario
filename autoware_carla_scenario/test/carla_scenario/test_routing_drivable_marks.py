"""RoutingAction's ``_drivable``: a waypoint left out as on no lane route
takes its time and condition with it (#2532).

On a fake routing graph: lanelet 99 is reachable from nowhere, so the
waypoint placed on it is left out, and its gate must not reach the pacer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import autoware_carla_scenario.coordinate.map_manager as map_manager
import autoware_carla_scenario.trajectory.relative_lane as relative_lane
from autoware_carla_scenario.actions.routing import RoutingAction, RoutingMode
from autoware_carla_scenario.conditions.always_true import AlwaysTrueCondition
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.trajectory.model import (
    ResolvedTrajectory,
    Trajectory,
    TrajectoryVertex,
)

_UNREACHABLE = 99


class _Graph:
    def shortestPath(self, start: Any, goal: Any) -> Optional[object]:  # noqa: N802
        return None if goal == _UNREACHABLE else object()


class _Layer(dict):  # type: ignore[type-arg]
    def __getitem__(self, key: Any) -> Any:
        return key


def _action() -> RoutingAction:
    trajectory = Trajectory(
        "t",
        [
            TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0, yaw=0.0)),
            TrajectoryVertex(
                CarlaWorldPose(10.0, 0.0, 0.0, yaw=0.0), AlwaysTrueCondition()
            ),
            TrajectoryVertex(CarlaWorldPose(20.0, 0.0, 0.0, yaw=0.0)),
        ],
    )
    action = RoutingAction("npc1", trajectory=trajectory, mode=RoutingMode.DRIVER)
    action._start_pose = CarlaWorldPose(0.0, 0.0, 0.0, yaw=0.0)
    return action


def _drivable(action: RoutingAction, stops: list[Lanelet2Pose]) -> list[Lanelet2Pose]:
    fake = SimpleNamespace(
        lanelet_map=SimpleNamespace(laneletLayer=_Layer()), routing_graph=_Graph()
    )
    with (
        mock.patch.object(map_manager.MapManager, "get_instance", return_value=fake),
        mock.patch.object(
            relative_lane,
            "reference_lane_pose_of",
            return_value=Lanelet2Pose(lanelet_id=1, s=0.0),
        ),
    ):
        return action._drivable(stops)


def _pace(action: RoutingAction) -> list[tuple[int, float, bool]]:
    path = ResolvedTrajectory([0.0, 20.0], [0.0, 0.0], [0.0, 0.0], [None, None])
    actor: Any = SimpleNamespace(get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0))
    action._start_pacing(path, actor, 1.0)
    assert action._pacer is not None
    return [(m.vertex, m.distance, m.gate is not None) for m in action._pacer.marks]


def test_a_waypoint_left_out_is_not_paced_to_nor_waited_at() -> None:
    action = _action()
    action._mark_points = [(1, 10.0, 0.0), (2, 20.0, 0.0)]
    action._stop_vertices = [1, 2]
    stops = [
        Lanelet2Pose(lanelet_id=_UNREACHABLE, s=5.0),
        Lanelet2Pose(lanelet_id=1, s=20.0),
    ]
    kept = _drivable(action, stops)
    assert [(s.lanelet_id, s.s) for s in kept] == [(1, 20.0)]
    assert action._stop_vertices == [2]
    assert action._mark_points == [(2, 20.0, 0.0)]
    # The gated vertex 1 is gone: nothing waits for its condition.
    assert _pace(action) == [(2, 20.0, False)]
    pacer = action._pacer
    assert pacer is not None
    # Nor is an arrival logged for it, even with its condition held.
    pacer.note_open(1, 1.0)
    pacer.observe(20.0, 3.0)
    assert pacer.take_arrivals() == []


def test_a_lanelet2_waypoint_left_out_leaves_the_lane_marks_too() -> None:
    action = _action()
    unreachable = Lanelet2Pose(lanelet_id=_UNREACHABLE, s=5.0)
    goal = Lanelet2Pose(lanelet_id=1, s=20.0)
    action._mark_lanes = [(1, unreachable), (2, goal)]
    action._stop_vertices = [1, 2]
    _drivable(action, [unreachable, goal])
    assert action._mark_lanes == [(2, goal)]


def test_every_waypoint_reachable_keeps_every_mark() -> None:
    action = _action()
    action._mark_points = [(1, 10.0, 0.0), (2, 20.0, 0.0)]
    action._stop_vertices = [1, 2]
    stops = [Lanelet2Pose(lanelet_id=1, s=10.0), Lanelet2Pose(lanelet_id=1, s=20.0)]
    assert _drivable(action, stops) == stops
    assert action._stop_vertices == [1, 2]
    assert _pace(action) == [(1, 10.0, True), (2, 20.0, False)]
