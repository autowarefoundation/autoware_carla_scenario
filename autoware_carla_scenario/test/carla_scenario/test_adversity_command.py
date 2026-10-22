"""Adversarial commands, their planned paths, and how a path is driven (#391).

On the synthetic crossroads of :mod:`._route_maps`: the eastbound W approach
is W_IN2 (x -60..-10, y = -1.75), the outer lane W_OUT_LANE2 at y = -5.25 on
its right, and the straight junction lanelet leads on to E_OUT1 (x 10..60).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import _adversarial_execution as execution
from autoware_carla_scenario.adversity import (
    AdversarialCommand,
    AgentFootprint,
    CommandKind,
    PredictedState,
    plan_command,
)
from autoware_carla_scenario.adversity.observation import AgentSnapshot, Scene
from autoware_carla_scenario.route.geometry import MapFeatures
from autoware_carla_scenario.trajectory.model import MapPose, TrajectoryFollowingMode

from . import _route_maps as rm

CAR = AgentFootprint(5.0, 2.0)
TWO_STATES = [
    PredictedState(0.0, 0.0, 0.0, 1.0, 0.0),
    PredictedState(1.0, 0.0, 0.0, 1.0, 1.0),
]
INNER = -1.75
OUTER = -5.25


@pytest.fixture(scope="module")
def features() -> MapFeatures:
    lanelet_map = rm.crossroads()
    return MapFeatures(lanelet_map, rm.routing_graph(lanelet_map))


def _seen(
    features: MapFeatures, x: float, y: float = INNER, speed: float = 10.0
) -> Any:
    agent = AgentSnapshot("npc1", x, y, 0.0, speed, 0.0, CAR)
    return Scene(features, [agent], {}).vehicle("npc1")


# -- commands -------------------------------------------------------------------


def test_the_commands() -> None:
    brake = AdversarialCommand.accelerate(-4.0, 2.0)
    assert (brake.kind, brake.acceleration, brake.duration) == (
        CommandKind.ACCELERATION,
        -4.0,
        2.0,
    )
    assert AdversarialCommand.change_lane("left").duration == 1.0
    with pytest.raises(ValueError, match="left or right"):
        AdversarialCommand.change_lane("up")
    with pytest.raises(ValueError, match="two states"):
        AdversarialCommand.follow([PredictedState(0.0, 0.0, 0.0, 0.0, 0.0)])


# -- planning -------------------------------------------------------------------


def test_braking_keeps_to_the_lane(features: MapFeatures) -> None:
    path = plan_command(
        features, _seen(features, -40.0), AdversarialCommand.accelerate(-4.0, 2.0)
    )
    assert len(path) == 21
    assert [p.t for p in path] == pytest.approx([i * 0.1 for i in range(21)])
    # 10 m/s braking at 4 m/s² for 2 s: 2 m/s and 12 m on.
    assert path[-1].speed == pytest.approx(2.0)
    assert (path[-1].x, path[-1].y) == pytest.approx((-28.0, INNER))
    assert all(abs(p.yaw) < 1e-9 for p in path)


def test_the_speed_is_held_at_the_desired_speed(features: MapFeatures) -> None:
    # 50 km/h without a speed limit tag: 10 m/s at 2 m/s² tops out there.
    path = plan_command(
        features, _seen(features, -50.0), AdversarialCommand.accelerate(2.0, 3.0)
    )
    assert max(p.speed for p in path) == pytest.approx(50.0 / 3.6)


def test_the_path_carries_on_past_the_lanelet_end(features: MapFeatures) -> None:
    # From x = -15, 20 m on: through the straight junction lanelet onto E_OUT1.
    path = plan_command(
        features, _seen(features, -15.0), AdversarialCommand.accelerate(0.0, 2.0)
    )
    assert (path[-1].x, path[-1].y) == pytest.approx((5.0, INNER), abs=0.05)


def test_a_lane_change_crosses_within_its_duration(features: MapFeatures) -> None:
    path = plan_command(
        features, _seen(features, -40.0), AdversarialCommand.change_lane("right")
    )
    assert len(path) == 11
    assert (path[0].x, path[0].y) == pytest.approx((-40.0, INNER))
    # The sideways 3.5 m in 1 s comes out of the 10 m/s.
    forward = math.sqrt(10.0**2 - 3.5**2)
    assert (path[-1].x, path[-1].y) == pytest.approx((-40.0 + forward, OUTER))
    # Heading to the right while it crosses.
    assert path[5].yaw < 0.0


def test_a_lane_change_needs_a_lane_there(features: MapFeatures) -> None:
    with pytest.raises(ValueError, match="no lane on the left"):
        plan_command(
            features, _seen(features, -40.0), AdversarialCommand.change_lane("left")
        )


def test_a_vehicle_off_the_lanes_has_no_lane_to_plan_on(features: MapFeatures) -> None:
    with pytest.raises(ValueError, match="off the lanes"):
        plan_command(
            features,
            _seen(features, -40.0, -30.0),
            AdversarialCommand.accelerate(1.0, 1.0),
        )


def test_a_given_path_is_retimed_from_zero(features: MapFeatures) -> None:
    given = [
        PredictedState(0.0, 0.0, 0.0, 1.0, 5.0),
        PredictedState(1.0, 0.0, 0.0, 1.0, 6.0),
    ]
    path = plan_command(
        features, _seen(features, -40.0), AdversarialCommand.follow(given)
    )
    assert [p.t for p in path] == [0.0, 1.0]
    assert [p.x for p in path] == [0.0, 1.0]


# -- driving --------------------------------------------------------------------


def test_the_vehicle_is_left_moving_for_traffic_to_take_over() -> None:
    run = execution.AdversarialExecution("npc1", TWO_STATES)
    assert run._follow._stop_at_end is False


def test_the_path_becomes_a_timed_trajectory_in_the_map_frame() -> None:
    path = [
        PredictedState(1.0, 2.0, 0.5, 3.0, 4.0),
        PredictedState(2.0, 3.0, 0.6, 3.0, 4.5),
    ]
    trajectory = execution.path_trajectory("cut-in", path)
    assert trajectory.name == "cut-in"
    assert [v.time for v in trajectory.vertices] == [0.0, 0.5]
    assert trajectory.vertices[0].position == MapPose(1.0, 2.0, 0.5)


def test_the_vehicle_is_handed_back_once_the_path_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed: list[Any] = []
    vehicle = SimpleNamespace(resume_traffic=lambda world: resumed.append(world))
    monkeypatch.setattr(execution, "find_vehicle_entity", lambda name: vehicle)
    path = [
        PredictedState(0.0, 0.0, 0.0, 1.0, 0.0),
        PredictedState(1.0, 0.0, 0.0, 1.0, 1.0),
    ]
    run = execution.AdversarialExecution("npc1", path, TrajectoryFollowingMode.POSITION)
    follow = SimpleNamespace(state=ActionState.RUNNING, label="adversity", ticks=0)

    def tick(world: Any, elapsed: float) -> None:
        follow.ticks += 1

    follow.tick = tick
    run._follow = follow  # type: ignore[assignment]
    world = object()
    run.tick(world, 0.0)
    assert not run.done and resumed == []
    follow.state = ActionState.COMPLETE
    run.tick(world, 0.1)
    assert run.done and resumed == [world]
    run.tick(world, 0.2)  # done: nothing more, and handed back once
    assert follow.ticks == 2 and resumed == [world]


def test_a_pedestrian_is_left_where_the_path_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "find_vehicle_entity", lambda name: None)
    path = [
        PredictedState(0.0, 0.0, 0.0, 1.0, 0.0),
        PredictedState(1.0, 0.0, 0.0, 1.0, 1.0),
    ]
    run = execution.AdversarialExecution("p1", path)
    run._follow = SimpleNamespace(  # type: ignore[assignment]
        state=ActionState.COMPLETE, label="adversity", tick=lambda world, elapsed: None
    )
    run.tick(object(), 0.0)
    assert run.done


class _Backend:
    """Records what a traffic backend is asked of a background vehicle (#417)."""

    def __init__(self) -> None:
        self.released: list[Any] = []
        self.resumed: list[Any] = []

    def release_actor(self, world: Any, actor: Any) -> None:
        self.released.append(actor)

    def resume_actor(self, world: Any, actor: Any) -> None:
        self.resumed.append(actor)


def test_a_background_vehicle_is_handed_back_to_its_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "find_vehicle_entity", lambda name: None)
    actor, backend = SimpleNamespace(id=7), _Backend()
    run = execution.AdversarialExecution(
        "background7",
        TWO_STATES,
        actor=actor,
        backend=backend,  # type: ignore[arg-type]
    )
    assert isinstance(run._follow, execution._ActorFollow)
    run._follow = SimpleNamespace(  # type: ignore[assignment]
        state=ActionState.COMPLETE, label="adversity", tick=lambda world, elapsed: None
    )
    run.tick(object(), 0.0)
    assert run.done and backend.resumed == [actor]


def test_a_background_vehicle_s_actor_is_driven_without_an_entity() -> None:
    from autoware_carla_scenario import CarlaWorldPose, Trajectory, TrajectoryVertex
    from autoware_carla_scenario.conditions import TrajectoryTimeCondition
    from autoware_carla_scenario.entity.registry import find_vehicle_entity

    from .test_follow_trajectory_action import _Actor, _run

    actor, backend = _Actor(), _Backend()
    line = Trajectory(
        "line",
        [
            TrajectoryVertex(
                CarlaWorldPose(0.0, 0.0, 0.0), TrajectoryTimeCondition(0.0)
            ),
            TrajectoryVertex(
                CarlaWorldPose(10.0, 0.0, 0.0), TrajectoryTimeCondition(1.0)
            ),
        ],
    )
    follow = execution._ActorFollow(
        "background7",
        actor,  # type: ignore[arg-type]
        backend,  # type: ignore[arg-type]
        line,
        TrajectoryFollowingMode.POSITION,
        "adversity",
    )
    _run(follow, actor, 0.5)
    assert find_vehicle_entity("background7") is None
    assert backend.released == [actor]  # once, when the run began
    assert actor.x == pytest.approx(5.0, abs=0.6)
