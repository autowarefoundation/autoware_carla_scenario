"""Two init actions on one control domain both run, the last one standing (#1015).

``run_init`` used to tick every init action before settling any of them, and
settling is what gives an action's control domains back.  A second action on
the same domain of the same entity -- the ego's own goal and an init Set Goal
card, two speeds handed to one NPC -- then raised ``ControlConflictError``.
Each action is now settled as soon as it has fired.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    EgoConfig,
    Lanelet2Pose,
    SetSpeedAction,
    SpawnTransform,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.routing import RoutingAction
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return False


def _ego_config() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(carla.Transform(carla.Location())),
    )


class _Ego:
    """An ego that plans its own route: it records every goal it is handed."""

    use_autopilot = False
    actor = None

    def __init__(self) -> None:
        self.goals: list[tuple[int, float]] = []

    def route_to(
        self,
        world: Any,
        goal: Lanelet2Pose,
        *,
        initial_pose: Any = None,
        ground_projection: Any = None,
        waypoints: Any = (),
    ) -> None:
        self.goals.append((goal.lanelet_id, goal.s))


class _Npc:
    """An NPC vehicle that records every speed it is handed."""

    actor = None

    def __init__(self) -> None:
        self.speeds: list[float] = []

    def set_desired_speed(self, world: Any, speed_kmh: float) -> None:
        self.speeds.append(speed_kmh)


@pytest.fixture
def ego() -> Iterator[_Ego]:
    entity = _Ego()
    register_entity(EGO_ROLE_NAME, entity)
    yield entity
    unregister_entity(EGO_ROLE_NAME)


@pytest.fixture
def npc() -> Iterator[_Npc]:
    entity = _Npc()
    register_entity("npc1", entity)
    yield entity
    unregister_entity("npc1")


def test_two_init_routings_of_the_ego_deliver_both_goals_in_order(ego: _Ego) -> None:
    scenario = _Scenario(_ego_config())
    first = RoutingAction(EGO_ROLE_NAME, Lanelet2Pose(265, 12.5), label="first")
    second = RoutingAction(EGO_ROLE_NAME, Lanelet2Pose(300, 0.0), label="second")
    scenario.register_init(first)
    scenario.register_init(second)
    assert scenario._init_actions == [first, second]

    scenario.run_init(object())  # type: ignore[arg-type]

    # Registration order: the one registered last is the goal the ego keeps.
    assert ego.goals == [(265, 12.5), (300, 0.0)]
    assert first.state is ActionState.COMPLETE
    assert second.state is ActionState.COMPLETE


def test_two_init_speeds_on_one_npc_do_not_conflict(npc: _Npc) -> None:
    scenario = _Scenario(_ego_config())
    first = SetSpeedAction("npc1", 20.0, label="first")
    second = SetSpeedAction("npc1", 35.0, label="second")
    scenario.register_init(first)
    scenario.register_init(second)
    assert scenario._init_actions == [first, second]

    scenario.run_init(object())  # type: ignore[arg-type]

    assert npc.speeds == [20.0, 35.0]
    assert first.state is ActionState.COMPLETE
    assert second.state is ActionState.COMPLETE


def test_a_declarative_ego_goal_and_an_init_set_goal_card_both_run(
    ego: _Ego,
) -> None:
    """The issue's case: the validator's warning describes a run that works,
    and the card, registered after the ego's own goal, is the goal kept."""
    from autoware_carla_scenario.authoring.models import ActionNode, GoalSpec
    from autoware_carla_scenario.authoring.starter import _condition, new_document
    from autoware_carla_scenario.authoring.validator import validate_document
    from autoware_carla_scenario.coordinate import OpenDrivePose
    from autoware_carla_scenario.declarative import DeclarativeScenario

    document = new_document()
    # The starter's lane check reads the map, which this test does not load.
    document.assertions.pass_conditions = [_condition("elapsed_time")]
    assert document.ego is not None
    document.ego.goal = GoalSpec(lanelet_id=265, s=12.5)
    document.actions.append(
        ActionNode(
            type="routing",
            title="Set Goal",
            actor="ego",
            phase="init",
            params={"goal_lanelet_id": 300, "goal_s": 0.0},
        )
    )
    report = validate_document(document)
    assert report.ok
    assert any("replaces" in issue.message for issue in report.warnings)

    scenario = DeclarativeScenario(
        _ego_config(),
        spawn_pose=Lanelet2Pose(lanelet_id=183, s=0.0),
        document=document,
    )

    def _spawn_ego() -> OpenDrivePose:
        # What the real spawn does once it has snapped the ego onto the road.
        scenario.register_route_to_goal()
        return OpenDrivePose(road_id="1", lane_id=-1, s=0.0)

    scenario._prepare_route = lambda: None  # type: ignore[method-assign]
    scenario._setup_ego_spawn = _spawn_ego  # type: ignore[method-assign]
    scenario._spawn_npcs = lambda: None  # type: ignore[method-assign]
    scenario.setup()

    routings = [a for a in scenario._init_actions if isinstance(a, RoutingAction)]
    assert len(routings) == 2
    assert routings[0].label == "route_ego_to_goal"

    scenario.run_init(object())  # type: ignore[arg-type]

    assert ego.goals == [(265, 12.5), (300, 0.0)]
