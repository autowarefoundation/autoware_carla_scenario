"""A *Set Goal* card's route to a goal is a driver's route only (#1020).

The trajectory a goal stands for is its stops -- ``[via..., goal]``, or the
goal twice without via lanelets -- for the entity's driver to route between.
``follow`` and ``position`` would drive or place the entity along it as if it
were a way (braking it dead on a zero-length one, or teleporting it to the
goal), and *initial_distance_offset* would drop every stop, so the run ended
without routing anywhere.  The validator and the constructor refuse both.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoware_carla_scenario import Lanelet2Pose, RoutingAction, RoutingMode
from autoware_carla_scenario.authoring.builders import instantiate_action
from autoware_carla_scenario.authoring.compiler import BuildContext, compile_document
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Assertions,
    ConditionNode,
    Entity,
    ScenarioDocument,
    SpawnSpec,
)
from autoware_carla_scenario.authoring.validator import validate_document

_GOAL = Lanelet2Pose(104, 60.0)


def _document(params: dict[str, Any]) -> ScenarioDocument:
    return ScenarioDocument(
        id="s",
        entities=[
            Entity(id="ego", kind="ego", spawn=SpawnSpec(lanelet_id=1)),
            Entity(id="npc1", kind="vehicle", spawn=SpawnSpec(lanelet_id=2)),
        ],
        actions=[
            ActionNode(
                id="a1",
                type="routing",
                actor="npc1",
                params={"path_source": "goal", "goal_lanelet_id": 104, **params},
            )
        ],
        assertions=Assertions.model_validate(
            {
                "pass": [
                    ConditionNode(type="elapsed_time", params={"duration_seconds": 5.0})
                ]
            }
        ),
    )


def _errors(document: ScenarioDocument) -> list[tuple[str, str]]:
    return [(i.path, i.message) for i in validate_document(document).errors]


class TestValidator:
    @pytest.mark.parametrize("mode", ["follow", "position"])
    def test_a_goal_route_is_not_followed_or_placed(self, mode: str) -> None:
        errors = _errors(_document({"mode": mode}))
        assert len(errors) == 1
        path, message = errors[0]
        assert path.endswith(".mode")
        assert "write the waypoints out" in message
        assert "Driver" in message

    def test_a_goal_route_has_no_distance_to_start_along(self) -> None:
        errors = _errors(_document({"initial_distance_offset": 5.0}))
        assert len(errors) == 1
        path, message = errors[0]
        assert path.endswith(".initial_distance_offset")
        assert "no trajectory to start along" in message

    def test_both_are_reported(self) -> None:
        paths = [
            path
            for path, _ in _errors(
                _document({"mode": "follow", "initial_distance_offset": 2.0})
            )
        ]
        assert any(p.endswith(".mode") for p in paths)
        assert any(p.endswith(".initial_distance_offset") for p in paths)

    def test_a_driver_goal_route_is_unchanged(self) -> None:
        assert _errors(_document({})) == []
        explicit = _document({"mode": "driver", "initial_distance_offset": 0.0})
        assert _errors(explicit) == []

    def test_written_out_waypoints_keep_follow_and_an_offset(self) -> None:
        document = _document(
            {
                "path_source": "vertices",
                "goal_lanelet_id": None,
                "vertices": [[0.0, 0.0], [10.0, 0.0]],
                "mode": "follow",
                "initial_distance_offset": 3.0,
            }
        )
        assert _errors(document) == []


class TestConstructor:
    @pytest.mark.parametrize("mode", [RoutingMode.FOLLOW, RoutingMode.POSITION])
    def test_a_goal_route_is_not_followed_or_placed(self, mode: RoutingMode) -> None:
        with pytest.raises(ValueError, match="choose driver mode"):
            RoutingAction("npc1", _GOAL, mode=mode)

    def test_a_goal_route_has_no_distance_to_start_along(self) -> None:
        with pytest.raises(ValueError, match="initial_distance_offset"):
            RoutingAction("npc1", _GOAL, initial_distance_offset=5.0)

    def test_a_driver_goal_route_is_unchanged(self) -> None:
        action = RoutingAction(
            "npc1", _GOAL, waypoints=[Lanelet2Pose(103, 5.0)], mode=RoutingMode.DRIVER
        )
        assert action.mode is RoutingMode.DRIVER
        assert action.goal == _GOAL
        assert [v.position for v in action.trajectory.vertices] == [
            Lanelet2Pose(103, 5.0),
            _GOAL,
        ]
        bare = RoutingAction("npc1", _GOAL, initial_distance_offset=0.0)
        assert [v.position for v in bare.trajectory.vertices] == [_GOAL, _GOAL]

    def test_the_card_builds_a_driver_goal_route(self) -> None:
        compiled = compile_document(_document({"goal_s": 60.0}))
        (card,) = [a for a in compiled.actions if a.node.id == "a1"]
        action = instantiate_action(card, BuildContext(scenario=None))
        assert isinstance(action, RoutingAction)
        assert action.mode is RoutingMode.DRIVER
        assert action.goal == _GOAL
