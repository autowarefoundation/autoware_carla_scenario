"""``IntersectionPassingScenario``'s judgement of a turn, with CARLA stubbed:
the indicator it must show near the junction, and how often it may steer back."""

from __future__ import annotations

import unittest.mock
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

import typesafe_carla.carla as carla
from autoware_carla_scenario import EgoConfig, Lanelet2Pose, SpawnTransform
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult
from autoware_carla_scenario.examples import intersection_passing as module
from autoware_carla_scenario.examples.configs import IntersectionPassingConfig
from autoware_carla_scenario.examples.intersection_passing import (
    IntersectionPassingScenario,
)
from autoware_carla_scenario.utils.turn_indicator import TurnIndicatorState


class _Switch(BaseCondition):
    """Holds while :attr:`on`; records what it was built with."""

    def __init__(self, *args: Any, label: str, **kwargs: Any) -> None:
        super().__init__(label=label)
        self.args = args
        self.on = False

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        if not self.on:
            return None
        return ScenarioResult(passed=True, message=self.label, elapsed_seconds=elapsed)


class _Turn:
    """A set-up scenario whose nearness to the junction and indicator are
    switched by hand."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        junction: bool = False,
        route: tuple[int, ...] = (411, 207),
        **config: Any,
    ) -> None:
        self.near: Optional[_Switch] = None
        self.inside: Optional[_Switch] = None
        self.signalling: Optional[_Switch] = None

        def near(*args: Any, **kwargs: Any) -> _Switch:
            self.near = _Switch(*args, **kwargs)
            return self.near

        def signalling(*args: Any, **kwargs: Any) -> _Switch:
            self.signalling = _Switch(*args, **kwargs)
            return self.signalling

        def inside(*args: Any, **kwargs: Any) -> _Switch:
            self.inside = _Switch(*args, **kwargs)
            return self.inside

        # The pass condition's roads are the real class's.
        inside.anywhere_on_road = module.EntityLanePositionCondition.anywhere_on_road  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "EntityPositionDistanceCondition", near)
        monkeypatch.setattr(module, "EntityLanePositionCondition", inside)
        monkeypatch.setattr(module, "TurnIndicatorCondition", signalling)

        def in_junction(i: int) -> bool:
            return junction and i == 411

        monkeypatch.setattr(module, "_is_junction_lanelet", in_junction)
        monkeypatch.setattr(
            module,
            "route_road_ids",
            lambda ids: [f"road{i}" for i in ids if not in_junction(i)],
        )
        self.scenario = IntersectionPassingScenario(
            EgoConfig(
                spawn_location=SpawnTransform(carla.Transform(carla.Location())),
                vehicle_type="vehicle.mini.cooper",
            ),
            spawn_pose=Lanelet2Pose(lanelet_id=1, s=5.0),
            config=IntersectionPassingConfig(
                expected_route_lanelet_ids=list(route), **config
            ),
        )
        self.scenario.set_client(MagicMock(), 8000)
        with unittest.mock.patch.object(self.scenario, "_setup_ego_spawn"):
            self.scenario.setup()

    def failure(self, label: str) -> BaseCondition:
        return next(c for c in self.scenario._fail_conditions if c.label == label)

    def passes(self) -> list[str]:
        return [c.label for c in self.scenario._pass_conditions]

    def labels(self) -> list[str]:
        return [c.label for c in self.scenario._fail_conditions]


def _fired(condition: BaseCondition, t: float) -> bool:
    result = condition.check(MagicMock(), t)
    return result is not None and result.passed


def test_unset_a_turn_is_not_judged_by_its_indicator_or_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left")

    assert turn.labels() == ["scenario_timeout"]
    assert turn.near is None and turn.signalling is None


def test_the_indicator_is_the_turn_s_and_measured_from_the_junction_mouth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="right", turn_indicator_before_m=30.0)

    assert "no_right_indicator" in turn.labels()
    assert turn.signalling is not None and turn.near is not None
    assert turn.signalling.args[1] is TurnIndicatorState.RIGHT
    # The start of the route's first lanelet -- its stop line -- within 30 m.
    _, place, value, rule = turn.near.args
    assert (place.lanelet_id, place.s, value, rule.name) == (
        411,
        0.0,
        30.0,
        "LESS_THAN_OR_EQUAL",
    )


def test_a_route_from_the_approach_is_measured_from_its_junction_lanelet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(
        monkeypatch,
        junction=True,
        route=(100, 411, 207),
        turn_direction="left",
        turn_indicator_before_m=30.0,
    )

    assert turn.near is not None and turn.inside is not None
    assert turn.near.args[1].lanelet_id == 411
    assert turn.inside.args[1].lanelet_id == 411


def test_far_from_the_junction_no_indicator_is_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left", turn_indicator_before_m=30.0)
    no_indicator = turn.failure("no_left_indicator")

    assert not any(_fired(no_indicator, t / 10) for t in range(50))


def test_near_the_junction_without_the_indicator_fails_after_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left", turn_indicator_before_m=30.0)
    no_indicator = turn.failure("no_left_indicator")
    assert turn.near is not None

    turn.near.on = True
    assert not _fired(no_indicator, 0.0)
    assert not _fired(no_indicator, 0.4)
    assert _fired(no_indicator, 0.5)


def test_signalling_near_the_junction_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left", turn_indicator_before_m=30.0)
    no_indicator = turn.failure("no_left_indicator")
    assert turn.near is not None and turn.signalling is not None

    turn.near.on = turn.signalling.on = True
    assert not any(_fired(no_indicator, t / 10) for t in range(50))


def test_a_blink_of_the_command_is_not_a_turn_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left", turn_indicator_before_m=30.0)
    no_indicator = turn.failure("no_left_indicator")
    assert turn.near is not None and turn.signalling is not None

    turn.near.on = turn.signalling.on = True
    assert not _fired(no_indicator, 0.0)
    turn.signalling.on = False
    assert not _fired(no_indicator, 0.1)
    assert not _fired(no_indicator, 0.3)
    turn.signalling.on = True
    assert not _fired(no_indicator, 0.4)
    turn.signalling.on = False
    assert not _fired(no_indicator, 0.5)
    assert not _fired(no_indicator, 0.8)


def test_the_indicator_is_judged_until_the_ego_is_in_the_junction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left", turn_indicator_before_m=30.0)
    no_indicator = turn.failure("no_left_indicator")
    assert turn.near is not None and turn.inside is not None
    assert turn.signalling is not None
    # On the route's first lanelet: the junction's mouth.
    assert turn.inside.args[1] == Lanelet2Pose(lanelet_id=411, s=0.0)

    turn.near.on = turn.signalling.on = True
    assert not _fired(no_indicator, 0.0)
    # Inside, the mouth falls behind and the distance grows: the approach is
    # latched from where it began, and so is the junction.
    turn.near.on = False
    turn.inside.on = True
    assert not _fired(no_indicator, 0.5)
    # A driver cancels the indicator as the turn straightens out.
    turn.inside.on = False
    turn.signalling.on = False
    assert not any(_fired(no_indicator, 1.0 + t / 10) for t in range(30))


def test_steering_back_too_often_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    turn = _Turn(monkeypatch, turn_direction="right", max_steering_reversals=5)

    assert turn.failure("steering_reversals").get_details() == {
        "metric": "steering_reversals",
        "rule": "GREATER_THAN_OR_EQUAL",
        "value": 5.0,
        "unit": "count",
        "tolerance": 1e-6,
        "max_ng_seconds": 0.0,
    }


def test_the_indicator_and_the_steering_are_counted_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(
        monkeypatch,
        turn_direction="left",
        max_steering_reversals=6,
        max_turn_indicator_changes=3,
    )

    assert {
        label: turn.failure(label).get_details()["value"]
        for label in ("steering_reversals", "turn_indicator_changes")
    } == {"steering_reversals": 6.0, "turn_indicator_changes": 3.0}
    assert turn.failure("turn_indicator_changes").get_details()["metric"] == (
        "turn_indicator_changes"
    )


def test_a_turn_is_passed_on_the_road_after_the_junction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _Turn(monkeypatch, turn_direction="left").passes() == [
        "ego_on_road_road411_sticky AND ego_on_road_road207_sticky"
    ]
    assert _Turn(monkeypatch, junction=True, turn_direction="left").passes() == [
        "ego_on_road_road207_sticky"
    ]


#: Each lanelet's road: 208 continues 207's.
_ROADS = {100: "road100", 411: "road411", 207: "road207", 208: "road207"}


@pytest.mark.parametrize(
    "route, roads",
    [
        # A turn: through the junction's lanelet 411, whose road reads as another.
        ([411, 207], ["road207"]),
        # Two lanelets on one road are asserted once.
        ([100, 411, 207, 208], ["road100", "road207"]),
        # A junction alone is all there is to go by.
        ([411], ["road411"]),
    ],
)
def test_the_route_is_asserted_by_the_roads_outside_the_junction(
    monkeypatch: pytest.MonkeyPatch, route: list[int], roads: list[str]
) -> None:
    from autoware_carla_scenario.coordinate import transform

    layer = {
        i: SimpleNamespace(attributes={"turn_direction": "left"} if i == 411 else {})
        for i in _ROADS
    }
    mm = SimpleNamespace(lanelet_map=SimpleNamespace(laneletLayer=layer))
    monkeypatch.setattr(
        transform.MapManager, "get_instance", classmethod(lambda cls: mm)
    )
    monkeypatch.setattr(transform, "lanelet_length", lambda i: 20.0)
    monkeypatch.setattr(
        transform,
        "to_opendrive",
        lambda pose: SimpleNamespace(road_id=_ROADS[pose.lanelet_id]),
    )

    assert transform.route_road_ids(route) == roads


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"turn_indicator_before_m": 30.0},
            "turn_direction left or right",
        ),
        (
            {"turn_direction": "straight", "turn_indicator_before_m": 30.0},
            "turn_direction left or right",
        ),
        (
            {"turn_direction": "left", "turn_indicator_grace_seconds": 0.0},
            "grace",
        ),
        ({"max_steering_reversals": 0}, "at least 1"),
        ({"max_turn_indicator_changes": 0}, "at least 1"),
        (
            {
                "turn_direction": "left",
                "turn_indicator_before_m": 30.0,
                "max_turn_indicator_changes": 2,
            },
            "at least 3",
        ),
    ],
)
def test_a_check_that_cannot_hold_is_refused(
    config: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        IntersectionPassingConfig(**config)
