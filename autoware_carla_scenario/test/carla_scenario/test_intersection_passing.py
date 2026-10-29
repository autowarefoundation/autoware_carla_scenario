"""``IntersectionPassingScenario``'s judgement of a turn, with CARLA stubbed:
the indicator it must show near the junction, and how often it may steer back."""

from __future__ import annotations

import unittest.mock
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

    def __init__(self, monkeypatch: pytest.MonkeyPatch, **config: Any) -> None:
        self.near: Optional[_Switch] = None
        self.signalling: Optional[_Switch] = None

        def near(*args: Any, **kwargs: Any) -> _Switch:
            self.near = _Switch(*args, **kwargs)
            return self.near

        def signalling(*args: Any, **kwargs: Any) -> _Switch:
            self.signalling = _Switch(*args, **kwargs)
            return self.signalling

        monkeypatch.setattr(module, "EntityPositionDistanceCondition", near)
        monkeypatch.setattr(module, "TurnIndicatorCondition", signalling)
        monkeypatch.setattr(module, "_lanelet_start_road_id", lambda i: f"road{i}")
        self.scenario = IntersectionPassingScenario(
            EgoConfig(
                spawn_location=SpawnTransform(carla.Transform(carla.Location())),
                vehicle_type="vehicle.mini.cooper",
            ),
            spawn_pose=Lanelet2Pose(lanelet_id=1, s=5.0),
            config=IntersectionPassingConfig(
                expected_route_lanelet_ids=[411, 207], **config
            ),
        )
        self.scenario.set_client(MagicMock(), 8000)
        with unittest.mock.patch.object(self.scenario, "_setup_ego_spawn"):
            self.scenario.setup()

    def failure(self, label: str) -> BaseCondition:
        return next(c for c in self.scenario._fail_conditions if c.label == label)

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


def test_dropping_the_indicator_in_the_junction_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = _Turn(monkeypatch, turn_direction="left", turn_indicator_before_m=30.0)
    no_indicator = turn.failure("no_left_indicator")
    assert turn.near is not None and turn.signalling is not None

    turn.near.on = turn.signalling.on = True
    assert not _fired(no_indicator, 0.0)
    # Inside the junction the mouth falls behind and the distance grows: the
    # check is latched from where it began.
    turn.near.on = False
    turn.signalling.on = False
    assert not _fired(no_indicator, 1.0)
    assert _fired(no_indicator, 1.5)


def test_steering_back_too_often_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    turn = _Turn(monkeypatch, turn_direction="right", max_steering_reversals=5)

    assert turn.failure("steering_reversals").get_details() == {
        "metric": "steering_reversals",
        "rule": "GREATER_THAN_OR_EQUAL",
        "value": 5.0,
        "unit": "count",
        "tolerance": 1e-6,
    }


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
    ],
)
def test_a_check_that_cannot_hold_is_refused(
    config: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        IntersectionPassingConfig(**config)
