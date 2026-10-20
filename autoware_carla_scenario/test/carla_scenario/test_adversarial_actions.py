"""The phase-1 adversarial actions (#393).

On the synthetic crossroads of :mod:`._route_maps` (eastbound W_IN2 at
y = -1.75, the outer lane at y = -5.25); the world and its actors are the
fakes of :mod:`.test_adversity_feasible_condition`, and the path is driven by
a stand-in for :class:`AdversarialExecution` that finishes after three ticks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Optional

import pytest

import autoware_carla_scenario as acs
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import _adversarial_execution as executions
from autoware_carla_scenario.actions.adversarial import (
    AdversarialBrakeAction,
    AdversarialCutInAction,
    AdversarialMergeAction,
    HeadOnAction,
    IgnoreLeaderAction,
    JaywalkAction,
    LaneChangeAbortAction,
    PedestrianRunRedLightAction,
    RunTrafficRuleAction,
    StopCrossingAction,
)
from autoware_carla_scenario.adversity import (
    AdversityKind,
    AdversityLocation,
    PredictedState,
)
from autoware_carla_scenario.conditions.adversity_feasible import (
    AdversityFeasibleCondition,
)
from autoware_carla_scenario.conditions.always_true import AlwaysTrueCondition
from autoware_carla_scenario.trajectory.model import TrajectoryFollowingMode

from .test_adversity_feasible_condition import INNER, _actor, _world
from .test_adversity_feasible_condition import crossroads as crossroads  # noqa: F401

ANY = AdversityLocation.ANY


#: Every stand-in made in the test, in order.
made: list["_Execution"] = []


class _Execution:
    """Stands in for AdversarialExecution: done on its third tick."""

    def __init__(
        self,
        entity_name: str,
        path: list[PredictedState],
        following_mode: TrajectoryFollowingMode,
        label: str,
        actor: Any = None,
        backend: Any = None,
    ) -> None:
        self.entity_name = entity_name
        self.path = path
        self.following_mode = following_mode
        self.label = label
        self.ticks: list[float] = []
        self.closed = False
        made.append(self)

    @property
    def done(self) -> bool:
        return len(self.ticks) >= 3

    def tick(self, world: Any, elapsed: float) -> None:
        self.ticks.append(elapsed)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def execution(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    made.clear()
    monkeypatch.setattr(executions, "AdversarialExecution", _Execution)
    yield


def _brake_world(frame: int = 1) -> Any:
    # npc1 with npc2 close behind, both cruising: a brake check fires.
    return _world(
        _actor(1, "npc1", -40.0, INNER, 10.0),
        _actor(2, "npc2", -55.0, INNER, 10.0),
        frame=frame,
    )


def _run(action: Any, worlds: list[Any], start: float = 0.0) -> list[ActionState]:
    states = []
    for i, world in enumerate(worlds):
        action.tick(world, start + 0.1 * i)
        states.append(action.state)
    return states


def test_the_adversity_s_trigger_is_the_default_condition() -> None:
    action = AdversarialBrakeAction("npc1", AdversityLocation.INTERSECTION)
    trigger = action._condition
    assert isinstance(trigger, AdversityFeasibleCondition)
    assert trigger.kind is AdversityKind.BRAKE_CHECK
    assert trigger.location is AdversityLocation.INTERSECTION
    assert (action.kind, action.location) == (
        AdversityKind.BRAKE_CHECK,
        AdversityLocation.INTERSECTION,
    )


def test_a_triggered_adversity_is_driven_to_its_end() -> None:
    action = AdversarialBrakeAction("npc1", ANY)
    states = _run(action, [_brake_world(frame) for frame in range(1, 5)])
    assert states == [
        ActionState.START_TRANSITION,
        ActionState.RUNNING,
        ActionState.END_TRANSITION,
        ActionState.COMPLETE,
    ]
    (run,) = made
    assert run.entity_name == "npc1" and run.label == "adversarial_brake"
    assert run.following_mode is TrajectoryFollowingMode.FOLLOW
    # Planned from the scene it fired on: braking from 10 m/s in its lane.
    assert run.path[0].x == pytest.approx(-40.0) and run.path[0].t == 0.0
    assert run.path[-1].speed < run.path[0].speed
    assert run.ticks == pytest.approx([0.0, 0.1, 0.2])
    assert action.finished


def test_nothing_fires_where_the_adversity_may_not() -> None:
    action = AdversarialBrakeAction("npc1", AdversityLocation.HIGHWAY)
    assert _run(action, [_brake_world(1), _brake_world(2)]) == [ActionState.STANDBY] * 2
    assert made == []


def test_a_given_condition_fires_it_and_the_trigger_is_asked_then(
    caplog: pytest.LogCaptureFixture,
) -> None:
    held = AdversarialBrakeAction("npc1", ANY, AlwaysTrueCondition(), label="held")
    assert not isinstance(held._condition, AdversityFeasibleCondition)
    _run(held, [_brake_world(1), _brake_world(2), _brake_world(3)])
    assert len(made) == 1

    # No follower: the trigger does not hold, and the run ends with nothing done.
    alone = _world(_actor(1, "npc1", -40.0, INNER, 10.0), frame=9)
    unheld = AdversarialBrakeAction("npc1", ANY, AlwaysTrueCondition(), label="unheld")
    with caplog.at_level(logging.WARNING):
        states = _run(unheld, [alone, alone, alone])
    assert states[-1] is ActionState.COMPLETE
    assert len(made) == 1
    assert "'npc1' cannot brake check now" in caplog.text


def test_a_command_with_no_path_to_plan_does_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A cut-in into the outer lane that the planner refuses.
    def refuse(*args: Any, **kwargs: Any) -> list[PredictedState]:
        raise ValueError("npc1 has no lane on the right")

    world = _world(
        _actor(1, "npc1", -40.0, INNER, 10.0),
        _actor(2, "npc2", -46.0, -5.25, 10.0),
    )
    action = AdversarialCutInAction("npc1", ANY)
    with pytest.MonkeyPatch.context() as patch, caplog.at_level(logging.WARNING):
        patch.setattr(executions, "plan_command", refuse)
        states = _run(action, [world, world, world])
    assert states[-1] is ActionState.COMPLETE and made == []
    assert "no lane on the right; nothing done" in caplog.text


def test_it_repeats_when_not_once() -> None:
    action = AdversarialBrakeAction("npc1", ANY, once=False)
    # Three ticks a run, the next fired on the tick the last one ends.
    _run(action, [_brake_world(frame) for frame in range(1, 8)])
    assert len(made) == 3
    assert all(len(run.ticks) == 3 for run in made[:2])


def test_closing_closes_the_path() -> None:
    action = AdversarialBrakeAction("npc1", ANY)
    _run(action, [_brake_world(1)])
    action.close()
    assert made[0].closed


@pytest.mark.parametrize(
    ("cls", "kind", "label"),
    [
        (AdversarialBrakeAction, AdversityKind.BRAKE_CHECK, "adversarial_brake"),
        (IgnoreLeaderAction, AdversityKind.IGNORE_LEADER, "ignore_leader"),
        (AdversarialCutInAction, AdversityKind.CUT_IN, "adversarial_cut_in"),
        (RunTrafficRuleAction, AdversityKind.TRAFFIC_RULE, "run_traffic_rule"),
        (JaywalkAction, AdversityKind.JAYWALKING, "jaywalk"),
        (LaneChangeAbortAction, AdversityKind.LANE_CHANGE_ABORT, "lane_change_abort"),
        (HeadOnAction, AdversityKind.HEAD_ON, "head_on"),
        (
            PedestrianRunRedLightAction,
            AdversityKind.RUNNING_RED_LIGHT,
            "pedestrian_run_red_light",
        ),
        (StopCrossingAction, AdversityKind.STOP_CROSSING, "stop_crossing"),
        (AdversarialMergeAction, AdversityKind.MERGE, "adversarial_merge"),
    ],
)
def test_each_action_is_its_kind(cls: Any, kind: AdversityKind, label: str) -> None:
    action = cls("npc1", ANY)
    assert action.kind is kind and action.label == label
    assert getattr(acs, cls.__name__) is cls


def test_a_jaywalker_walks_its_seeded_path() -> None:
    def walk(seed: int) -> Optional[PredictedState]:
        made.clear()
        walker = _actor(7, "p1", -40.0, -12.0, 0.0, type_id="walker.pedestrian.0001")
        _run(JaywalkAction("p1", ANY, seed=seed), [_world(walker, frame=seed)])
        return made[0].path[-1] if made else None

    assert walk(3) == walk(3) != walk(4)


def test_the_location_must_be_given() -> None:
    with pytest.raises(TypeError):
        AdversarialBrakeAction("npc1")  # type: ignore[call-arg]
