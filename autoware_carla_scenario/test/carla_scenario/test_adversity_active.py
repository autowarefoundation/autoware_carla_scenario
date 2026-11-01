"""What is under way (adversity_activity), and AdversityActiveCondition over it.

An execution notes its adversity when it first drives and forgets it when it
hands the road user back; the condition asks which of the road users an
EntityFilter takes are carrying out what.  The world is test_entity_filter's
fake: the ego at the origin facing +x.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator

import pytest

from autoware_carla_scenario import (
    AdversityActiveCondition,
    AdversityKind,
    EntityFilter,
    RoadUserKind,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions import _adversarial_execution as execution
from autoware_carla_scenario.adversity.geometry import PredictedState
from autoware_carla_scenario.adversity_activity import (
    ActiveAdversity,
    active_adversities,
    begin_adversity,
    clear_adversities,
    end_adversity,
)
from autoware_carla_scenario.conditions.adversity_active import (
    adversity_kinds_from_field,
)
from autoware_carla_scenario.trajectory.model import TrajectoryFollowingMode

from .test_entity_filter import AHEAD, BEHIND, EGO, LEFT, WALKER, _world

WORLD = _world(EGO, AHEAD, BEHIND, LEFT, WALKER)


@pytest.fixture(autouse=True)
def nothing_under_way() -> Iterator[None]:
    clear_adversities()
    yield
    clear_adversities()


def _under_way(agent: str, kind: str, responding_to: str = "") -> None:
    begin_adversity(ActiveAdversity(agent, kind, "test", responding_to, 1.5))


class TestActivity:
    def test_one_adversity_a_road_user_at_a_time_until_it_ends(self) -> None:
        _under_way("npc1", "cut_in")
        _under_way("npc1", "brake_check")
        _under_way("background4", "ignore_leader")
        assert [(a.agent, a.kind) for a in active_adversities()] == [
            ("npc1", "brake_check"),
            ("background4", "ignore_leader"),
        ]
        end_adversity("npc1")
        end_adversity("nobody")
        assert [a.agent for a in active_adversities()] == ["background4"]

    def test_an_execution_is_under_way_from_its_first_tick_to_its_hand_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            execution,
            "find_vehicle_entity",
            lambda name: SimpleNamespace(resume_traffic=lambda world: None),
        )
        path = [
            PredictedState(0.0, 0.0, 0.0, 1.0, 0.0),
            PredictedState(1.0, 0.0, 0.0, 1.0, 1.0),
        ]
        run = execution.AdversarialExecution(
            "npc1", path, TrajectoryFollowingMode.POSITION, kind="cut_in"
        )
        follow = SimpleNamespace(state=ActionState.RUNNING, label="cut_in")
        follow.tick = lambda world, elapsed: None
        run._follow = follow  # type: ignore[assignment]
        assert active_adversities() == []  # planned, not driven yet
        run.tick(object(), 2.0)
        (noted,) = active_adversities()
        assert (noted.agent, noted.kind, noted.since) == ("npc1", "cut_in", 2.0)
        follow.state = ActionState.COMPLETE
        run.tick(object(), 3.0)
        assert run.done and active_adversities() == []

    def test_a_closed_execution_is_over(self) -> None:
        run = execution.AdversarialExecution(
            "npc1",
            [
                PredictedState(0.0, 0.0, 0.0, 1.0, 0.0),
                PredictedState(1.0, 0.0, 0.0, 1.0, 1.0),
            ],
            kind="brake_check",
        )
        follow = SimpleNamespace(
            state=ActionState.RUNNING, label="b", close=lambda: None
        )
        follow.tick = lambda world, elapsed: None
        run._follow = follow  # type: ignore[assignment]
        run.tick(object(), 0.0)
        run.close()
        assert active_adversities() == []


class TestAdversityActiveCondition:
    def test_it_holds_while_a_road_user_taken_carries_one_out(self) -> None:
        condition = AdversityActiveCondition(label="any")
        assert condition.check(WORLD, 0.0) is None
        _under_way("background4", "brake_check")
        result = condition.check(WORLD, 2.0)
        assert result is not None and result.passed
        assert "'background4' brake check since 1.50s" in result.message
        end_adversity("background4")
        assert condition.check(WORLD, 3.0) is None

    def test_the_filter_says_whose(self) -> None:
        _under_way("background4", "cut_in")  # 15 m behind the ego
        near = AdversityActiveCondition(EntityFilter(max_distance_m=10.0), label="near")
        assert near.check(WORLD, 0.0) is None
        ahead = AdversityActiveCondition(
            EntityFilter(names=("background*",), max_ahead_m=0.0), label="behind"
        )
        assert ahead.check(WORLD, 0.0) is not None
        walkers = AdversityActiveCondition(
            EntityFilter(kind=RoadUserKind.PEDESTRIAN), label="walkers"
        )
        assert walkers.check(WORLD, 0.0) is None
        # One the world does not hold is nobody the filter takes.
        _under_way("gone", "cut_in")
        assert [a.agent for a in ahead.active] == ["background4"]

    def test_the_kinds_say_what(self) -> None:
        _under_way("npc1", "ignore_leader")
        cut_ins = AdversityActiveCondition(kinds=[AdversityKind.CUT_IN], label="c")
        assert cut_ins.check(WORLD, 0.0) is None
        either = AdversityActiveCondition(
            kinds=[AdversityKind.CUT_IN, AdversityKind.IGNORE_LEADER], label="e"
        )
        assert either.check(WORLD, 0.0) is not None

    def test_a_victim_s_response_counts_only_when_asked(self) -> None:
        _under_way("background7", "brake_check", responding_to="npc1")
        assert AdversityActiveCondition(label="a").check(WORLD, 0.0) is None
        responses = AdversityActiveCondition(include_responses=True, label="r")
        result = responses.check(WORLD, 0.0)
        assert result is not None
        assert "responding to npc1's brake check" in result.message

    def test_a_document_names_one_kind_or_any(self) -> None:
        assert adversity_kinds_from_field("ANY") == []
        assert adversity_kinds_from_field("cut_in") == [AdversityKind.CUT_IN]
        with pytest.raises(ValueError, match="adversity kind"):
            adversity_kinds_from_field("drift")
