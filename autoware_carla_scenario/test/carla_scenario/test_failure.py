"""Why a run failed: the failure category of its fail condition (failure.py)."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    AndCondition,
    BaseScenario,
    CollisionCondition,
    ComparisonRule,
    EgoConfig,
    MetricCondition,
    NotCondition,
    PersistentCondition,
    ScenarioResult,
    SpawnTransform,
    StickyCondition,
    TimeoutCondition,
)
from autoware_carla_scenario.batch_report import print_summary
from autoware_carla_scenario.conditions import ConditionStatus
from autoware_carla_scenario.failure import (
    MOVING,
    NEVER_MOVED,
    STALLED,
    Failure,
    FailureCategory,
    progress_detail,
)
from autoware_carla_scenario.metrics import (
    _FAILURE_CATEGORIES,
    OFFROAD_OR_COLLISION,
    STEERING_REVERSALS,
    _Motion,
    _track_progress,
)


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return False


def _scenario() -> _Scenario:
    return _Scenario(
        EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0, y=0, z=0))
            ),
            vehicle_type="vehicle.mini.cooper",
        )
    )


def _collision() -> CollisionCondition:
    return CollisionCondition(label="hit")


def _timeout() -> TimeoutCondition:
    return TimeoutCondition(10.0, label="late")


class TestTheCategoryOfAFailCondition:
    def test_a_condition_that_watches_one_thing_says_what(self) -> None:
        scenario = _scenario()
        scenario.register_fail_condition(_collision())
        scenario.register_fail_condition(_timeout())
        assert scenario.fail_category(0) == FailureCategory.COLLISION
        assert scenario.fail_category(1) == FailureCategory.PROGRESS

    def test_a_wrapper_of_one_condition_says_what_it_wraps(self) -> None:
        scenario = _scenario()
        scenario.register_fail_condition(StickyCondition(_collision()))
        scenario.register_fail_condition(PersistentCondition(_timeout(), 1.0))
        assert scenario.fail_category(0) == FailureCategory.COLLISION
        assert scenario.fail_category(1) == FailureCategory.PROGRESS

    def test_a_combination_says_nothing(self) -> None:
        scenario = _scenario()
        scenario.register_fail_condition(AndCondition([_collision(), _timeout()]))
        scenario.register_fail_condition(NotCondition(_collision()))
        assert scenario.fail_category(0) == FailureCategory.OTHER
        assert scenario.fail_category(1) == FailureCategory.OTHER

    def test_the_category_registered_wins(self) -> None:
        scenario = _scenario()
        scenario.register_fail_condition(
            NotCondition(_collision()), category=FailureCategory.LANE_KEEPING
        )
        scenario.register_fail_condition(
            _collision(), category=FailureCategory.SAFETY_MARGIN
        )
        assert scenario.fail_category(0) == FailureCategory.LANE_KEEPING
        assert scenario.fail_category(1) == FailureCategory.SAFETY_MARGIN

    def test_a_metric_condition_says_what_its_metric_reads(self) -> None:
        scenario = _scenario()
        scenario.register_fail_condition(
            MetricCondition(
                STEERING_REVERSALS, ComparisonRule.GREATER_THAN, 10, label="saw"
            )
        )
        scenario.register_fail_condition(
            MetricCondition(OFFROAD_OR_COLLISION, label="either")
        )
        assert scenario.fail_category(0) == FailureCategory.STABILITY
        # Off the road *or* a collision: no one category.
        assert OFFROAD_OR_COLLISION not in _FAILURE_CATEGORIES
        assert scenario.fail_category(1) == FailureCategory.OTHER

    def test_a_new_run_keeps_the_categories_its_conditions_were_registered_with(
        self,
    ) -> None:
        scenario = _scenario()
        scenario.register_fail_condition(
            NotCondition(_collision()), category=FailureCategory.LANE_KEEPING
        )
        scenario._reset_registrations()  # the first run: taken
        scenario.register_fail_condition(_timeout())  # what the run added
        scenario._reset_registrations()  # the next run: back to the first's
        assert len(scenario._fail_conditions) == len(scenario._fail_categories) == 1
        assert scenario.fail_category(0) == FailureCategory.LANE_KEEPING


class TestProgressDetail:
    @pytest.mark.parametrize(
        ("moved", "still_s", "detail"),
        [(False, 30.0, NEVER_MOVED), (True, 3.0, STALLED), (True, 2.9, MOVING)],
    )
    def test_how_the_ego_ended_a_run_out_of_time(
        self, moved: bool, still_s: float, detail: str
    ) -> None:
        assert progress_detail(moved, still_s) == detail

    def test_the_motion_notes_a_move_and_since_when_the_ego_stands(self) -> None:
        motion = _Motion()
        _track_progress(motion, 0.4, 1.0, 1.0)  # creeping: not still
        assert (motion.moved, motion.still_since) == (False, None)
        _track_progress(motion, 0.4, 2.0, 1.0)  # a creep of 0.8 m, then 1.2 m
        _track_progress(motion, 0.4, 3.0, 1.0)
        assert motion.moved
        _track_progress(motion, 0.1, 4.0, 1.0)
        _track_progress(motion, 0.0, 5.0, 1.0)
        assert motion.still_since == 4.0
        _track_progress(motion, 1.0, 6.0, 1.0)
        assert motion.still_since is None


class TestTheFailureARunRecords:
    @staticmethod
    def _status(index: int, label: str, satisfied: bool) -> ConditionStatus:
        return ConditionStatus(
            label=f"fail[{index}]({label})",
            satisfied=satisfied,
            message="",
            role="fail",
        )

    def test_it_names_the_others_that_held_and_not_itself(self) -> None:
        from autoware_carla_scenario.scenario_runner import _failure

        statuses = [
            ConditionStatus(
                label="pass[0](there)", satisfied=True, message="", role="pass"
            ),
            self._status(0, "hit", True),
            self._status(1, "left", True),
            self._status(2, "late", False),
        ]
        failure = _failure(
            "fail[0](hit)", FailureCategory.COLLISION, 4.5, "crash", statuses
        )
        assert failure.to_dict() == {
            "condition": "fail[0](hit)",
            "category": "collision",
            "at": 4.5,
            "message": "crash",
            "detail": None,
            "also_triggered": ["fail[1](left)"],
        }

    @pytest.mark.parametrize(
        ("progress", "detail"), [((True, 5.0), STALLED), (None, None)]
    )
    def test_a_run_out_of_time_says_how_the_ego_ended_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        progress: Optional[tuple[bool, float]],
        detail: Optional[str],
    ) -> None:
        from autoware_carla_scenario import scenario_runner

        monkeypatch.setattr(scenario_runner, "ego_progress", lambda: progress)
        failure = scenario_runner._failure(
            "fail[1](late)", FailureCategory.PROGRESS, 30.0, "late", []
        )
        assert failure.detail == detail

    def test_a_failure_of_another_version_reads_back(self) -> None:
        failure = Failure.from_dict({"category": "something_new", "message": "?"})
        assert (failure.category, failure.condition, failure.at) == (
            FailureCategory.OTHER,
            "",
            0.0,
        )

    def test_a_fail_condition_added_past_registration_is_inferred(self) -> None:
        scenario = _scenario()
        scenario._fail_conditions.append(TimeoutCondition(1.0, label="late"))
        last = len(scenario._fail_conditions) - 1
        assert scenario.fail_category(last) == FailureCategory.PROGRESS

    def test_the_result_writes_it_only_when_there_is_one(self) -> None:
        failure = Failure("fail[0](hit)", FailureCategory.COLLISION, 1.0, "crash")
        failed = ScenarioResult(False, "crash", 1.0, failure=failure)
        assert failed.to_dict()["failure"]["category"] == "collision"
        assert "failure" not in ScenarioResult(True, "ok", 1.0).to_dict()

    def test_the_summary_prints_the_category(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        failure = Failure(
            "fail[1](late)", FailureCategory.PROGRESS, 30.0, "late", detail=STALLED
        )
        print_summary(
            ["late", "plain"],
            [
                ScenarioResult(False, "late", 30.0, failure=failure),
                MagicMock(
                    spec=[
                        "passed",
                        "message",
                        "elapsed_seconds",
                        "condition_statuses",
                        "to_dict",
                    ],
                    passed=False,
                    message="old",
                    elapsed_seconds=1.0,
                    condition_statuses=[],
                    to_dict=lambda: {"passed": False},
                ),
            ],
            tmp_path,
        )
        out = capsys.readouterr().out
        assert "[progress: stalled]" in out
        # A result without a failure prints none.
        assert out.count("[progress") == 1
