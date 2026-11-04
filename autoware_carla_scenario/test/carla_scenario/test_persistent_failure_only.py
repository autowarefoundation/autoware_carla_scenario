"""A persistent condition around a condition that only reports a failure (#1374).

``PersistentCondition`` counts only the ticks its child passes, so around a
collision, off-road, existence or timeout condition -- which fail or say
nothing -- it never fires.  Its semantics stay; building one warns, and so
does the editor's validator.
"""

from __future__ import annotations

import logging
from typing import Optional
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario import (
    AndCondition,
    BaseCondition,
    CollisionCondition,
    ElapsedTimeCondition,
    EntityExistenceCondition,
    NotCondition,
    OffRoadFractionCondition,
    PersistentCondition,
    ScenarioResult,
    StickyCondition,
    TimeoutCondition,
)
from autoware_carla_scenario.authoring._builders_generated import (
    FAILURE_ONLY_CONDITIONS,
)
from autoware_carla_scenario.authoring.models import (
    Assertions,
    ConditionNode,
    Entity,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.conditions.base import reports_only_failure
from autoware_carla_scenario.examples.traffic_light_compliance import (
    RedLightViolationCondition,
)

_LOGGER = "autoware_carla_scenario.conditions.persistent"


def _failure_only() -> list[BaseCondition]:
    return [
        CollisionCondition(label="hit"),
        OffRoadFractionCondition(0.5, label="off_road"),
        EntityExistenceCondition("ego", label="ego_exists"),
        TimeoutCondition(10.0, label="timeout"),
    ]


class _Passing(BaseCondition):
    def __init__(self) -> None:
        super().__init__(label="passing")

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return ScenarioResult(passed=True, message="ok", elapsed_seconds=elapsed)


class TestMarker:
    @pytest.mark.parametrize("condition", _failure_only(), ids=lambda c: c.label)
    def test_failure_only_conditions_say_so(self, condition: BaseCondition) -> None:
        assert reports_only_failure(condition)

    def test_the_red_light_example_says_so(self) -> None:
        assert RedLightViolationCondition.REPORTS_ONLY_FAILURE

    @pytest.mark.parametrize(
        "condition",
        [
            _Passing(),
            ElapsedTimeCondition(1.0, label="elapsed"),
            NotCondition(TimeoutCondition(10.0, label="t")),
            StickyCondition(_Passing()),
            AndCondition([_Passing(), _Passing()]),
        ],
        ids=["passing", "elapsed", "not", "sticky", "and"],
    )
    def test_other_conditions_do_not(self, condition: BaseCondition) -> None:
        assert not reports_only_failure(condition)

    def test_the_editor_knows_the_same_types(self) -> None:
        """Generated from the marker on each spec's target class."""
        assert FAILURE_ONLY_CONDITIONS == {
            "collision",
            "off_road_fraction",
            "entity_existence",
            "timeout",
        }


class TestPersistentWarns:
    @pytest.mark.parametrize("child", _failure_only(), ids=lambda c: c.label)
    def test_around_a_failure_only_child(
        self, child: BaseCondition, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            PersistentCondition(child, 0.5)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert repr(child.label) in message
        assert type(child).__name__ in message
        assert "never fires" in message
        assert "NotCondition(child)" in message
        assert "holds_when_pending=True" in message

    @pytest.mark.parametrize(
        "child",
        [
            _Passing(),
            ElapsedTimeCondition(1.0, label="elapsed"),
            NotCondition(TimeoutCondition(10.0, label="t")),
            NotCondition(_Passing(), holds_when_pending=True),
        ],
        ids=["passing", "elapsed", "not", "not_pending"],
    )
    def test_not_around_another_child(
        self, child: BaseCondition, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            PersistentCondition(child, 0.5)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestSemanticsUnchanged:
    def test_still_never_fires_around_a_timeout(self) -> None:
        condition = PersistentCondition(TimeoutCondition(1.0, label="t"), 0.5)
        world = MagicMock()
        for tick in range(60):
            assert condition.check(world, elapsed=tick * 0.05) is None

    def test_a_plain_not_makes_the_failure_count(self) -> None:
        """The remedy the warning names: the NOT turns the failure into a pass."""
        condition = PersistentCondition(
            NotCondition(TimeoutCondition(1.0, label="t")), 0.5
        )
        world = MagicMock()
        assert condition.check(world, elapsed=0.5) is None
        assert condition.check(world, elapsed=1.0) is None
        result = condition.check(world, elapsed=1.5)
        assert result is not None
        assert result.passed is True


def _document(child: ConditionNode) -> ScenarioDocument:
    return ScenarioDocument(
        id="s",
        entities=[Entity(id="ego", kind="ego")],
        assertions=Assertions(
            fail=[
                ConditionNode(
                    id="p1",
                    type="persistent",
                    params={"duration": 0.5},
                    children=[child],
                )
            ]
        ),
    )


def _persistent_warnings(document: ScenarioDocument) -> list[tuple[str, str | None]]:
    return [
        (issue.path, issue.object_id)
        for issue in validate_document(document).warnings
        if "Persistent never fires" in issue.message
    ]


class TestValidatorWarns:
    @pytest.mark.parametrize("type_id", sorted(FAILURE_ONLY_CONDITIONS))
    def test_on_the_persistent_block(self, type_id: str) -> None:
        document = _document(ConditionNode(id="c1", type=type_id))
        assert _persistent_warnings(document) == [("assertions.fail[0]", "p1")]

    def test_names_the_remedies(self) -> None:
        document = _document(ConditionNode(id="c1", type="collision"))
        (issue,) = [
            i
            for i in validate_document(document).warnings
            if "Persistent never fires" in i.message
        ]
        assert "Collision" in issue.message
        assert "NOT" in issue.message
        assert "holds_when_pending" in issue.message

    def test_not_when_the_child_is_under_a_not(self) -> None:
        document = _document(
            ConditionNode(
                id="n1",
                type="not",
                children=[ConditionNode(id="c1", type="collision")],
            )
        )
        assert _persistent_warnings(document) == []

    def test_not_around_another_condition(self) -> None:
        document = _document(
            ConditionNode(id="e1", type="elapsed_time", params={"duration": 1.0})
        )
        assert _persistent_warnings(document) == []
