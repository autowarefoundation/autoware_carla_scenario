"""A NOT that fires when its child does (#2430).

By default ``NotCondition`` flips the verdict of a child that has fired and
says nothing while the child says nothing.  An action's trigger and the
runner's pass and fail conditions count any result, so there it fires exactly
when its child does.  Its semantics stay; the editor's validator warns.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from autoware_carla_scenario import ElapsedTimeCondition, NotCondition
from autoware_carla_scenario.authoring._builders_generated import (
    FAILURE_ONLY_CONDITIONS,
)
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Assertions,
    ConditionNode,
    Entity,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.validator import validate_document

_MARK = "This NOT fires when"


def _elapsed(node_id: str = "e1") -> ConditionNode:
    return ConditionNode(
        id=node_id, type="elapsed_time", params={"duration_seconds": 1.0}
    )


def _not(child: ConditionNode, **params: Any) -> ConditionNode:
    return ConditionNode(id="n1", type="not", params=params, children=[child])


def _document(
    *,
    trigger: Optional[ConditionNode] = None,
    passing: Optional[ConditionNode] = None,
    failing: Optional[ConditionNode] = None,
) -> ScenarioDocument:
    actions = []
    if trigger is not None:
        actions.append(
            ActionNode(
                id="a1",
                type="set_speed",
                actor="npc",
                params={"target_speed_kmh": 10.0},
                trigger=trigger,
            )
        )
    return ScenarioDocument(
        id="s",
        entities=[Entity(id="ego", kind="ego"), Entity(id="npc", kind="vehicle")],
        actions=actions,
        assertions=Assertions.model_validate(
            {
                "pass": [passing if passing is not None else _elapsed("p0")],
                "fail": [
                    failing
                    if failing is not None
                    else ConditionNode(
                        id="f0", type="timeout", params={"timeout_seconds": 60.0}
                    )
                ],
            }
        ),
    )


def _not_warnings(document: ScenarioDocument) -> list[tuple[str, str | None]]:
    return [
        (issue.path, issue.object_id)
        for issue in validate_document(document).warnings
        if _MARK in issue.message
    ]


class TestSemanticsUnchanged:
    def test_as_a_trigger_it_fires_when_its_child_does(self) -> None:
        """The probe of the issue: fires at t = 1 s, not at t = 0."""
        condition = NotCondition(ElapsedTimeCondition(1.0, label="t"))
        world = MagicMock()
        assert condition.check(world, 0.0) is None
        result = condition.check(world, 1.0)
        assert result is not None
        assert result.passed is False

    def test_holds_when_pending_holds_while_the_child_has_not_fired(self) -> None:
        condition = NotCondition(
            ElapsedTimeCondition(1.0, label="t"), holds_when_pending=True
        )
        world = MagicMock()
        result = condition.check(world, 0.0)
        assert result is not None
        assert result.passed is True
        assert condition.check(world, 1.0) is None


class TestValidatorWarns:
    def test_as_a_trigger(self) -> None:
        document = _document(trigger=_not(_elapsed()))
        assert _not_warnings(document) == [("actions[0].trigger", "n1")]

    def test_as_a_pass_condition(self) -> None:
        document = _document(passing=_not(_elapsed()))
        assert _not_warnings(document) == [("assertions.pass[0]", "n1")]

    def test_as_a_fail_condition(self) -> None:
        document = _document(failing=_not(_elapsed()))
        assert _not_warnings(document) == [("assertions.fail[0]", "n1")]

    @pytest.mark.parametrize("composition", ["all", "any"])
    def test_under_all_or_any(self, composition: str) -> None:
        trigger = ConditionNode(
            id="c1",
            type=composition,
            children=[_not(_elapsed()), _elapsed("e2")],
        )
        document = _document(trigger=trigger)
        assert _not_warnings(document) == [("actions[0].trigger.children[0]", "n1")]

    def test_holds_when_pending_false_said_explicitly(self) -> None:
        document = _document(trigger=_not(_elapsed(), holds_when_pending=False))
        assert _not_warnings(document) == [("actions[0].trigger", "n1")]

    def test_names_the_child_and_the_remedy(self) -> None:
        document = _document(trigger=_not(_elapsed()))
        (issue,) = [
            i for i in validate_document(document).warnings if _MARK in i.message
        ]
        assert issue.severity == "warning"
        assert "Elapsed" in issue.message
        assert "trigger" in issue.message
        assert "holds_when_pending" in issue.message
        assert "Holds while the child does not" in issue.message

    def test_a_warning_not_an_error(self) -> None:
        report = validate_document(_document(trigger=_not(_elapsed())))
        assert not [i for i in report.errors if i.object_id in ("n1", "e1")]


class TestValidatorDoesNotWarn:
    @pytest.mark.parametrize("value", [True, "true"])
    def test_with_holds_when_pending(self, value: Any) -> None:
        document = _document(trigger=_not(_elapsed(), holds_when_pending=value))
        assert _not_warnings(document) == []

    @pytest.mark.parametrize("type_id", sorted(FAILURE_ONLY_CONDITIONS))
    def test_around_a_failure_only_child(self, type_id: str) -> None:
        """The default turns that child's failure into a pass: what it is for."""
        child = ConditionNode(id="c1", type=type_id)
        document = _document(failing=_not(child), passing=_not(child))
        assert _not_warnings(document) == []

    @pytest.mark.parametrize("wrapper", ["persistent", "sticky"])
    def test_under_persistent_or_sticky(self, wrapper: str) -> None:
        params = {"duration": 0.5} if wrapper == "persistent" else {}
        trigger = ConditionNode(
            id="w1", type=wrapper, params=params, children=[_not(_elapsed())]
        )
        assert _not_warnings(_document(trigger=trigger)) == []

    def test_a_document_without_a_not(self) -> None:
        assert _not_warnings(_document(trigger=_elapsed())) == []
