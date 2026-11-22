"""A condition judged once the run is over is no trigger (#2316).

A Metric condition reads nothing while the run goes on, so ``BaseAction``
refuses it as a trigger and every composition refuses it as a child
(``refuse_post_run``) -- but only once the scenario is built.  The validator
passed such a document, which saved, compiled and exported, and then failed
in ``DeclarativeScenario.setup``.  It now refuses what the runtime refuses,
from the same marker (``IS_POST_RUN``), and the editor stops offering it
where it cannot go.
"""

from __future__ import annotations

import importlib
import re
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.authoring import compiler, registry
from autoware_carla_scenario.authoring._builders_generated import POST_RUN_CONDITIONS
from autoware_carla_scenario.authoring.builders import (
    instantiate_action,
    instantiate_condition,
)
from autoware_carla_scenario.authoring.compiler import BuildContext, compile_document
from autoware_carla_scenario.authoring.models import ScenarioDocument
from autoware_carla_scenario.authoring.validator import (
    ValidationReport,
    validate_document,
)
from autoware_carla_scenario.scenario_base import BaseScenario

_METRIC: dict[str, Any] = {
    "type": "metric",
    "params": {
        "metric": "collision_at_fault",
        "rule": "greater_than_or_equal",
        "value": 1.0,
    },
}
_ELAPSED: dict[str, Any] = {
    "type": "elapsed_time",
    "params": {"duration_seconds": 1.0},
}


def _document(
    trigger: Optional[dict[str, Any]],
    *,
    passes: Optional[list[dict[str, Any]]] = None,
    fails: Optional[list[dict[str, Any]]] = None,
) -> ScenarioDocument:
    spec = registry.get_action_spec("environment")
    assert spec is not None
    action: dict[str, Any] = {
        "id": "weather",
        "type": "environment",
        "params": registry.default_params(spec.fields),
    }
    if trigger is not None:
        action["trigger"] = trigger
    return ScenarioDocument.model_validate(
        {
            "entities": [{"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 242}}],
            "actions": [action],
            "assertions": {
                "pass": passes
                if passes is not None
                else [{"type": "elapsed_time", "params": {"duration_seconds": 5.0}}],
                "fail": fails
                if fails is not None
                else [{"type": "timeout", "params": {"timeout_seconds": 30.0}}],
            },
        }
    )


def _post_run_errors(report: ValidationReport) -> list[str]:
    return [
        issue.path
        for issue in report.errors
        if "judged once the run is over" in issue.message
    ]


def test_post_run_conditions_are_the_marked_classes() -> None:
    """The generated set is the condition types whose class carries
    ``IS_POST_RUN`` -- the marker ``refuse_post_run`` reads."""
    marked = set()
    for spec in registry.condition_specs():
        module, _, name = spec.target.partition(":")
        cls = getattr(
            importlib.import_module(module, "autoware_carla_scenario.authoring"), name
        )
        if getattr(cls, "IS_POST_RUN", False):
            marked.add(spec.type_id)
    assert POST_RUN_CONDITIONS == marked
    assert "metric" in POST_RUN_CONDITIONS


def test_a_metric_trigger_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The issue's document: it validated, and building its action raised."""
    document = _document(_METRIC)
    report = validate_document(document)
    assert not report.ok
    assert _post_run_errors(report) == ["actions[0].trigger"]

    # What the validator now says is what the action says once built.
    monkeypatch.setattr(compiler, "validate_document", lambda _: ValidationReport())
    with pytest.raises(ValueError, match=re.escape("is a MetricCondition")):
        instantiate_action(
            compile_document(document).actions[0], BuildContext(scenario=None)
        )


@pytest.mark.parametrize(
    ("trigger", "path"),
    [
        (
            {"type": "all", "children": [_ELAPSED, _METRIC]},
            "actions[0].trigger.children[1]",
        ),
        (
            {"type": "any", "children": [_METRIC, _ELAPSED]},
            "actions[0].trigger.children[0]",
        ),
        ({"type": "not", "children": [_METRIC]}, "actions[0].trigger.children[0]"),
        (
            {
                "type": "all",
                "children": [
                    _ELAPSED,
                    {
                        "type": "any",
                        "children": [{"type": "not", "children": [_METRIC]}],
                    },
                ],
            },
            "actions[0].trigger.children[1].children[0].children[0]",
        ),
    ],
    ids=["all", "any", "not", "deep"],
)
def test_a_metric_nested_in_a_trigger_is_refused(
    trigger: dict[str, Any], path: str
) -> None:
    report = validate_document(_document(trigger))
    assert not report.ok
    assert _post_run_errors(report) == [path]


@pytest.mark.parametrize(
    "trigger",
    [
        None,
        _ELAPSED,
        {"type": "all", "children": [_ELAPSED, _ELAPSED]},
        {"type": "not", "children": [_ELAPSED]},
    ],
    ids=["none", "elapsed", "all", "not"],
)
def test_other_triggers_are_accepted(trigger: Optional[dict[str, Any]]) -> None:
    report = validate_document(_document(trigger))
    assert report.ok, [f"{i.path}: {i.message}" for i in report.errors]


def test_a_metric_fail_condition_is_accepted() -> None:
    """On its own as a fail condition, which is where it belongs -- and
    where the scenario registers it."""
    document = _document(_ELAPSED, fails=[_METRIC])
    report = validate_document(document)
    assert report.ok, [f"{i.path}: {i.message}" for i in report.errors]
    built = instantiate_condition(
        compile_document(document).fail_conditions[0], BuildContext(scenario=None)
    )
    taken = SimpleNamespace(_fail_conditions=[], _fail_categories=[])
    BaseScenario.register_fail_condition(taken, built)  # type: ignore[arg-type]
    assert taken._fail_conditions == [built]


def test_a_metric_pass_condition_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Metric can fail a run, never pass one: ``register_pass_condition``
    refuses it, which the declarative scenario calls in its setup."""
    document = _document(_ELAPSED, passes=[_ELAPSED, _METRIC])
    report = validate_document(document)
    assert not report.ok
    assert _post_run_errors(report) == ["assertions.pass[1]"]

    monkeypatch.setattr(compiler, "validate_document", lambda _: ValidationReport())
    built = instantiate_condition(
        compile_document(document).pass_conditions[1], BuildContext(scenario=None)
    )
    with pytest.raises(ValueError, match=re.escape("is a MetricCondition")):
        taken = SimpleNamespace(_pass_conditions=[])
        BaseScenario.register_pass_condition(taken, built)  # type: ignore[arg-type]


@pytest.mark.parametrize("verdict", ["pass", "fail"])
@pytest.mark.parametrize(
    ("node", "path"),
    [
        ({"type": "all", "children": [_ELAPSED, _METRIC]}, "children[1]"),
        ({"type": "any", "children": [_METRIC, _ELAPSED]}, "children[0]"),
        ({"type": "not", "children": [_METRIC]}, "children[0]"),
        ({"type": "sticky", "children": [_METRIC]}, "children[0]"),
        (
            {"type": "persistent", "params": {"duration": 1.0}, "children": [_METRIC]},
            "children[0]",
        ),
    ],
    ids=["all", "any", "not", "sticky", "persistent"],
)
def test_a_metric_inside_an_assertions_composition_is_refused(
    monkeypatch: pytest.MonkeyPatch, verdict: str, node: dict[str, Any], path: str
) -> None:
    """A composition reads its children every tick, as a trigger does, and
    refuses a post-run one wherever it sits -- inside an assertion too."""
    slots: dict[str, Any] = {"passes" if verdict == "pass" else "fails": [node]}
    document = _document(_ELAPSED, **slots)
    report = validate_document(document)
    assert not report.ok
    assert _post_run_errors(report) == [f"assertions.{verdict}[0].{path}"]

    monkeypatch.setattr(compiler, "validate_document", lambda _: ValidationReport())
    compiled = compile_document(document)
    (condition,) = (
        compiled.pass_conditions if verdict == "pass" else compiled.fail_conditions
    )
    with pytest.raises(ValueError, match=re.escape("is a MetricCondition")):
        instantiate_condition(condition, BuildContext(scenario=None))


_ROWS = [[100.0, 200.0, None], [110.0, 200.0, None], [120.0, 200.0, None]]


def _waypoint_document(condition: dict[str, Any]) -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "entities": [
                {"id": "ego", "kind": "ego", "spawn": {"lanelet_id": 1}},
                {"id": "npc1", "kind": "vehicle", "spawn": {"lanelet_id": 2}},
            ],
            "actions": [
                {
                    "id": "a1",
                    "type": "routing",
                    "actor": "npc1",
                    "params": {
                        "time_domain": "none",
                        "path_source": "vertices",
                        "vertices": _ROWS,
                    },
                    "advance_conditions": [{"vertex": 2, "condition": condition}],
                }
            ],
            "assertions": {"pass": [_ELAPSED]},
        }
    )


@pytest.mark.parametrize(
    ("condition", "path"),
    [
        (_METRIC, "actions[0].advance_conditions[0].condition"),
        (
            {"type": "all", "children": [_ELAPSED, _METRIC]},
            "actions[0].advance_conditions[0].condition.children[1]",
        ),
    ],
    ids=["whole", "nested"],
)
def test_a_metric_waypoint_condition_is_refused(
    monkeypatch: pytest.MonkeyPatch, condition: dict[str, Any], path: str
) -> None:
    """A vertex waits on its condition every tick, where a Metric reads
    nothing: the entity would wait there for ever.  The vertex refuses one
    as the validator does, rather than build an action that never moves on."""
    document = _waypoint_document(condition)
    report = validate_document(document)
    assert not report.ok
    assert _post_run_errors(report) == [path]

    monkeypatch.setattr(compiler, "validate_document", lambda _: ValidationReport())
    with pytest.raises(ValueError, match=re.escape("is a MetricCondition")):
        instantiate_action(
            compile_document(document).actions[0], BuildContext(scenario=None)
        )


def test_a_waypoint_condition_read_every_tick_is_accepted() -> None:
    """The same card with a condition that holds while the run goes on."""
    document = _waypoint_document(_ELAPSED)
    report = validate_document(document)
    assert report.ok, [f"{i.path}: {i.message}" for i in report.errors]
    instantiate_action(
        compile_document(document).actions[0], BuildContext(scenario=None)
    )


def test_a_trajectory_vertex_refuses_a_metric() -> None:
    """At the runtime API too, for a trajectory written in Python."""
    from autoware_carla_scenario import MetricCondition
    from autoware_carla_scenario.coordinate.poses import CarlaWorldPose
    from autoware_carla_scenario.trajectory.model import TrajectoryVertex

    metric = MetricCondition("collision_at_fault", label="hit")
    with pytest.raises(ValueError, match="wait at this vertex for ever"):
        TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0), metric)
