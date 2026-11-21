"""A blank field means its default everywhere a document is read (#2281).

The compiler reads an absent, empty or ``None`` field as its spec default.
The validator, the cross-field checks and the sweep config a constraint or a
binding is written to read it the same way, so the inspector, the editor's
map preview and the run all agree on what a card does.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoware_carla_scenario.authoring import registry
from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.registry import FieldSpec, default_params
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import (
    _check_constraint,
    _check_field,
    _Collector,
    _Refs,
    validate_document,
)
from autoware_carla_scenario.sweeper.bindings import parse_binding
from autoware_carla_scenario.sweeper.constraints import parse_constraint

_LEAF_CONSTRAINTS = [s for s in registry.constraint_specs() if s.kind == "leaf"]


# ---------------------------------------------------------------------------
# Constraints and bindings: the sweep config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _LEAF_CONSTRAINTS, ids=lambda s: s.type_id)
@pytest.mark.parametrize("blank", ["absent", None, ""])
def test_a_blank_constraint_field_sweeps_as_its_spec_default(
    spec: registry.ConstraintSpec, blank: Any
) -> None:
    params = {} if blank == "absent" else {f.name: blank for f in spec.fields}
    node = ConstraintNode(type=spec.type_id, params=params)
    expected = parse_constraint({"type": spec.type_id, **default_params(spec.fields)})
    assert parse_constraint(node.to_sweep_dict()) == expected


@pytest.mark.parametrize("spec", registry.binding_specs(), ids=lambda s: s.type_id)
def test_an_empty_binding_sweeps_as_its_spec_defaults(
    spec: registry.BindingSpec,
) -> None:
    node = BindingRef(type=spec.type_id, params={})
    expected = {"type": spec.type_id, **default_params(spec.fields), **dict(spec.fixed)}
    assert node.to_sweep_dict() == expected
    parse_binding("x", node.to_sweep_dict())


def test_a_quoted_number_the_validator_accepts_sweeps_as_a_number() -> None:
    node = ConstraintNode(type="lanelet_length", params={"value": "30"})
    out = _Collector()
    _check_constraint(out, "c", node)
    assert [i for i in out.issues if i.severity == "error"] == []
    constraint = parse_constraint(node.to_sweep_dict())
    assert getattr(constraint, "value") == 30.0


def test_quoted_whole_numbers_sweep_as_numbers() -> None:
    node = ConstraintNode(type="on_route", params={"lanelet_ids": ["12", 13]})
    assert node.to_sweep_dict()["lanelet_ids"] == [12, 13]
    binding = BindingRef(type="route_through", params={"depth": " 2 "})
    assert binding.to_sweep_dict()["depth"] == 2


@pytest.mark.parametrize(
    ("type_id", "params"),
    [
        ("in_set", {"values": "${map.no_3d_model_lanelet_ids}"}),
        ("lanelet_length", {"value": "${some.length}"}),
        ("lanelet_length", {"value": "long"}),
        ("has_adjacent", {"include_forks": "maybe"}),
    ],
)
def test_a_reference_or_a_misspelling_is_left_as_written(
    type_id: str, params: dict[str, Any]
) -> None:
    out = ConstraintNode(type=type_id, params=params).to_sweep_dict()
    for key, value in params.items():
        assert out[key] == value


# ---------------------------------------------------------------------------
# Spans: start_s/end_s and start_time/end_time with their defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_id", "params", "path"),
    [
        ("construction_zone", {"end_s": 50.0}, "end_s"),
        ("construction_zone", {"start_s": 300.0}, "end_s"),
        ("construction_zone", {"end_time": -1.0}, "end_time"),
        ("construction_zone", {"start_s": None, "end_s": 50.0}, "end_s"),
        ("urban_construction_zone", {"end_s": 0.0}, "end_s"),
        ("stalled_object", {"end_time": 0.0}, "end_time"),
    ],
)
def test_a_span_is_checked_against_the_default_of_its_other_end(
    type_id: str, params: dict[str, Any], path: str
) -> None:
    document = new_document()
    actor = "npc1" if type_id == "stalled_object" else None
    document.actions.append(
        ActionNode(type=type_id, actor=actor, params={"lanelet_id": 100, **params})
    )
    index = len(document.actions) - 1
    errors = [i.path for i in validate_document(document).errors]
    assert errors == [f"actions[{index}].{path}"]


def test_a_short_zone_from_the_default_start_is_warned_about() -> None:
    document = new_document()
    document.actions.append(
        ActionNode(type="construction_zone", params={"lanelet_id": 100, "end_s": 110.0})
    )
    index = len(document.actions) - 1
    report = validate_document(document)
    assert report.ok, report.errors
    assert any(
        i.path == f"actions[{index}].end_s" and "no work zone" in i.message
        for i in report.warnings
    )


# ---------------------------------------------------------------------------
# An explicit null on a required field with a default
# ---------------------------------------------------------------------------


def _fields_with_defaults() -> list[tuple[str, FieldSpec]]:
    specs: list[
        registry.ActionSpec
        | registry.ConditionSpec
        | registry.ConstraintSpec
        | registry.BindingSpec
    ] = [
        *registry.action_specs(),
        *registry.condition_specs(),
        *registry.constraint_specs(),
        *registry.binding_specs(),
    ]
    return [
        (spec.type_id, field_spec)
        for spec in specs
        for field_spec in spec.fields
        if field_spec.default is not None and field_spec.default != ""
    ]


_REFS = _Refs(entities={"ego", "npc1"}, actions=set())


@pytest.mark.parametrize("blank", [None, "", "  "])
def test_a_blank_field_is_checked_as_its_default(blank: Any) -> None:
    fields = _fields_with_defaults()
    assert fields
    for type_id, field_spec in fields:
        blank_out, default_out = _Collector(), _Collector()
        _check_field(blank_out, "p", field_spec, {field_spec.name: blank}, _REFS, None)
        _check_field(
            default_out,
            "p",
            field_spec,
            {field_spec.name: field_spec.default},
            _REFS,
            None,
        )
        assert blank_out.issues == default_out.issues, (type_id, field_spec.name)


def test_a_null_timeout_validates_and_compiles_to_its_default() -> None:
    document = new_document()
    document.assertions.fail_conditions.append(
        ConditionNode(type="timeout", params={"timeout_seconds": None})
    )
    report = validate_document(document)
    assert report.ok, report.errors
    assert compile_document(document).fail_conditions[-1].params == {
        "timeout_seconds": 30.0
    }


def test_a_null_flag_validates_and_compiles_to_its_default() -> None:
    document = new_document()
    spec = registry.get_action_spec("adversarial_traffic")
    assert spec is not None
    params = {
        **default_params(spec.fields),
        "adversities": [{"kind": "brake_check"}],
        "challenge_npcs": None,
    }
    document.actions.append(ActionNode(type="adversarial_traffic", params=params))
    report = validate_document(document)
    assert report.ok, report.errors
    assert compile_document(document).actions[-1].params["challenge_npcs"] is False


def test_a_null_select_reads_as_its_default_in_the_cross_field_checks() -> None:
    # A null path_source is the default, a goal: it needs its goal lanelet.
    document = new_document()
    document.actions.append(
        ActionNode(
            type="routing",
            actor="npc1",
            params={"path_source": None, "time_domain": None, "mode": None},
        )
    )
    index = len(document.actions) - 1
    errors = [(i.path, i.message) for i in validate_document(document).errors]
    assert errors == [
        (
            f"actions[{index}].goal_lanelet_id",
            "A route to a goal needs its goal lanelet.",
        )
    ]


# ---------------------------------------------------------------------------
# Set Goal's path speed
# ---------------------------------------------------------------------------


def _lanelet_path(speed: Any) -> ScenarioDocument:
    document = new_document()
    document.actions.append(
        ActionNode(
            type="routing",
            actor="npc1",
            params={
                "path_source": "lanelets",
                "lanelet_ids": [100, 101],
                "mode": "follow",
                "time_domain": "relative",
                "speed_kmh": speed,
            },
        )
    )
    return document


@pytest.mark.parametrize("speed", ["", None])
def test_an_empty_path_speed_times_the_path_at_30_kmh(speed: Any) -> None:
    document = _lanelet_path(speed)
    report = validate_document(document)
    assert report.ok, report.errors
    assert compile_document(document).actions[-1].params["speed_kmh"] == 30.0


def test_a_zero_path_speed_cannot_be_timed() -> None:
    document = _lanelet_path(0.0)
    index = len(document.actions) - 1
    errors = [i.path for i in validate_document(document).errors]
    assert errors == [f"actions[{index}].speed_kmh"]


def test_the_path_speed_help_says_what_empty_means() -> None:
    spec = registry.get_action_spec("routing")
    assert spec is not None
    field_spec = next(f for f in spec.fields if f.name == "speed_kmh")
    assert field_spec.default == 30.0
    assert (
        field_spec.help
        == "Times the lanelet path; 0 leaves it untimed; empty uses 30 km/h."
    )
