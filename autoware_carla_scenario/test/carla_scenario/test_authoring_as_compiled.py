"""The validator judges a document as it is compiled and exported (#2579).

* A constraint's or a binding's parameter its spec does not know is warned
  about and left out of the sweep, whose parsers refuse an unknown key.
* An area card's fields are read with their defaults, as the compiler reads
  them, and a field that failed its own check is not reported twice.
* A Set Goal card's goal lanelet of 0 or less is the empty field, unless the
  sweep searches for it.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.hydra_config import build_scenario_config
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    LaneletChoice,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.sweeper.bindings import parse_binding
from autoware_carla_scenario.sweeper.constraints import parse_constraint

# ---------------------------------------------------------------------------
# Unknown parameters of constraints and bindings
# ---------------------------------------------------------------------------


def _with_stray_keys() -> ScenarioDocument:
    document = new_document()
    npc = document.entity("npc1")
    assert npc is not None
    root = npc.spawn.constraints[0]
    adjacent = next(c for c in root.walk() if c.type == "has_adjacent")
    adjacent.params["vlaue_old"] = "left"
    assert npc.spawn.s.binding is not None
    npc.spawn.s.binding.params["ofset"] = 3
    return document


def test_stray_keys_are_warned_about_and_left_out_of_the_sweep() -> None:
    document = _with_stray_keys()
    report = validate_document(document)
    assert report.ok
    warned = [i.path for i in report.warnings if "will be dropped" in i.message]
    assert any(p.endswith(".vlaue_old") for p in warned)
    assert any(p.endswith(".ofset") for p in warned)

    sweep = build_scenario_config(document)["sweep"]
    for constraints in sweep["constraints"].values():
        for constraint in constraints:
            parse_constraint(constraint)
            assert "vlaue_old" not in str(constraint)
    for key, binding in sweep["bindings"].items():
        parse_binding(key, binding)
        assert "ofset" not in binding


def test_a_known_spec_sweeps_only_its_fields() -> None:
    node = ConstraintNode(type="has_adjacent", params={"value": "right", "x": 1})
    out = node.to_sweep_dict()
    assert "x" not in out
    assert out["value"] == "right"
    binding = BindingRef(type="stop_line_offset", params={"offset": 5.0, "x": 1})
    swept = binding.to_sweep_dict()
    assert "x" not in swept
    assert swept["offset"] == 5.0
    parse_binding("x", swept)


def test_an_unknown_type_keeps_its_params_as_written() -> None:
    node = ConstraintNode(type="no_such_constraint", params={"a": 1})
    assert node.to_sweep_dict() == {"type": "no_such_constraint", "a": 1}
    binding = BindingRef(type="no_such_binding", params={"a": 1})
    assert binding.to_sweep_dict() == {"type": "no_such_binding", "a": 1}


# ---------------------------------------------------------------------------
# An area card
# ---------------------------------------------------------------------------


def _area_document(**params: Any) -> ScenarioDocument:
    document = new_document()
    document.assertions.fail_conditions.append(
        ConditionNode(
            id="area",
            type="area_distance",
            params={"entity": "ego", "distance": 5.0, **params},
        )
    )
    return document


def _area_errors(document: ScenarioDocument) -> list[tuple[str, str]]:
    return [
        (i.path, i.message)
        for i in validate_document(document).errors
        if i.object_id == "area"
    ]


@pytest.mark.parametrize("blank", ["", "  ", None])
def test_a_blank_relief_is_its_default(blank: Any) -> None:
    document = _area_document(ground="SURFACE_RELIEF", relief_m=blank)
    assert _area_errors(document) == []
    compile_document(document)


def test_a_blank_ground_is_its_spec_default() -> None:
    document = _area_document(ground="", attributes="")
    assert _area_errors(document) == []
    compile_document(document)


@pytest.mark.parametrize(
    "params",
    [{"relief_m": "abc"}, {"ground": "NOWHERE"}, {"lanelet_ids": ["x"]}],
    ids=["relief", "ground", "lanelets"],
)
def test_a_field_that_failed_its_own_check_is_reported_once(
    params: dict[str, Any],
) -> None:
    errors = _area_errors(_area_document(**params))
    assert len(errors) == 1, errors
    assert errors[0][0].endswith(f".{next(iter(params))}")


def test_an_area_no_map_can_hold_is_still_refused() -> None:
    # Lanelets narrow a lanelet ground only, never the road surface.
    errors = _area_errors(_area_document(ground="SURFACE_RELIEF", lanelet_ids=[1]))
    assert any(message.startswith("Area:") for _, message in errors), errors


# ---------------------------------------------------------------------------
# A Set Goal card's goal lanelet
# ---------------------------------------------------------------------------


def _goal_document(goal: Any) -> ScenarioDocument:
    document = new_document()
    document.actions.append(
        ActionNode(
            id="goal",
            type="routing",
            actor="npc1",
            params={"path_source": "goal", "goal_lanelet_id": goal},
        )
    )
    return document


def _goal_errors(document: ScenarioDocument) -> list[str]:
    return [i.path for i in validate_document(document).errors if i.object_id == "goal"]


@pytest.mark.parametrize("goal", [0, -5, "0", " -5 "])
def test_a_goal_lanelet_of_0_or_less_is_refused(goal: Any) -> None:
    errors = _goal_errors(_goal_document(goal))
    assert len(errors) == 1, errors
    assert errors[0].endswith("].goal_lanelet_id")


def test_a_positive_goal_lanelet_is_accepted() -> None:
    assert _goal_errors(_goal_document(104)) == []


def test_a_searched_goal_lanelet_is_accepted() -> None:
    document = _goal_document(0)
    document.actions[-1].searches["goal_lanelet_id"] = LaneletChoice(
        mode="constraint_search",
        constraints=[ConstraintNode(type="is_junction")],
    )
    assert _goal_errors(document) == []
