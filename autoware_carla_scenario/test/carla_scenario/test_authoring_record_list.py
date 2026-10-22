"""The ``record_list`` field kind: an ordered list of typed records (#394).

Checked at each place a field kind is handled: coercion, validation, the
references walked through a node's fields, the generated builder, the
inspector's rows and their buttons, and a run's ``param_overrides``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.adversity import AdversityLocation
from autoware_carla_scenario.authoring import codegen
from autoware_carla_scenario.authoring import validator
from autoware_carla_scenario.authoring.compiler import coerce_params
from autoware_carla_scenario.authoring.registry import (
    FieldSpec,
    SelectOption,
    default_params,
    field_holders,
)
from autoware_carla_scenario.declarative import _merge_records
from autoware_carla_scenario.editor.app import create_app
from autoware_carla_scenario.editor.forms import parse_params

LOCATIONS = (SelectOption("any", "Anywhere"), SelectOption("highway", "Highway"))

ROWS = FieldSpec(
    name="adversities",
    label="Adversities",
    kind="record_list",
    default=(),
    item_fields=(
        FieldSpec("location", "Where", "select", "any", LOCATIONS),
        FieldSpec("probability", "Probability", "number", 0.01),
        FieldSpec("enabled", "Enabled", "bool", True),
        FieldSpec("target", "Target", "entity", None, required=False),
    ),
)


def _records(*rows: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


# -- coercion ---------------------------------------------------------------------


def test_each_record_is_coerced_as_a_node_s_params_are() -> None:
    params = coerce_params(
        (ROWS,),
        {
            "adversities": [
                {"location": "highway", "probability": "0.5", "enabled": "on"},
                {"location": "any", "probability": "1", "stale": "x"},
            ]
        },
    )
    assert params["adversities"] == [
        {"location": "highway", "probability": 0.5, "enabled": True, "target": None},
        {"location": "any", "probability": 1.0, "enabled": True, "target": None},
    ]


@pytest.mark.parametrize("unset", [None, ""])
def test_no_records_is_an_empty_list(unset: Any) -> None:
    assert coerce_params((ROWS,), {"adversities": unset}) == {"adversities": []}
    assert coerce_params((ROWS,), {}) == {"adversities": []}


def test_a_default_list_is_not_shared() -> None:
    spec = FieldSpec(
        "rows",
        "Rows",
        "record_list",
        ({"probability": 0.1},),
        item_fields=ROWS.item_fields,
    )
    first, second = default_params((spec,)), default_params((spec,))
    first["rows"][0]["probability"] = 0.9
    assert second["rows"][0] == {"probability": 0.1}


# -- validation -------------------------------------------------------------------


def _issues(value: Any, *, required: bool = True) -> list[tuple[str, str, str]]:
    spec = FieldSpec(
        "adversities",
        "Adversities",
        "record_list",
        (),
        required=required,
        item_fields=ROWS.item_fields,
    )
    out = validator._Collector()
    refs = validator._Refs(entities={"npc1"}, actions=set(), entity_kinds={})
    validator._check_field(out, "nade", spec, {"adversities": value}, refs, None)
    return [(issue.severity, issue.path, issue.message) for issue in out.issues]


def test_each_record_is_validated_by_its_fields() -> None:
    issues = _issues(
        _records(
            {"location": "highway", "probability": 0.1, "target": "npc1"},
            {"location": "moon", "probability": "x", "target": "ghost", "stale": 1},
        )
    )
    assert issues == [
        (
            "error",
            "nade.adversities[1].location",
            "Where must be one of ['any', 'highway'], got 'moon'.",
        ),
        ("error", "nade.adversities[1].probability", "Probability must be a number."),
        (
            "error",
            "nade.adversities[1].target",
            "Target references unknown entity 'ghost'.",
        ),
        (
            "warning",
            "nade.adversities[1].stale",
            "Parameter 'stale' is not used by this type and will be dropped.",
        ),
    ]


def test_a_required_list_needs_an_entry_and_an_optional_one_does_not() -> None:
    assert _issues([]) == [("error", "nade.adversities", "Adversities is required.")]
    assert _issues(None, required=False) == []


def test_records_are_mappings_in_a_list() -> None:
    assert _issues("rows") == [
        ("error", "nade.adversities", "Adversities must be a list of records.")
    ]
    assert _issues([1, 2])[0][2] == "Adversities must be a list of records."


# -- references inside records ----------------------------------------------------


def test_a_reference_in_a_record_is_found_as_a_top_level_one_is() -> None:
    fields = (FieldSpec("actor", "Actor", "entity"), ROWS)
    params = {
        "actor": "npc0",
        "adversities": _records({"target": "npc1"}, {"target": "npc2"}),
    }
    found = [
        holder[spec.name]
        for spec, holder in field_holders(fields, params)
        if spec.kind == "entity"
    ]
    assert found == ["npc0", "npc1", "npc2"]


# -- the generated builder --------------------------------------------------------


@dataclass
class _Adversity:
    location: AdversityLocation
    probability: float
    enabled: bool = True
    target: str | None = None


class _Nade:
    def __init__(self, adversities: list[_Adversity], label: str = "nade") -> None:
        self.adversities = adversities


def _record_build(fields: tuple[FieldSpec, ...] = (ROWS,)) -> Any:
    imports: list[str] = []
    parameters = {p.name: p for p in codegen._parameters(_Nade, {})}
    by_kwarg = {field.name: field for field in fields}
    with pytest.MonkeyPatch.context() as patch:
        # The test's item class lives outside the package: say where it is.
        patch.setattr(
            codegen, "_import_line", lambda obj, name, preferred: f"import {name}"
        )
        builds = codegen._record_builds(
            parameters, by_kwarg, set(), "nade", "..adversity", {}, imports
        )
    return builds, imports


def test_a_record_list_builds_one_item_per_record() -> None:
    (build,), imports = _record_build()
    assert (build.kwarg, build.mode, build.cls, build.over) == (
        "adversities",
        "over",
        "_Adversity",
        "adversities",
    )
    assert dict(build.parts[0].pieces) == {
        # A select lifted into its enum, as at the top level.
        "location": 'AdversityLocation[str(item["location"]).upper()]',
        "probability": 'item["probability"]',
        "enabled": 'item["enabled"]',
        "target": '(str(item["target"]) if item["target"] is not None else None)',
    }
    assert "import _Adversity" in imports


def test_a_record_field_must_reach_the_item_constructor() -> None:
    stray = FieldSpec(
        "adversities",
        "Adversities",
        "record_list",
        (),
        item_fields=(*ROWS.item_fields, FieldSpec("colour", "Colour")),
    )
    with pytest.raises(codegen.GenerationError, match="'colour' reaches no argument"):
        _record_build((stray,))


def test_an_item_constructor_s_required_argument_must_be_fed() -> None:
    thin = FieldSpec(
        "adversities",
        "Adversities",
        "record_list",
        (),
        item_fields=ROWS.item_fields[:1],
    )
    with pytest.raises(codegen.GenerationError, match="probability"):
        _record_build((thin,))


# -- the inspector's rows ---------------------------------------------------------


def _render(rows: list[dict[str, Any]], tmp_path: Path) -> dict[str, str]:
    """The control's rows as the form posts them: every input and select."""
    env = create_app(draft_dir=tmp_path).state.templates.env
    html = env.from_string(
        '{% import "partials/fields.html" as f %}{{ f.control(spec, rows, none) }}'
    ).render(spec=ROWS, rows=rows)
    form: dict[str, str] = {}
    for tag in re.findall(r"<input[^>]*>", html):
        name = re.search(r'name="([^"]+)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if (
            name
            and (value or "checkbox" not in tag)
            and ("checkbox" not in tag or "checked" in tag)
        ):
            form[name.group(1)] = value.group(1) if value else ""
    for name, body in re.findall(
        r'<select name="([^"]+)"[^>]*>([\s\S]*?)</select>', html
    ):
        chosen = re.search(r'<option value="([^"]*)"\s*selected', body)
        form[name] = chosen.group(1) if chosen else ""
    return form


def test_the_rows_post_what_they_show(tmp_path: Path) -> None:
    rows = _records(
        {"location": "highway", "probability": 0.5, "enabled": False},
        {"location": "any", "probability": 0.25, "enabled": True},
    )
    form = _render(rows, tmp_path)
    assert form["adversities.__rows"] == "2"
    assert form["adversities.0.location"] == "highway"
    assert "adversities.0.enabled" not in form  # an unchecked box posts nothing
    assert parse_params((ROWS,), form)["adversities"] == [
        {"location": "highway", "probability": 0.5, "enabled": False, "target": None},
        {"location": "any", "probability": 0.25, "enabled": True, "target": None},
    ]


@pytest.mark.parametrize(
    ("op", "probabilities"),
    [
        ("add", [0.1, 0.2, 0.3, 0.01]),
        ("delete:1", [0.1, 0.3]),
        ("up:2", [0.1, 0.3, 0.2]),
        ("down:0", [0.2, 0.1, 0.3]),
        ("up:0", [0.1, 0.2, 0.3]),  # already first
        ("down:2", [0.1, 0.2, 0.3]),  # already last
    ],
)
def test_the_row_buttons(tmp_path: Path, op: str, probabilities: list[float]) -> None:
    rows = _records(*({"probability": p} for p in (0.1, 0.2, 0.3)))
    form = _render(rows, tmp_path)
    form["adversities.__op"] = op
    parsed = parse_params((ROWS,), form)["adversities"]
    assert [row["probability"] for row in parsed] == probabilities


def test_the_add_button_is_there_with_no_rows(tmp_path: Path) -> None:
    form = _render([], tmp_path)
    assert form == {"adversities.__rows": "0"}
    form["adversities.__op"] = "add"
    assert parse_params((ROWS,), form)["adversities"] == [
        {"location": "any", "probability": 0.01, "enabled": True, "target": None}
    ]


def test_a_form_without_the_rows_keeps_the_default() -> None:
    assert parse_params((ROWS,), {}) == {"adversities": ()}


@pytest.mark.parametrize("op", ["delete:5", "sideways:0", "delete:x"])
def test_a_malformed_row_button_is_refused(op: str) -> None:
    with pytest.raises(ValueError):
        parse_params((ROWS,), {"adversities.__rows": "1", "adversities.__op": op})


@pytest.mark.parametrize("rows", ["-1", "1001", "many"])
def test_a_forged_row_count_is_refused(rows: str) -> None:
    with pytest.raises(ValueError):
        parse_params((ROWS,), {"adversities.__rows": rows})


def test_a_row_that_does_not_parse_says_which() -> None:
    form = {"adversities.__rows": "1", "adversities.0.probability": "lots"}
    with pytest.raises(ValueError, match="row 1"):
        parse_params((ROWS,), form)


# -- param_overrides --------------------------------------------------------------


def test_an_override_reaches_one_record_s_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    records = _records(
        {"location": "any", "probability": 0.1}, {"location": "any", "probability": 0.2}
    )
    _merge_records("nade", "adversities", records, {"1": {"probability": 0.5}, 7: {}})
    assert records == [
        {"location": "any", "probability": 0.1},
        {"location": "any", "probability": 0.5},
    ]
    assert "no record 7 of nade.adversities" in caplog.text
