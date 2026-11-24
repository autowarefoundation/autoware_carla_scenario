"""The editor's gate fields and row names, moved out of editor.service (#1727)."""

from __future__ import annotations

from typing import Optional

import pytest

from autoware_carla_scenario.editor import service
from autoware_carla_scenario.editor.gate_fields import (
    GATE_CONSISTENCY_FIELDS,
    GATE_FIELDS,
    GATE_ITEM_FIELD,
    GateField,
)
from autoware_carla_scenario.editor.naming import unused_name

_SHARE = GateField("min_success_rate", "Minimum success rate", "%", 90, 100, 1)
_METRES = GateField("max_lateral_spread_m", "Lateral", "m", 0.5, 2.0, 0.05)
_COUNT = GateField("min_runs", "Minimum runs", "", 1, 100, 1, optional=False)


def test_the_service_re_exports_the_moved_names() -> None:
    assert service.GateField is GateField
    assert service.GATE_FIELDS is GATE_FIELDS
    assert service.GATE_CONSISTENCY_FIELDS is GATE_CONSISTENCY_FIELDS
    assert service.GATE_ITEM_FIELD is GATE_ITEM_FIELD
    assert service._unused_name is unused_name


def test_the_fields_are_listed_in_order() -> None:
    assert [f.name for f in GATE_FIELDS] == [
        "min_runs",
        "min_success_rate",
        "min_case_coverage",
        "min_scenario_coverage",
        "min_odd_coverage",
        "max_lateral_spread_m",
        "max_speed_spread_mps",
    ]
    assert [f.name for f in GATE_CONSISTENCY_FIELDS] == [
        "consistency_stations",
        "consistency_min_runs",
    ]
    assert not any(f.optional for f in GATE_CONSISTENCY_FIELDS)
    assert GATE_ITEM_FIELD == GateField("item", "Coverage", "%", 100, 100, 1)


def test_percent_is_whether_the_unit_is_a_share() -> None:
    assert _SHARE.percent
    assert not _METRES.percent
    assert not _COUNT.percent
    assert [f.name for f in GATE_FIELDS if f.percent] == [
        "min_success_rate",
        "min_case_coverage",
        "min_scenario_coverage",
        "min_odd_coverage",
    ]


@pytest.mark.parametrize(
    ("stored", "share", "plain"),
    [
        (0.0, "0", "0"),
        (0.9, "90", "0.9"),
        (0.07, "7", "0.07"),
        (0.29, "29", "0.29"),
        (1.0, "100", "1"),
        (0.123456789, "12.3456789", "0.123456789"),
        (0.1234567, "12.34567", "0.1234567"),
        (0.9999999, "99.99999", "0.9999999"),
        (0.3333333, "33.33333", "0.3333333"),
        (1e-9, "1e-07", "1e-09"),
        (2.5, "250", "2.5"),
        (1234567.0, "123456700", "1234567"),
    ],
)
def test_shown_is_every_digit_stored(stored: float, share: str, plain: str) -> None:
    """No ``:g`` rounding (#2442); a share drops only the scaling's float
    noise, and a whole number its ``.0``."""
    assert _SHARE.shown(stored) == share
    assert _METRES.shown(stored) == plain
    assert _COUNT.shown(stored) == plain


def test_a_count_is_shown_as_the_whole_number_it_is() -> None:
    """``1.23457e+06`` was no whole number to the form, which refused it."""
    assert _COUNT.shown(1234567) == "1234567"
    # Too long for repr(): described, never raised on (value_text.shown).
    assert _COUNT.shown(10**5000) == "an integer of about 10**5000"


@pytest.mark.parametrize(
    ("field", "stored"),
    [
        (_SHARE, 0.1234567),
        (_SHARE, 0.9999999),
        (_SHARE, 0.29),
        (_SHARE, 1e-9),
        (GATE_ITEM_FIELD, 0.9999999),
        (_METRES, 0.123456789),
        (_COUNT, 1234567),
    ],
)
def test_a_value_posted_as_shown_is_kept_as_stored(
    field: GateField, stored: float
) -> None:
    """Posting a row back unchanged -- as every other row is when one is
    edited -- must not move it (#2442)."""
    form = {field.name: field.shown(stored)}
    kept = service._gate_number(form, field, field.name, stored=stored)
    assert kept == stored and type(kept) is type(stored)
    changed = service._gate_number({field.name: "3"}, field, field.name, stored=stored)
    assert changed == (0.03 if field.percent else 3)


@pytest.mark.parametrize(
    ("field", "shown"),
    [(_SHARE, "90"), (_METRES, "0.5"), (_COUNT, "1"), (GATE_ITEM_FIELD, "100")],
)
def test_shown_is_the_default_when_unset(field: GateField, shown: str) -> None:
    unset: Optional[float] = None
    assert field.shown(unset) == shown


def test_unused_name_takes_the_first_free_index() -> None:
    assert unused_name("group", []) == "group_1"
    assert unused_name("group", ["group_1", "group_2", "group_4"]) == "group_3"
    assert unused_name("phase", ("group_1", "phase_2")) == "phase_1"
