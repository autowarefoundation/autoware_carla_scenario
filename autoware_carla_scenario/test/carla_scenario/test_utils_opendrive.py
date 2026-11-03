"""``sanitize_opendrive`` matches whole tag and attribute names (#1300)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from autoware_carla_scenario.utils.opendrive import sanitize_opendrive

_XODR = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku_carla.xodr"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            '<object id="1" subtype="-1" type="-1" s="0"/>',
            '<object id="1" subtype="-1" type="none" s="0"/>',
            id="subtype-before-type",
        ),
        pytest.param(
            '<object type="-1" subtype="-1"/>',
            '<object type="none" subtype="-1"/>',
            id="type-first-then-subtype",
        ),
        pytest.param(
            '<object id="7" name="pole" s="1.5" t="-2" zOffset="0" type="-1"/>',
            '<object id="7" name="pole" s="1.5" t="-2" zOffset="0" type="none"/>',
            id="multiple-attributes",
        ),
        pytest.param(
            '<object\n    id="1"\n    subtype="-1"\n    type="-1"\n/>',
            '<object\n    id="1"\n    subtype="-1"\n    type="none"\n/>',
            id="multi-line-tag",
        ),
        pytest.param(
            '<object\ttype="-1"/>',
            '<object\ttype="none"/>',
            id="tab-before-type",
        ),
    ],
)
def test_an_object_of_type_minus_one_gets_type_none(text: str, expected: str) -> None:
    assert sanitize_opendrive(text) == expected


def test_other_tags_with_type_or_subtype_minus_one_are_left_alone() -> None:
    text = (
        '<signal id="2" subtype="-1" type="-1" s="0"/>'
        '<objects><objectReference id="3" type="-1"/></objects>'
    )
    assert sanitize_opendrive(text) == text


def test_an_attribute_ending_in_color_or_height_is_not_the_attribute() -> None:
    out = sanitize_opendrive(
        '<roadMark edgecolor="x"/><cornerLocal u="0" lineheight="1"/>'
    )
    assert out == (
        '<roadMark color="standard" edgecolor="x"/>'
        '<cornerLocal height="0" u="0" lineheight="1"/>'
    )


def test_a_tag_whose_name_is_followed_by_a_newline_is_still_patched() -> None:
    out = sanitize_opendrive('<roadMark\n  sOffset="0"/><cornerRoad\n  s="0"/>')
    assert (
        out
        == '<roadMark color="standard"\n  sOffset="0"/><cornerRoad height="0"\n  s="0"/>'
    )


def test_no_object_in_the_fixture_map_keeps_type_minus_one() -> None:
    text = _XODR.read_text(encoding="utf-8")
    out = sanitize_opendrive(text)
    objects = re.findall(r"<object\b[^>]*>", out)
    assert objects
    assert not [o for o in objects if re.search(r'\stype="-1"', o)]
    # Only ``type`` changes: every ``subtype`` survives as it was.
    assert out.count('subtype="-1"') == text.count('subtype="-1"')
