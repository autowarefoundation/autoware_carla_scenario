"""A ``route_progress`` whose target lies off the route is warned about (#1385).

An entity's route s is clamped to ``[0, length]``, so a target before the
route's start or past its end is one the rule always or never meets.  The
length is the map's, so only an anchor pinned to a bound -- the route's start
or end, the first segment's start, the last segment's end, a junction that is
the first or last segment -- is judged.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from autoware_carla_scenario.authoring.models import ScenarioDocument
from autoware_carla_scenario.authoring.validator import validate_document


def _document(
    params: dict[str, Any], segments: Optional[list[dict[str, Any]]] = None
) -> ScenarioDocument:
    return ScenarioDocument.model_validate(
        {
            "route": {
                "segments": segments
                or [
                    {"kind": "lane", "length": {"min": 30, "max": 60}},
                    {"kind": "junction", "turn": "left"},
                    {"kind": "lane", "length": {"max": 40}},
                ],
            },
            "entities": [{"id": "ego", "kind": "ego"}],
            "assertions": {
                "pass": [{"type": "route_progress", "params": params}],
                "fail": [{"type": "timeout", "params": {"timeout_seconds": 60}}],
            },
        }
    )


def _warnings(document: ScenarioDocument) -> list[str]:
    report = validate_document(document)
    assert report.ok, report.errors
    return [i.message for i in report.warnings if "clamped" in i.message]


@pytest.mark.parametrize(
    ("params", "verdict"),
    [
        # Before the start: rising rules always hold, falling ones never.
        ({"anchor": "start", "value": -15}, "holds from the first tick"),
        ({"value": -15, "rule": "greater_than"}, "holds from the first tick"),
        ({"value": -15, "rule": "less_than_or_equal"}, "can never hold"),
        ({"anchor": "start", "value": -15, "rule": "less_than"}, "can never hold"),
        ({"anchor": "segment:0:start", "value": -1}, "holds from the first tick"),
        ({"value": -15, "rule": "equal_to", "tolerance": 0.5}, "can never hold"),
        # Past the end: rising rules never hold, falling ones always.
        ({"anchor": "end", "value": 5}, "can never hold"),
        ({"anchor": "end", "value": 5, "rule": "greater_than"}, "can never hold"),
        ({"anchor": "end", "value": 5, "rule": "less_than"}, "holds from the first"),
        ({"anchor": "segment:2:end", "value": 5}, "can never hold"),
        ({"anchor": "end", "value": 5, "rule": "equal_to"}, "can never hold"),
    ],
)
def test_a_target_off_the_route_is_warned_about(
    params: dict[str, Any], verdict: str
) -> None:
    messages = _warnings(_document({"entity": "ego", **params}))
    assert len(messages) == 1, messages
    assert verdict in messages[0]
    assert "clamped" in messages[0]


@pytest.mark.parametrize(
    "params",
    [
        {"anchor": "start", "value": 0},
        {"anchor": "start", "value": 1000},
        {"anchor": "end", "value": -2},
        {"anchor": "end", "value": 0},
        # The length of the first segment is the map's: 40 m before the
        # junction may well be on the route.
        {"anchor": "junction:0:entry", "value": -40},
        {"anchor": "segment:1:start", "value": -15},
        {"anchor": "junction:0:exit", "value": 30},
        {"anchor": "segment:0:end", "value": -5},
        # An "=" whose tolerance reaches back onto the route can still hold.
        {"anchor": "start", "value": -0.3, "rule": "equal_to", "tolerance": 0.5},
        {"anchor": "end", "value": 0.3, "rule": "equal_to", "tolerance": 0.5},
    ],
)
def test_a_target_that_may_be_on_the_route_is_not(params: dict[str, Any]) -> None:
    assert _warnings(_document({"entity": "ego", **params})) == []


def test_a_junction_at_a_bound_of_the_pattern_is_judged() -> None:
    segments = [{"kind": "junction", "turn": "left"}, {"kind": "lane"}]
    entry = _document({"anchor": "junction:0:entry", "value": -10}, segments)
    assert any("before the route's start" in m for m in _warnings(entry))
    segments = [{"kind": "lane"}, {"kind": "junction", "turn": "right"}]
    exit_ = _document({"anchor": "junction:0:exit", "value": 10}, segments)
    assert any("past the route's end" in m for m in _warnings(exit_))


def test_an_anchor_the_pattern_lacks_is_only_the_error() -> None:
    report = validate_document(_document({"anchor": "junction:3:entry", "value": -5}))
    assert any("has 1 junction(s)" in i.message for i in report.errors)
    assert not [i for i in report.warnings if "clamped" in i.message]
