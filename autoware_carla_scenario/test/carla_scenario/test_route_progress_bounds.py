"""A ``route_progress`` whose target lies off the route is warned about (#1385).

An entity's route s is clamped to ``[0, length]``, so a target before the
route's start or past its end is one the rule always or never meets.  The
length is the map's, so only an anchor pinned to a bound -- the route's start
or end, the first segment's start, the last segment's end, a junction that is
the first or last segment -- is judged.  A target exactly on a bound is
judged too, for a strict rule pointing off the route (#1389).
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
        # On a bound, a strict rule pointing off the route never holds (#1389).
        ({"anchor": "start", "value": 0, "rule": "less_than"}, "can never hold"),
        ({"value": 0, "rule": "less_than"}, "can never hold"),
        ({"anchor": "segment:0:start", "value": 0, "rule": "less_than"}, "never"),
        ({"anchor": "end", "value": 0, "rule": "greater_than"}, "can never hold"),
        ({"anchor": "segment:2:end", "value": 0, "rule": "greater_than"}, "never"),
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
        # On a bound, every rule but a strict one pointing off the route can
        # hold: the defaults, the strict rules pointing onto it, and "=".
        {"anchor": "start", "value": 0, "rule": "greater_than_or_equal"},
        {"anchor": "start", "value": 0, "rule": "greater_than"},
        {"anchor": "start", "value": 0, "rule": "less_than_or_equal"},
        {"anchor": "start", "value": 0, "rule": "equal_to"},
        {"anchor": "end", "value": 0, "rule": "less_than_or_equal"},
        {"anchor": "end", "value": 0, "rule": "less_than"},
        {"anchor": "end", "value": 0, "rule": "greater_than_or_equal"},
        {"anchor": "end", "value": 0, "rule": "equal_to"},
        # Off a bound, the length is the map's.
        {"anchor": "junction:0:entry", "value": 0, "rule": "less_than"},
        {"anchor": "junction:0:exit", "value": 0, "rule": "greater_than"},
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


def test_a_strict_rule_off_a_bound_says_which_bound() -> None:
    (start,) = _warnings(_document({"value": 0, "rule": "less_than"}))
    assert "the target is the route's start" in start
    assert '"less than" can never hold' in start
    assert 'Use "less than or equal" (or a positive distance) instead.' in start
    assert "Give a distance" not in start
    (end,) = _warnings(_document({"anchor": "end", "value": 0, "rule": "greater_than"}))
    assert "the target is the route's end" in end
    assert '"greater than" can never hold' in end
    assert 'Use "greater than or equal" (or a negative distance) instead.' in end
    assert "Give a distance" not in end


def test_an_always_true_target_holds_once_the_entity_is_on_the_map() -> None:
    (message,) = _warnings(_document({"value": -15}))
    assert "holds from the first tick the entity is on the map" in message
    assert message.endswith("Give a distance that puts the target on the route.")
