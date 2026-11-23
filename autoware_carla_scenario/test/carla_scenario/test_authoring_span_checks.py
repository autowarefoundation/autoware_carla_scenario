"""The validator's route-progress, span and vertex-time checks, moved into the
checked :mod:`~autoware_carla_scenario.authoring.route_progress_checks` and
:mod:`~autoware_carla_scenario.authoring.span_checks` (#2386).

The validator read the document's pydantic nodes; the checks now take the
numbers, the anchor's text, the segment kinds and the vertex times it reads
off them, rewritten for Codon (``sum()`` and ``any()`` as loops, an
``Optional`` bound compared through a ``str``).  These tests hold them to the
code they replace over many generated inputs, issue for issue and in the
same order.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.authoring import validator
from autoware_carla_scenario.authoring.models import ActionNode, ConditionNode
from autoware_carla_scenario.authoring.registry import EQUAL_TO_DEFAULT_TOLERANCE
from autoware_carla_scenario.authoring.route_progress_checks import (
    route_bound,
    route_progress_warning,
)
from autoware_carla_scenario.authoring.span_checks import (
    check_highway_zone_length,
    check_spans,
    check_vertex_times,
)
from autoware_carla_scenario.authoring.validator import _Collector
from autoware_carla_scenario.route.model import parse_anchor

# ---------------------------------------------------------------------------
# The code validator ran before #2386, verbatim but for the names (the
# validator's helpers are reached through the module, so a test can stand in
# for _node_number).
# ---------------------------------------------------------------------------


def _old_check_vertex_times(out: Any, path: str, node: Any, count: int) -> None:
    times = validator._vertex_times(node)
    if not times or count < 2:
        return
    gated = {gate.vertex for gate in node.advance_conditions}
    bare = [vertex for vertex in range(1, count + 1) if vertex not in gated]
    if bare:
        out.error(
            f"{path}.advance_conditions",
            "Either every vertex has a time or none has (a vertex with another "
            f"waypoint condition counts as having one): vertex {bare[0]} has "
            "none.",
            node.id,
        )
    ordered = [
        float(times[vertex])
        for vertex in sorted(times)
        if vertex <= count and validator._coercible_float(times[vertex])
    ]
    if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
        out.error(
            f"{path}.advance_conditions",
            "The vertex times decrease along the path.",
            node.id,
        )


def _old_check_spans(out: Any, path: str, node: Any) -> None:
    if node.type != "stalled_object":
        start_s, end_s = (
            validator._node_number(node, "start_s"),
            validator._node_number(node, "end_s"),
        )
        if start_s is not None and end_s is not None and not end_s > start_s:
            out.error(
                f"{path}.end_s",
                "The zone's end (end_s) must be further along than its start "
                "(start_s).",
                node.id,
            )
        elif start_s is not None and end_s is not None:
            if node.type == "construction_zone":
                _old_check_highway_zone_length(out, path, node, end_s - start_s)
    start, end = (
        validator._node_number(node, "start_time"),
        validator._node_number(node, "end_time"),
    )
    if start is None or end is None:
        return
    if not end > start:
        out.error(
            f"{path}.end_time",
            "The end time must be after the start time; leave it empty for never.",
            node.id,
        )
    elif node.type == "stalled_object" and end > _old_stalled_forever_s():
        out.error(
            f"{path}.end_time",
            f"An end time past {_old_stalled_forever_s():g} s is never: leave it empty.",
            node.id,
        )


def _old_check_highway_zone_length(
    out: Any, path: str, node: Any, length: float
) -> None:
    parts: list[tuple[str, float]] = []
    for key in ("taper_length_in", "buffer_zone_length", "taper_length_out"):
        value = validator._node_number(node, key)
        if value is None:
            return
        parts.append((key, value))
    needed = sum(value for _, value in parts)
    if not length < needed:
        return
    named = " + ".join(f"{key} {value:g} m" for key, value in parts)
    out.warn(
        f"{path}.end_s",
        f"The zone is {length:g} m long (start_s to end_s), shorter than "
        f"{named} = {needed:g} m: it has no work zone, and the tapers and "
        "the buffer are shortened in proportion to fit it.",
        node.id,
    )


def _old_stalled_forever_s() -> float:
    from autoware_carla_scenario.actions.stalled_object import FOREVER_S

    return FOREVER_S


def _old_route_bound(anchor: Any, route: Any) -> Optional[str]:
    text = "start" if anchor is None or not str(anchor).strip() else str(anchor)
    try:
        kind, index, end = parse_anchor(text)
    except ValueError:
        return None
    if kind in ("start", "end"):
        return kind
    if kind == "junction":
        junction_at = [
            k for k, segment in enumerate(route.segments) if segment.kind == "junction"
        ]
        if index >= len(junction_at):
            return None
        index = junction_at[index]
    if index == 0 and end in ("start", "entry"):
        return "start"
    if index == len(route.segments) - 1 and end in ("end", "exit"):
        return "end"
    return None


def _old_check_route_progress_bounds(out: Any, node: Any, route: Any) -> None:
    params = node.params
    value = validator._node_number(node, "value")
    if value is None:
        return
    bound = _old_route_bound(params.get("anchor"), route)
    rule = str(validator._value(node, "rule") or "greater_than_or_equal")
    hint = "Give a distance that puts the target on the route."
    if bound == "start" and value < 0:
        where = f"{-value:g} m before the route's start"
        below = True
    elif bound == "end" and value > 0:
        where = f"{value:g} m past the route's end"
        below = False
    elif bound == "start" and value == 0 and rule == "less_than":
        where = "the route's start"
        below = True
        hint = 'Use "less than or equal" (or a positive distance) instead.'
    elif bound == "end" and value == 0 and rule == "greater_than":
        where = "the route's end"
        below = False
        hint = 'Use "greater than or equal" (or a negative distance) instead.'
    else:
        return
    rising = rule in ("greater_than", "greater_than_or_equal")
    falling = rule in ("less_than", "less_than_or_equal")
    if rule == "equal_to":
        tolerance = validator._node_number(node, "tolerance")
        if tolerance is None:
            tolerance = EQUAL_TO_DEFAULT_TOLERANCE
        if abs(value) <= tolerance:
            return
        always = False
    elif rising or falling:
        always = rising == below
    else:
        return  # an unknown rule is the field check's
    reading = rule.replace("_", " ")
    if always:
        extreme = "behind the route's start" if below else "at the route's end"
        verdict = (
            f'"{reading}" holds from the first tick the entity is on the map, '
            f"even for one {extreme}"
        )
    else:
        verdict = (
            f'"{reading}" can never hold: an action waiting on it never '
            "fires, and a waypoint never departs"
        )
    out.warn(
        node.id,
        f"Route progress: the target is {where}. Progress is clamped to the "
        f"route, from 0 to its length, so {verdict}. {hint}",
        node.id,
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

#: Field values as a card may hold them: numbers, number text, blanks,
#: non-finite numbers and text that is no number.
_RAW: list[Any] = [
    None,
    "",
    "  ",
    0,
    -0.0,
    1,
    -1,
    2.5,
    -7.25,
    30,
    100,
    1e9,
    1e9 + 1,
    2e9,
    10**400,
    "10",
    " 12.5 ",
    "-3",
    "1e9",
    "abc",
    "nan",
    "inf",
    "-inf",
    math.nan,
    math.inf,
    -math.inf,
    True,
    [1],
    {"a": 1},
]

#: Raw floats a stand-in _node_number hands both paths, past the finite
#: filter of the real one.
_FLOATS: list[Optional[float]] = [
    None,
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -15.0,
    5.0,
    1e-7,
    -1e-7,
    1e-6,
    1e9,
    1e9 + 1.0,
    math.nan,
    math.inf,
    -math.inf,
]


def _number(rng: random.Random) -> Any:
    if rng.random() < 0.5:
        return rng.choice(_RAW)
    return rng.choice([rng.uniform(-200, 200), rng.randint(-200, 200)])


def _float(rng: random.Random) -> Optional[float]:
    if rng.random() < 0.6:
        return rng.choice(_FLOATS)
    return rng.uniform(-200, 200)


def _issues(out: _Collector) -> list[Any]:
    """The issues, with a NaN-safe view of each (Issue compares by value)."""
    return [(i.severity, i.path, i.message, i.object_id) for i in out.issues]


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------

_SPAN_TYPES = ["construction_zone", "urban_construction_zone", "stalled_object"]
_SPAN_KEYS = [
    "start_s",
    "end_s",
    "start_time",
    "end_time",
    "taper_length_in",
    "buffer_zone_length",
    "taper_length_out",
]


def _span_node(rng: random.Random) -> Any:
    params = {key: _number(rng) for key in _SPAN_KEYS if rng.random() < 0.8}
    return ActionNode.model_validate(
        {
            "id": f"a{rng.randint(0, 9)}",
            "type": rng.choice(_SPAN_TYPES),
            "params": params,
        }
    )


@pytest.mark.parametrize("seed", range(20))
def test_spans_match_the_old_checks(seed: int) -> None:
    """Over real cards, read with the real _node_number (defaults included)."""
    rng = random.Random(seed)
    for _ in range(250):
        node = _span_node(rng)
        old = _Collector()
        _old_check_spans(old, "actions[0]", node)
        new = _Collector()
        validator._check_spans(new, "actions[0]", node)
        assert _issues(new) == _issues(old)


@pytest.mark.parametrize("seed", range(20))
def test_spans_match_the_old_checks_on_any_float(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over any floats, NaN and infinities included, which the real
    _node_number never returns."""
    rng = random.Random(100 + seed)
    numbers: dict[str, Optional[float]] = {}
    monkeypatch.setattr(validator, "_node_number", lambda node, key: numbers[key])
    for _ in range(250):
        numbers.clear()
        numbers.update({key: _float(rng) for key in _SPAN_KEYS})
        node: Any = SimpleNamespace(id="a", type=rng.choice(_SPAN_TYPES + ["other"]))
        old = _Collector()
        _old_check_spans(old, "p", node)
        new = _Collector()
        validator._check_spans(new, "p", node)
        assert _issues(new) == _issues(old)


def test_a_short_highway_zone_and_a_stalled_object_forever() -> None:
    found = check_spans(
        "p", "construction_zone", 0.0, 50.0, None, None, 30.0, 20.0, 10.0, 0.0
    )
    assert found == [
        (
            "warning",
            "p.end_s",
            "The zone is 50 m long (start_s to end_s), shorter than "
            "taper_length_in 30 m + buffer_zone_length 20 m + taper_length_out "
            "10 m = 60 m: it has no work zone, and the tapers and the buffer are "
            "shortened in proportion to fit it.",
        )
    ]
    assert check_highway_zone_length("p", 50.0, 30.0, None, 10.0) == []
    assert check_spans(
        "p", "stalled_object", None, None, 1.0, 2e9, None, None, None, 1e9
    ) == [("error", "p.end_time", "An end time past 1e+09 s is never: leave it empty.")]


# ---------------------------------------------------------------------------
# Vertex times
# ---------------------------------------------------------------------------

_TIMES: list[Any] = [
    None,
    "",
    0,
    0.0,
    -0.0,
    1,
    2,
    2.5,
    -1,
    -3.5,
    10,
    "3",
    " 4.5 ",
    "x",
    "nan",
    "inf",
    math.nan,
    math.inf,
    -math.inf,
    10**400,
    True,
    [2],
]


def _vertex_node(rng: random.Random) -> Any:
    gates = []
    for _ in range(rng.randint(0, 8)):
        kind = rng.choice(["trajectory_time", "trajectory_time", "distance", "speed"])
        params: dict[str, Any] = {}
        if rng.random() < 0.9:
            params["time"] = (
                rng.choice(_TIMES) if rng.random() < 0.5 else rng.uniform(-5, 20)
            )
        gates.append(
            SimpleNamespace(
                vertex=rng.randint(1, 8),
                condition=SimpleNamespace(type=kind, params=params),
            )
        )
    return SimpleNamespace(id="t", advance_conditions=gates)


@pytest.mark.parametrize("seed", range(20))
def test_vertex_times_match_the_old_checks(seed: int) -> None:
    rng = random.Random(200 + seed)
    for _ in range(300):
        node = _vertex_node(rng)
        count = rng.randint(-1, 9)
        old = _Collector()
        _old_check_vertex_times(old, "actions[1]", node, count)
        new = _Collector()
        validator._check_vertex_times(new, "actions[1]", node, count)
        assert _issues(new) == _issues(old)


def test_vertex_times_of_none_are_kept_but_not_ordered() -> None:
    """A time that is no number still makes the card timed, and is skipped
    when the times are ordered."""
    found = check_vertex_times("p", 3, {1, 2}, {1: 5.0, 2: None, 3: 4.0})
    assert [message for _, _, message in found] == [
        "Either every vertex has a time or none has (a vertex with another "
        "waypoint condition counts as having one): vertex 3 has none.",
        "The vertex times decrease along the path.",
    ]
    assert check_vertex_times("p", 3, {1, 2, 3}, {1: None, 2: 1.0, 5: 0.0}) == []
    assert check_vertex_times("p", 1, set(), {1: 1.0}) == []
    assert check_vertex_times("p", 3, set(), {}) == []


# ---------------------------------------------------------------------------
# Route progress
# ---------------------------------------------------------------------------

_ANCHORS: list[Any] = [
    None,
    "",
    "   ",
    "start",
    "end",
    " start ",
    "END",
    "segment:0:start",
    "segment:0:end",
    "segment:1:start",
    "segment:2:end",
    "segment:3:end",
    "segment:9:end",
    "junction:0:entry",
    "junction:0:exit",
    "junction:1:exit",
    "junction:2:entry",
    "junction:x:exit",
    "segment:-1:end",
    "segment::end",
    "segment:0",
    "lane:0:start",
    "junction:0:start",
    "segment:0:entry",
    "nonsense",
    0,
    1,
    3.5,
    True,
]

_RULES: list[Any] = [
    None,
    "",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "equal_to",
    "not_equal",
    "bogus",
]


def _route(rng: random.Random) -> Any:
    kinds = [
        rng.choice(["lane", "junction", "junction", "", "Junction"])
        for _ in range(rng.randint(0, 5))
    ]
    return SimpleNamespace(segments=[SimpleNamespace(kind=kind) for kind in kinds])


def _anchor(rng: random.Random) -> Any:
    if rng.random() < 0.8:
        return rng.choice(_ANCHORS)
    return f"{rng.choice(['segment', 'junction'])}:{rng.randint(0, 5)}:" + rng.choice(
        ["start", "end", "entry", "exit"]
    )


def test_route_bound_matches_the_old_bound() -> None:
    rng = random.Random(300)
    for _ in range(5000):
        route = _route(rng)
        anchor = _anchor(rng)
        expected = _old_route_bound(anchor, route)
        assert validator._route_bound(anchor, route) == expected
        assert (
            route_bound(
                "" if anchor is None else str(anchor),
                [segment.kind for segment in route.segments],
            )
            == expected
        )


def _progress_node(rng: random.Random) -> Any:
    params: dict[str, Any] = {"entity": "ego"}
    if rng.random() < 0.8:
        params["anchor"] = _anchor(rng)
    if rng.random() < 0.9:
        params["value"] = (
            _number(rng) if rng.random() < 0.5 else rng.choice([0, -15, 5, -1e-7, 1e-7])
        )
    if rng.random() < 0.8:
        params["rule"] = rng.choice(_RULES)
    if rng.random() < 0.5:
        params["tolerance"] = rng.choice([None, 0, 1e-7, 1e-6, 0.5, 20, -1, "x", "nan"])
    return ConditionNode.model_validate(
        {"id": "c", "type": "route_progress", "params": params}
    )


@pytest.mark.parametrize("seed", range(20))
def test_route_progress_matches_the_old_check(seed: int) -> None:
    """Over real conditions, read with the real _node_number and _value."""
    rng = random.Random(400 + seed)
    for _ in range(250):
        node = _progress_node(rng)
        route = _route(rng)
        old = _Collector()
        _old_check_route_progress_bounds(old, node, route)
        new = _Collector()
        validator._check_route_progress_bounds(new, node, route)
        assert _issues(new) == _issues(old)


@pytest.mark.parametrize("seed", range(20))
def test_route_progress_matches_the_old_check_on_any_float(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over any floats, NaN and infinities included, which the real
    _node_number never returns."""
    rng = random.Random(500 + seed)
    numbers: dict[str, Optional[float]] = {}
    monkeypatch.setattr(validator, "_node_number", lambda node, key: numbers[key])
    for _ in range(250):
        numbers.clear()
        numbers["value"] = _float(rng)
        numbers["tolerance"] = _float(rng)
        node = _progress_node(rng)
        route = _route(rng)
        old = _Collector()
        _old_check_route_progress_bounds(old, node, route)
        new = _Collector()
        validator._check_route_progress_bounds(new, node, route)
        assert _issues(new) == _issues(old)


def test_route_progress_warning_defaults_its_tolerance() -> None:
    assert route_progress_warning(-1e-7, "start", "equal_to", None, 1e-6) is None
    assert route_progress_warning(-1e-7, "start", "equal_to", 1e-8, 1e-6) is not None
    assert route_progress_warning(-1.0, None, "less_than", None, 1e-6) is None
    assert route_progress_warning(-1.0, "start", "bogus", None, 1e-6) is None
