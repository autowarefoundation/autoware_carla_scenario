"""A sweep's cases are distinct (#1950), and a sweep of the wrong shape is
refused with a ``ValueError`` naming what is wrong (#1951).

``stop_line_offset``'s walk-back may move the swept key itself off the pick,
so different picks can end as the same case: it runs once, the first pick's,
and the log names the picks merged.  A constraint, binding,
``sweep.constraints``, ``sweep.bindings`` or ``sweep.exclude`` of the wrong
shape, an unknown key or a bad value used to escape as a ``TypeError`` or
``AttributeError`` traceback; the sweeper, ``scenario-expand`` and the batch
print a ``ValueError`` as one line, ``Error: <scenario>: ...``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.sweeper import expand as expand_module
from autoware_carla_scenario.sweeper.bindings import (
    RouteThroughBinding,
    StopLineApproachBinding,
    StopLineOffsetBinding,
    parse_binding,
)
from autoware_carla_scenario.sweeper.constraints import parse_constraint
from autoware_carla_scenario.sweeper.expand import expand_sweep
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

_DATA = Path(__file__).resolve().parents[3] / "data"

_PICK = "ego.spawn_lanelet_id"


@pytest.fixture(scope="module")
def nishishinjuku() -> Any:
    return load_lanelet2_map(
        _DATA / "nishishinjuku.osm", _DATA / "nishishinjuku_carla.xodr"
    )


# -- #1950: identical cases are merged ----------------------------------------


def test_picks_that_walk_back_to_one_case_give_it_once(
    nishishinjuku: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # On master: 281 cases, many of them the same override list 2-3 times
    # (picks 324, 519 and 577 all walk back to lanelet 54).
    sweep = {
        "constraints": {_PICK: [{"type": "has_traffic_light_stop_line"}]},
        "bindings": {"ego.spawn_s": {"type": "stop_line_offset", "offset": 15}},
    }
    with caplog.at_level(logging.INFO, logger=expand_module.__name__):
        cases = expand_sweep(sweep, nishishinjuku)
    distinct = {tuple(case) for case in cases}
    assert len(cases) == len(distinct)
    assert 0 < len(cases) < 281
    merged = [
        record.getMessage()
        for record in caplog.records
        if record.name == expand_module.__name__
        and "give the same case" in record.getMessage()
    ]
    assert merged
    assert any(
        "324" in message and "519" in message and "577" in message for message in merged
    ), merged


class TestMergedCases:
    """The merge, on stand-ins for the constraint machinery."""

    @pytest.fixture
    def _picks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(expand_module, "parse_constraint", lambda c: c)
        monkeypatch.setattr(expand_module, "create_routing_graph", lambda m: "graph")
        monkeypatch.setattr(
            expand_module, "find_matching_lanelets", lambda c, m, g: [5, 6, 7, 8]
        )

        def _resolve(lid: int, m: Any, g: Any) -> SimpleNamespace:
            # 5, 6 and 8 walk back to 50, 7 fits on itself.
            if lid == 7:
                return SimpleNamespace(value=3.0, lanelet_id_override=None)
            return SimpleNamespace(value=1.5, lanelet_id_override=50)

        monkeypatch.setattr(
            expand_module,
            "parse_binding",
            lambda key, cfg: SimpleNamespace(target_key=key, resolve=_resolve),
        )

    @pytest.mark.usefixtures("_picks")
    def test_each_case_runs_once_as_the_first_pick_gives_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=expand_module.__name__):
            cases = expand_sweep(
                {
                    "constraints": {_PICK: [{}]},
                    "bindings": {"ego.spawn_s": {}},
                },
                lanelet_map=object(),
                arguments=["map=x"],
            )
        assert cases == [
            ["ego.spawn_lanelet_id=50", "ego.spawn_s=1.5", "map=x"],
            ["ego.spawn_lanelet_id=7", "ego.spawn_s=3.0", "map=x"],
        ]
        assert "Lanelets 5, 6, 8 give the same case" in caplog.text
        assert "as lanelet 5's" in caplog.text

    @pytest.mark.usefixtures("_picks")
    def test_a_case_index_counts_distinct_cases(self) -> None:
        sweep = {
            "constraints": {_PICK: [{}]},
            "bindings": {"ego.spawn_s": {}},
            "case_index": 1,
        }
        assert expand_sweep(sweep, lanelet_map=object()) == [
            ["ego.spawn_lanelet_id=7", "ego.spawn_s=3.0"]
        ]
        with pytest.raises(ValueError, match="expands to 2 case"):
            expand_sweep({**sweep, "case_index": 2}, lanelet_map=object())


# -- #1951: constraints of the wrong shape ------------------------------------


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            {"type": "has_adjacent", "side": "left"},
            r"has_adjacent constraint: unknown key\(s\) \['side'\]; it takes "
            r"\['value', 'include_forks'\]",
        ),
        ({"type": "is_junction", "value": 1}, r"is_junction constraint: unknown"),
        ("is_junction", r"A constraint must be a mapping with a 'type'.*'is_junction'"),
        (["is_junction"], r"A constraint must be a mapping"),
        ({"type": ["is_junction"]}, r"'type' must be a string"),
        ({"type": "nope"}, r"Unknown constraint type: 'nope'"),
        (
            {"type": "and", "constraints": {"type": "is_junction"}},
            r"'and' constraint requires a 'constraints' list",
        ),
        (
            {"type": "previous_of", "constraints": "is_junction"},
            r"'previous_of' constraint requires a 'constraints' list",
        ),
        (
            {"type": "not", "constraint": ["is_junction"]},
            r"A constraint must be a mapping",
        ),
        (
            {"type": "or", "constraints": [{"type": "has_adjacent", "side": "x"}]},
            r"has_adjacent constraint: unknown key",
        ),
    ],
)
def test_a_constraint_of_the_wrong_shape_is_a_value_error(
    cfg: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_constraint(cfg)


def test_valid_constraints_still_parse() -> None:
    parsed = parse_constraint(
        {
            "type": "and",
            "constraints": [
                {"type": "turn_direction", "value": "right"},
                {"type": "has_adjacent", "value": "left", "include_forks": True},
                {"type": "in_set", "values": [1, 2]},
                {"type": "not", "constraint": {"type": "is_junction"}},
            ],
        }
    )
    assert type(parsed).__name__ == "AndConstraint"


@pytest.mark.parametrize("composite", ["and", "or", "previous_of", "following_of"])
def test_a_composite_of_hydra_nodes_parses(composite: str) -> None:
    # A ListConfig of children is a list (background traffic regions hand the
    # parser Hydra nodes), nested inside a not too.
    children = [{"type": "has_stop_line"}, {"type": "is_junction"}]
    node: Any = OmegaConf.create({"type": composite, "constraints": children})
    assert parse_constraint(node) == parse_constraint(
        {"type": composite, "constraints": children}
    )
    wrapped: Any = OmegaConf.create(
        {"type": "not", "constraint": {"type": composite, "constraints": children}}
    )
    assert type(parse_constraint(wrapped)).__name__ == "NotConstraint"


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            {"type": "and", "constraints": [{"type": "is_junction"}], "value": 3},
            r"and constraint: unknown key\(s\) \['value'\]; it takes \['constraints'\]",
        ),
        (
            {"type": "not", "constraints": [{"type": "is_junction"}]},
            r"not constraint: unknown key\(s\) \['constraints'\]; it takes \['constraint'\]",
        ),
        (
            {"type": "following_of", "constraint": {"type": "is_junction"}},
            r"following_of constraint: unknown key\(s\) \['constraint'\]",
        ),
    ],
)
def test_a_composite_with_an_unknown_key_is_refused(cfg: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_constraint(cfg)


# -- #1951: bindings of the wrong shape ---------------------------------------


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            {"type": "stop_line_offset", "ofset": 3},
            r"stop_line_offset binding ego.spawn_s: unknown key\(s\) \['ofset'\]; "
            r"it takes \['offset'\]",
        ),
        ({"type": "matched", "depth": 1}, r"matched binding ego.spawn_s: unknown"),
        (15, r"Binding ego.spawn_s must be a mapping with a 'type'.*15"),
        ({"type": ["matched"]}, r"Unknown binding type: \['matched'\]"),
        ({"type": "route_through", "depth": "2"}, r"depth must be a whole number"),
        ({"type": "route_through", "depth": True}, r"depth must be a whole number"),
        ({"type": "route_through", "depth": 1.5}, r"depth must be a whole number"),
        ({"type": "route_through", "depth": 0}, r"depth must be >= 1"),
    ],
)
def test_a_binding_of_the_wrong_shape_is_a_value_error(cfg: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_binding("ego.spawn_s", cfg)


@pytest.mark.parametrize("kind", ["stop_line_offset", "stop_line_approach"])
@pytest.mark.parametrize("offset", ["15", math.nan, math.inf, True, False, -5, -0.1])
def test_a_stop_line_offset_that_is_no_finite_distance_is_refused(
    kind: str, offset: Any
) -> None:
    with pytest.raises(ValueError, match=rf"{kind} offset: a finite number of metres"):
        parse_binding("ego.spawn_s", {"type": kind, "offset": offset})


@pytest.mark.parametrize("offset", [0, 15, 25.0, np.float64(2.5), np.int64(3)])
def test_a_stop_line_offset_of_metres_is_taken(offset: Any) -> None:
    assert StopLineOffsetBinding("ego.spawn_s", offset).offset == offset
    assert StopLineApproachBinding("x.lanelet_id", offset).offset == offset
    assert RouteThroughBinding("x", depth=2).depth == 2


# -- #1951: sweep sections of the wrong shape ---------------------------------


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        (
            {"constraints": {_PICK: {"type": "is_junction"}}},
            r"sweep.constraints.ego.spawn_lanelet_id must be a list of "
            r"constraints \(- type: ...\), got a mapping",
        ),
        (
            {"constraints": [{"type": "is_junction"}]},
            r"sweep.constraints must map the one parameter it sweeps .* got a list",
        ),
        (
            {"constraints": {_PICK: [{}]}, "exclude": 5},
            r"sweep.exclude must map each parameter .* got 5",
        ),
        (
            {"constraints": {_PICK: [{}]}, "exclude": {_PICK: 54}},
            r"sweep.exclude.ego.spawn_lanelet_id must be a list of values, got 54",
        ),
        (
            {"constraints": {_PICK: [{}]}, "exclude": {_PICK: "54"}},
            r"sweep.exclude.ego.spawn_lanelet_id must be a list of values, got '54'",
        ),
        (
            {"constraints": {_PICK: [{}]}, "bindings": [{"type": "matched"}]},
            r"sweep.bindings must map each parameter it sets to a binding",
        ),
        (
            {"constraints": {_PICK: [{}]}, "bindings": {"ego.spawn_s": 15}},
            r"sweep.bindings.ego.spawn_s must be a binding \(\{type: ...\}\), got 15",
        ),
    ],
)
def test_a_sweep_section_of_the_wrong_shape_is_refused_before_the_map(
    sweep: dict[str, Any], message: str
) -> None:
    # No map: the shape is checked before anything is searched.
    with pytest.raises(ValueError, match=message):
        expand_sweep(sweep, lanelet_map=None)
    with pytest.raises(ValueError, match=message):
        expand_module.check_constraint_keys(sweep)


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        (
            {"constraints": {_PICK: [{"type": "has_adjacent", "side": "left"}]}},
            r"has_adjacent constraint: unknown key",
        ),
        (
            {
                "constraints": {_PICK: [{"type": "has_traffic_light_stop_line"}]},
                "bindings": {"ego.spawn_s": {"type": "stop_line_offset", "ofset": 3}},
            },
            r"stop_line_offset binding ego.spawn_s: unknown key",
        ),
        (
            {
                "constraints": {_PICK: [{"type": "has_traffic_light_stop_line"}]},
                "bindings": {
                    "ego.spawn_s": {"type": "stop_line_offset", "offset": "15"}
                },
            },
            r"stop_line_offset offset: a finite number of metres >= 0, got '15'",
        ),
        (
            {
                "constraints": {_PICK: [{"type": "has_traffic_light_stop_line"}]},
                "bindings": {"x.ids": {"type": "route_through", "depth": "2"}},
            },
            r"route_through depth must be a whole number >= 1, got '2'",
        ),
    ],
)
def test_a_constraint_or_binding_that_does_not_parse_fails_the_sweep(
    nishishinjuku: Any, sweep: dict[str, Any], message: str
) -> None:
    # Not a warning per pick ending in the empty-sweep error: the sweep fails.
    with pytest.raises(ValueError, match=message):
        expand_sweep(sweep, nishishinjuku)


# -- The one-line error ---------------------------------------------------------


@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "++sweep.bindings={ego.spawn_s:{type:stop_line_offset,offset:-5}}",
            "stop_line_offset offset: a finite number of metres >= 0, got -5",
        ),
        (
            "++sweep.bindings={ego.spawn_s:{type:stop_line_offset,ofset:3}}",
            "stop_line_offset binding ego.spawn_s: unknown key(s) ['ofset']",
        ),
        (
            "++sweep.constraints={ego.spawn_lanelet_id:[{type:has_adjacent,side:left}]}",
            "has_adjacent constraint: unknown key(s) ['side']",
        ),
        ("++sweep.exclude=5", "sweep.exclude must map each parameter"),
    ],
)
def test_scenario_expand_reports_a_bad_shape_in_one_line(
    capsys: pytest.CaptureFixture[str], override: str, message: str
) -> None:
    from autoware_carla_scenario.examples.expand import main

    code = main(["scenario=intersection_passing/right_turn", override])
    captured = capsys.readouterr()
    assert code == 1
    assert f"Error: intersection_passing/right_turn: {message}" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "sweep",
    [
        {"constraints": [{"type": "is_junction"}]},
        {"constraints": {_PICK: {"type": "is_junction"}}},
        {"constraints": {_PICK: [{}]}, "exclude": 5},
    ],
)
def test_the_sweeper_reports_a_bad_shape_in_one_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: dict[str, Any],
) -> None:
    from unittest.mock import MagicMock

    from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

    monkeypatch.delenv("HYDRA_FULL_ERROR", raising=False)
    sweeper = mod.LaneletConstraintSweeper()
    sweeper.config = OmegaConf.create(
        {
            "map": {"source": "https://example.com/never-cloned.git"},
            "sweep": sweep,
            "hydra": {"runtime": {"choices": {"scenario": "cut_in/left"}}},
        }
    )
    sweeper.launcher = MagicMock()
    with pytest.raises(SystemExit) as caught:
        sweeper.sweep([])
    assert caught.value.code == 1
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1, lines
    assert lines[0].startswith("Error: cut_in/left: sweep."), lines
    sweeper.launcher.launch.assert_not_called()
