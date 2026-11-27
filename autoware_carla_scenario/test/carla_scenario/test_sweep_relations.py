"""``sweep.relations``: relations between drawn parameters, solved with Z3.

Beside the distributions in ``sweep.constraints``, a sweep may list relations
between the parameters they draw ("the ego is faster than the lead"); each
case's values are drawn conditioned on them
(:mod:`autoware_carla_scenario.sweeper.relations`).
"""

from __future__ import annotations

import random
import statistics
from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.sweeper.distributions import UniformDistribution
from autoware_carla_scenario.sweeper.expand import check_constraint_keys, expand_sweep
from autoware_carla_scenario.sweeper.relations import (
    RelationSolver,
    RelationsUnsatisfiable,
    parse_relation,
    parse_relations,
)

_EGO = "ego.initial_speed_kmh"
_LEAD = "scenario.lead_speed_kmh"
_GAP = "scenario.gap_m"
_OTHER = "scenario.other"


def _uniform(low: float, high: float) -> list[dict[str, Any]]:
    return [{"type": "uniform", "low": low, "high": high}]


def _sweep(relations: list[str], samples: int = 50, **extra: Any) -> dict[str, Any]:
    return {
        "constraints": {
            _EGO: _uniform(20.0, 60.0),
            _LEAD: _uniform(10.0, 50.0),
            _GAP: [{"type": "normal", "mean": 30.0, "std": 10.0, "low": 5.0}],
            _OTHER: _uniform(0.0, 1.0),
            **extra,
        },
        "relations": relations,
        "samples": samples,
    }


def _values(case: list[str]) -> dict[str, float]:
    return {o.split("=", 1)[0]: float(o.split("=", 1)[1]) for o in case}


# -- Parsing ---------------------------------------------------------------------


def test_a_relation_names_its_dotted_keys() -> None:
    relation = parse_relation(
        "ego.initial_speed_kmh > max(scenario.lead_speed_kmh, 5) + abs(-1)", "r"
    )
    assert relation.keys == {_EGO, _LEAD}
    assert relation.holds({_EGO: 30.0, _LEAD: 20.0})
    assert not relation.holds({_EGO: 21.0, _LEAD: 20.0})


def test_chained_comparisons_and_boolean_operators() -> None:
    relation = parse_relation("0 < a.x < b.y and not (a.x == 1 or b.y >= 9)", "r")
    assert relation.holds({"a.x": 2.0, "b.y": 3.0})
    assert not relation.holds({"a.x": 1.0, "b.y": 3.0})
    assert not relation.holds({"a.x": 2.0, "b.y": 9.0})
    assert not relation.holds({"a.x": 4.0, "b.y": 3.0})


def test_a_division_by_zero_fails_the_relation() -> None:
    assert not parse_relation("1 / a.x > 0", "r").holds({"a.x": 0.0})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a.x +", "does not parse"),
        ("a.x + 1", "is a number, where a condition"),
        ("a.x > (b.y < 1)", "takes a number, but (b.y < 1) is a bool"),
        ("a.x > 'fast'", "takes a number, but 'fast' is a string"),
        ("a.x and 3", "takes a bool, but 3 is a number"),
        ("a.x in b.y", "'in' takes a list of literals"),
        ("a.x in [1, 'a']", "mixes numbers and strings"),
        ("a.x in [b.y]", "is not a literal"),
        ("a.x is 3", "no comparison"),
        ("a.x == None", "None is not a number, string or bool"),
        ("a.x > sqrt(b.y)", "not a function"),
        ("a.x > abs(b.y, 1)", "abs() takes 1 argument"),
        ("a.x > b.y ** 2", "is not something a relation takes"),
        ("__import__('os').system('true') > 0", "is not something a relation takes"),
        ("", "must be a relation"),
        (3, "must be a relation"),
    ],
)
def test_malformed_relations_are_refused_naming_the_item(
    text: Any, message: str
) -> None:
    with pytest.raises(ValueError, match="sweep.relations\\[0\\]") as exc:
        parse_relations([text])
    assert message in str(exc.value)


def test_relations_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        parse_relations("a.x > 1")
    assert parse_relations(None) == []
    assert len(parse_relations(OmegaConf.create(["a.x > 1"]))) == 1


# -- Checking the sweep before the map ----------------------------------------------


def test_unsatisfiable_relations_are_refused_naming_the_conflict() -> None:
    sweep = _sweep([f"{_EGO} < {_LEAD} - 45", f"{_OTHER} > 0.5"])
    with pytest.raises(RelationsUnsatisfiable) as exc:
        check_constraint_keys(sweep)
    assert "sweep.relations[0]" in str(exc.value)
    # The other relation is not part of the conflict.
    assert "sweep.relations[1]" not in str(exc.value)


def test_relations_need_something_to_expand() -> None:
    with pytest.raises(ValueError, match="no constraints, route or odd_sample"):
        check_constraint_keys({"relations": ["a.x > 1"]})


def test_a_drawn_parameter_compared_with_a_string_is_refused() -> None:
    with pytest.raises(ValueError, match="compares values of one type"):
        check_constraint_keys(_sweep([f"{_EGO} == 'fast'"]))
    with pytest.raises(ValueError, match="'in' over that list takes a string"):
        check_constraint_keys(_sweep([f"{_EGO} in ['a', 'b']"]))


# -- Drawing -------------------------------------------------------------------------


def test_every_draw_meets_the_relations() -> None:
    relations = [
        f"{_EGO} > {_LEAD} + 5",
        f"{_GAP} >= ({_EGO} - {_LEAD}) / 3.6 * 2",
    ]
    cases = expand_sweep(_sweep(relations, samples=200), None, scenario="s")
    assert len(cases) == 200
    for case in cases:
        v = _values(case)
        assert v[_EGO] > v[_LEAD] + 5
        assert v[_GAP] >= (v[_EGO] - v[_LEAD]) / 3.6 * 2 - 1e-6
        assert 20.0 <= v[_EGO] <= 60.0 and 10.0 <= v[_LEAD] <= 50.0
        assert v[_GAP] >= 5.0


def test_draws_are_the_distributions_conditioned_on_the_relations() -> None:
    # x, y uniform on [0, 1] conditioned on x > y: E[x] = 2/3, E[y] = 1/3.
    solver = RelationSolver(
        parse_relations(["a.x > a.y"]),
        {"a.x": UniformDistribution(0.0, 1.0), "a.y": UniformDistribution(0.0, 1.0)},
    )
    rng = random.Random(1)
    draws = [solver.draw({}, rng) for _ in range(4000)]
    assert statistics.fmean(d["a.x"] for d in draws) == pytest.approx(2 / 3, abs=0.02)
    assert statistics.fmean(d["a.y"] for d in draws) == pytest.approx(1 / 3, abs=0.02)


def test_ranges_are_narrowed_to_what_the_relations_leave() -> None:
    solver = RelationSolver(
        parse_relations(["a.x >= a.y + 8"]),
        {"a.x": UniformDistribution(0.0, 10.0), "a.y": UniformDistribution(0.0, 10.0)},
    )
    assert solver._case({}).ranges == {"a.x": (8.0, 10.0), "a.y": (0.0, 2.0)}


def test_a_range_no_relation_bounds_keeps_the_distribution_s() -> None:
    # Z3's box mode (several objectives in one Optimize) answered this with
    # a.x in [0, 0] on z3 4.15; each bound is optimised on its own instead.
    solver = RelationSolver(
        parse_relations(["scenario.weather != 'rain' or a.x < 20", "a.y <= a.x"]),
        {"a.x": UniformDistribution(0.0, 100.0), "a.y": UniformDistribution(0.0, 50.0)},
    )
    assert solver._case({"scenario.weather": "clear"}).ranges == {
        "a.x": (0.0, 100.0),
        "a.y": (0.0, 50.0),
    }
    assert solver._case({"scenario.weather": "rain"}).ranges == {
        "a.x": (0.0, 20.0),
        "a.y": (0.0, 20.0),
    }


def test_a_parameter_in_no_relation_draws_as_without_relations() -> None:
    with_relations = expand_sweep(_sweep([f"{_EGO} > {_LEAD}"]), None, scenario="s")
    without = expand_sweep(_sweep([]), None, scenario="s")
    for a, b in zip(with_relations, without):
        assert _values(a)[_OTHER] == _values(b)[_OTHER]
        assert _values(a)[_GAP] == _values(b)[_GAP]


def test_draws_are_reproducible() -> None:
    sweep = _sweep([f"{_EGO} > {_LEAD} + 5"])
    assert expand_sweep(sweep, None, scenario="s") == expand_sweep(
        sweep, None, scenario="s"
    )
    assert expand_sweep(sweep, None, scenario="s") != expand_sweep(
        sweep, None, scenario="s", seed=99
    )


def test_an_equality_is_met_by_drawing_one_after_another() -> None:
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 10.0), "a.y": _uniform(0.0, 10.0)},
        "relations": ["a.x == 2 * a.y", "a.y >= 1"],
        "samples": 5,
    }
    cases = expand_sweep(sweep, None, scenario="s")
    assert len(cases) == 5
    for case in cases:
        v = _values(case)
        assert v["a.x"] == pytest.approx(2 * v["a.y"], abs=1e-8)
        assert 1.0 <= v["a.y"] <= 5.0
    # Not all the same point.
    assert len({_values(c)["a.y"] for c in cases}) > 1


def test_a_relation_may_name_a_number_the_case_sets() -> None:
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 10.0)},
        "relations": ["a.x > ego.spawn_s"],
        "samples": 20,
    }
    for case in expand_sweep(sweep, None, ["ego.spawn_s=7.5"], scenario="s"):
        assert _values(case)["a.x"] > 7.5


def test_a_case_the_relations_cannot_hold_for_is_dropped() -> None:
    solver = RelationSolver(
        parse_relations(["a.x > ego.spawn_s"]), {"a.x": UniformDistribution(0.0, 10.0)}
    )
    solver.check({"ego.spawn_s": 5.0})
    with pytest.raises(RelationsUnsatisfiable, match="ego.spawn_s=12"):
        solver.check({"ego.spawn_s": 12.0})
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 10.0)},
        "relations": ["a.x > ego.spawn_s"],
    }
    with pytest.raises(ValueError, match="hold for none"):
        expand_sweep(sweep, None, ["ego.spawn_s=12"], scenario="s")


def test_a_constant_the_case_does_not_set_is_refused() -> None:
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 10.0)},
        "relations": ["a.x > ego.spawn_s"],
    }
    with pytest.raises(ValueError, match="neither drawn"):
        expand_sweep(sweep, None, scenario="s")


# -- Values of the case: typed constants, and relations as a filter -------------


def test_relations_compare_strings_bools_and_ids() -> None:
    relation = parse_relation(
        "scenario.weather in ['rain', 'fog'] and not scenario.night "
        "and scenario.npc_lanelet_id != ego.spawn_lanelet_id",
        "r",
    )
    base = {
        "scenario.weather": "rain",
        "scenario.night": False,
        "scenario.npc_lanelet_id": 12.0,
        "ego.spawn_lanelet_id": 34.0,
    }
    assert relation.holds(base)
    assert not relation.holds({**base, "scenario.weather": "clear"})
    assert not relation.holds({**base, "scenario.night": True})
    assert not relation.holds({**base, "scenario.npc_lanelet_id": 34.0})


def test_a_value_of_the_case_conditions_the_draws() -> None:
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 100.0)},
        "relations": ["scenario.weather != 'rain' or a.x < 20"],
        "samples": 30,
    }
    rainy = expand_sweep(sweep, None, ["scenario.weather=rain"], scenario="s")
    assert all(float(_values_typed(c)["a.x"]) < 20 for c in rainy)
    clear = expand_sweep(sweep, None, ["scenario.weather=clear"], scenario="s")
    assert any(float(_values_typed(c)["a.x"]) >= 20 for c in clear)


def test_a_case_value_of_the_wrong_type_is_refused() -> None:
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 10.0)},
        "relations": ["a.x < scenario.weather"],
    }
    with pytest.raises(ValueError, match="sets scenario.weather to a string"):
        expand_sweep(sweep, None, ["scenario.weather=rain"], scenario="s")


def test_the_config_s_own_values_are_the_default() -> None:
    from autoware_carla_scenario.sweeper.expand import config_lookup
    from autoware_carla_scenario.sweeper.relations import constant_value

    cfg = OmegaConf.create(
        {
            "scenario": {"lead_speed_kmh": 30, "weather": "rain", "night": True},
            "ego": {"limit": "${scenario.lead_speed_kmh}", "missing": "???"},
        }
    )
    lookup = config_lookup(cfg)
    assert [
        constant_value(lookup(key))
        for key in (
            "scenario.lead_speed_kmh",
            "ego.limit",
            "scenario.night",
            "scenario.weather",
            "ego.missing",
            "ego.unknown",
            "scenario",
        )
    ] == [30.0, 30.0, True, "rain", None, None, None]
    sweep: dict[str, Any] = {
        "constraints": {_EGO: _uniform(0.0, 100.0)},
        "relations": [
            f"{_EGO} > scenario.lead_speed_kmh",
            f"{_EGO} >= ego.limit",
            "scenario.night and scenario.weather == 'rain'",
        ],
        "samples": 20,
    }
    for case in expand_sweep(sweep, None, scenario="s", config_value=lookup):
        assert float(_values_typed(case)[_EGO]) > 30.0
    # An override of the run wins over the config's value.
    for case in expand_sweep(
        sweep, None, ["scenario.lead_speed_kmh=80"], scenario="s", config_value=lookup
    ):
        assert float(_values_typed(case)[_EGO]) > 80.0
    # A key the config does not hold is refused, naming the relation.
    sweep["relations"] = [f"{_EGO} > ego.missing"]
    with pytest.raises(ValueError, match="names ego.missing"):
        expand_sweep(sweep, None, scenario="s", config_value=lookup)


def test_override_values_are_read_as_hydra_reads_them() -> None:
    sweep = {
        "constraints": {"a.x": _uniform(0.0, 10.0)},
        "relations": ["scenario.mode == 'none' or a.x < 1"],
        "samples": 5,
    }
    # Hydra reads `none` as the string "none", not as null.
    cases = expand_sweep(sweep, None, ["scenario.mode=none"], scenario="s")
    assert len(cases) == 5


def test_relations_over_values_alone_filter_the_lanelet_cases(
    nishishinjuku: Any,
) -> None:
    ids = sorted(lanelet.id for lanelet in nishishinjuku.laneletLayer)[:3]
    sweep = {
        "constraints": {
            "ego.spawn_lanelet_id": [{"type": "in_set", "values": ids}],
        },
        "relations": [f"ego.spawn_lanelet_id != {ids[1]}"],
    }
    cases = expand_sweep(sweep, nishishinjuku, scenario="s")
    picked = sorted(int(_values_typed(c)["ego.spawn_lanelet_id"]) for c in cases)
    assert picked == [ids[0], ids[2]]
    sweep["relations"] = ["ego.spawn_lanelet_id < 0"]
    with pytest.raises(ValueError, match="hold for none"):
        expand_sweep(sweep, nishishinjuku, scenario="s")


@pytest.fixture(scope="module")
def nishishinjuku() -> Any:
    from pathlib import Path

    from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

    data = Path(__file__).resolve().parents[3] / "data"
    return load_lanelet2_map(
        data / "nishishinjuku.osm", data / "nishishinjuku_carla.xodr"
    )


def _values_typed(case: list[str]) -> dict[str, str]:
    return {o.split("=", 1)[0]: o.split("=", 1)[1] for o in case}
