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
        ("a.x + 1", "is a number where a condition"),
        ("a.x > (b.y < 1)", "is a condition where a number"),
        ("a.x > 'fast'", "is not a number"),
        ("a.x > True", "is not a number"),
        ("a.x in b.y", "no comparison"),
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


def test_relations_need_a_distribution() -> None:
    with pytest.raises(ValueError, match="draws none"):
        check_constraint_keys(
            {"constraints": {"ego.spawn_lanelet_id": []}, "relations": ["a.x > 1"]}
        )


def test_a_relation_must_name_a_drawn_parameter() -> None:
    with pytest.raises(ValueError, match="names no parameter drawn"):
        check_constraint_keys(_sweep(["ego.spawn_s > 3"]))


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
    assert solver._ranges_for({}, {}) == {"a.x": (8.0, 10.0), "a.y": (0.0, 2.0)}


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
