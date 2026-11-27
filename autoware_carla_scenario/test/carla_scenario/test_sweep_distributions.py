"""A sweep draws scalar parameters from distributions in ``sweep.constraints``.

Beside the key it enumerates lanelets for, ``sweep.constraints`` may map a scalar
parameter to the one distribution it is drawn from (``type: uniform`` or
``type: normal``), drawn ``sweep.samples`` times for each case, from the run's
seed (:mod:`autoware_carla_scenario.sweeper.distributions`).
"""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.sweeper.distributions import (
    NormalDistribution,
    UniformDistribution,
    draw,
    draw_values,
    is_distribution,
    key_seed,
    normal_cdf,
    normal_ppf,
    parse_distribution,
)
from autoware_carla_scenario.sweeper.expand import (
    check_constraint_keys,
    expand_sweep,
    split_constraints,
)
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

_DATA = Path(__file__).resolve().parents[3] / "data"
_PICK = "ego.spawn_lanelet_id"
_DIST = "scenario.stop_line_distance_m"


@pytest.fixture(scope="module")
def nishishinjuku() -> Any:
    return load_lanelet2_map(
        _DATA / "nishishinjuku.osm", _DATA / "nishishinjuku_carla.xodr"
    )


def _value(case: list[str], key: str) -> float:
    return float(next(o for o in case if o.startswith(f"{key}=")).split("=", 1)[1])


# -- The distributions ---------------------------------------------------------


@pytest.mark.parametrize(
    "p", [1e-12, 1e-6, 0.01, 0.02425, 0.1, 0.5, 0.9, 0.99, 1 - 1e-9]
)
def test_the_normal_quantile_inverts_the_cdf(p: float) -> None:
    assert normal_cdf(normal_ppf(p)) == pytest.approx(p, rel=1e-9, abs=1e-15)


def test_a_quantile_outside_the_open_interval_is_refused() -> None:
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="0 < p < 1"):
            normal_ppf(p)


def test_uniform_draws_stay_in_range_and_spread_over_it() -> None:
    rng = random.Random(0)
    dist = UniformDistribution(low=0.5, high=3.0)
    values = [draw(dist, rng) for _ in range(5000)]
    assert min(values) >= 0.5 and max(values) <= 3.0
    assert statistics.fmean(values) == pytest.approx(1.75, abs=0.05)
    # Every fifth of the range is reached.
    assert {int((v - 0.5) / 0.5) for v in values} == {0, 1, 2, 3, 4}


def test_normal_draws_have_its_mean_and_std() -> None:
    rng = random.Random(1)
    values = [draw(NormalDistribution(mean=40.0, std=5.0), rng) for _ in range(20000)]
    assert statistics.fmean(values) == pytest.approx(40.0, abs=0.15)
    assert statistics.stdev(values) == pytest.approx(5.0, rel=0.03)


def test_a_truncated_normal_stays_inside_its_bounds_and_keeps_its_shape() -> None:
    rng = random.Random(2)
    dist = NormalDistribution(mean=40.0, std=5.0, low=38.0, high=60.0)
    values = [draw(dist, rng) for _ in range(20000)]
    assert min(values) >= 38.0 and max(values) <= 60.0
    # Truncated below the mean: the mean moves up, by phi(a) / (1 - Phi(a)).
    a = (38.0 - 40.0) / 5.0
    phi = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    expected = 40.0 + 5.0 * phi / (1 - normal_cdf(a))
    assert statistics.fmean(values) == pytest.approx(expected, abs=0.1)


def test_a_one_sided_truncation_is_one_bound() -> None:
    rng = random.Random(3)
    dist = NormalDistribution(mean=0.0, std=1.0, low=0.0)
    values = [draw(dist, rng) for _ in range(2000)]
    assert min(values) >= 0.0
    assert max(values) > 2.0


def test_draws_follow_the_seed_and_each_key_has_a_stream_of_its_own() -> None:
    dists = {"a": UniformDistribution(0.0, 1.0), "b": NormalDistribution(0.0, 1.0)}
    first = draw_values(dists, 10, seed=7)
    assert first == draw_values(dists, 10, seed=7)
    assert first != draw_values(dists, 10, seed=8)
    # Adding a key does not move the values the others draw.
    more = draw_values({**dists, "c": UniformDistribution(5.0, 6.0)}, 10, seed=7)
    assert [d["a"] for d in more] == [d["a"] for d in first]
    assert [d["b"] for d in more] == [d["b"] for d in first]
    assert key_seed(7, "a") != key_seed(7, "b")
    assert key_seed(7, "a") == key_seed(7, "a") >= 0


# -- Parsing -------------------------------------------------------------------


def test_parse_reads_each_distribution() -> None:
    assert parse_distribution(_DIST, [{"type": "uniform", "low": 0.5, "high": 3}]) == (
        UniformDistribution(0.5, 3.0)
    )
    assert parse_distribution(
        "ego.initial_speed_kmh",
        [{"type": "normal", "mean": 40, "std": 5, "low": 30, "high": None}],
    ) == NormalDistribution(40.0, 5.0, 30.0, math.inf)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([{"type": "uniform", "low": 3.0, "high": 0.5}], "below high"),
        ([{"type": "uniform", "low": 1.0, "high": 1.0}], "below high"),
        ([{"type": "uniform", "low": 0.5}], "high is required"),
        ([{"type": "uniform", "low": "0.5", "high": 3}], "must be a number"),
        ([{"type": "uniform", "low": True, "high": 3}], "must be a number"),
        ([{"type": "uniform", "low": 0, "high": math.inf}], "must be finite"),
        ([{"type": "uniform", "low": 0, "high": 1, "mean": 0}], "not 'mean'"),
        ([{"type": "normal", "mean": 0, "std": 0}], "std must be above 0"),
        ([{"type": "normal", "mean": 0, "std": -1}], "std must be above 0"),
        ([{"type": "normal", "mean": 0, "std": 1, "low": 2, "high": 1}], "below high"),
        ([{"type": "normal", "mean": 0, "std": 1, "low": 50}], "too far into the tail"),
        (
            [{"type": "uniform", "low": 0, "high": 1}, {"type": "is_junction"}],
            "nothing else",
        ),
    ],
)
def test_parse_refuses_a_bad_distribution_naming_its_key(
    items: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as error:
        parse_distribution(_DIST, items)
    assert f"sweep.constraints.{_DIST}" in str(error.value)


def test_only_a_list_with_a_distribution_type_is_one() -> None:
    assert is_distribution([{"type": "normal", "mean": 0, "std": 1}])
    assert not is_distribution([{"type": "has_traffic_light_stop_line"}])
    assert not is_distribution([])


# -- The sweep -----------------------------------------------------------------


def _sweep(**extra: Any) -> dict[str, Any]:
    return {
        "constraints": {
            _PICK: [{"type": "has_traffic_light_stop_line"}],
            _DIST: [{"type": "uniform", "low": 0.5, "high": 3.0}],
        },
        **extra,
    }


def test_a_distribution_key_is_no_second_swept_key() -> None:
    check_constraint_keys(_sweep())
    lanelet, dists = split_constraints(_sweep())
    assert list(lanelet) == [_PICK]
    assert dists == {_DIST: UniformDistribution(0.5, 3.0)}
    # Two lanelet keys are still refused.
    with pytest.raises(ValueError, match="sweeps one parameter"):
        check_constraint_keys(
            {
                "constraints": {
                    _PICK: [{"type": "is_junction"}],
                    "ego.goal_lanelet_id": [{"type": "is_junction"}],
                    _DIST: [{"type": "uniform", "low": 0, "high": 1}],
                }
            }
        )


def test_the_key_check_refuses_a_bad_distribution_or_sample_count() -> None:
    bad = _sweep()
    bad["constraints"][_DIST] = [{"type": "uniform", "low": 3, "high": 1}]
    with pytest.raises(ValueError, match="below high"):
        check_constraint_keys(bad)
    for samples in (0, -1, 1.5, "x", True):
        with pytest.raises(ValueError, match="sweep.samples"):
            check_constraint_keys(_sweep(samples=samples))


def test_each_lanelet_case_runs_samples_times_with_values_of_its_own(
    nishishinjuku: Any,
) -> None:
    plain = expand_sweep(
        {"constraints": {_PICK: [{"type": "has_traffic_light_stop_line"}]}},
        nishishinjuku,
    )
    cases = expand_sweep(_sweep(samples=3), nishishinjuku, ["foo=1"], seed=5)
    assert len(cases) == 3 * len(plain)
    for number, base in enumerate(plain):
        group = cases[3 * number : 3 * number + 3]
        for case in group:
            assert case[: len(base)] == base
            assert case[-1] == "foo=1"  # the run's own arguments last, as before
            assert 0.5 <= _value(case, _DIST) <= 3.0
        assert len({_value(case, _DIST) for case in group}) == 3
    # The same seed draws the same values; another draws others.
    assert cases == expand_sweep(_sweep(samples=3), nishishinjuku, ["foo=1"], seed=5)
    assert cases != expand_sweep(_sweep(samples=3), nishishinjuku, ["foo=1"], seed=6)


def test_without_a_seed_the_default_seed_draws(nishishinjuku: Any) -> None:
    from autoware_carla_scenario.seed import DEFAULT_SEED

    assert expand_sweep(_sweep(), nishishinjuku) == expand_sweep(
        _sweep(), nishishinjuku, seed=DEFAULT_SEED
    )


def test_a_sweep_of_distributions_alone_needs_no_map() -> None:
    sweep = {
        "constraints": {
            "ego.initial_speed_kmh": [{"type": "normal", "mean": 40, "std": 5}],
            _DIST: [{"type": "uniform", "low": 0.5, "high": 3.0}],
        },
        "samples": 4,
    }
    cases = expand_sweep(sweep, None, seed=1)
    assert len(cases) == 4
    for case in cases:
        assert [o.split("=")[0] for o in case] == ["ego.initial_speed_kmh", _DIST]


def test_a_drawn_key_may_not_be_set_by_an_override_or_a_binding(
    nishishinjuku: Any,
) -> None:
    with pytest.raises(ValueError, match="an override may not set it"):
        expand_sweep(_sweep(), nishishinjuku, [f"{_DIST}=2.0"])
    # A base case -- a binding's, a route match's or an ODD draw's -- that sets it.
    from autoware_carla_scenario.sweeper.expand import _with_distribution_draws

    with pytest.raises(ValueError, match="bindings, route or ODD sample set it"):
        _with_distribution_draws(
            [[f"{_PICK}=1", f"{_DIST}=2.0"]],
            {_DIST: UniformDistribution(0.5, 3.0)},
            1,
            0,
            [],
        )


def test_case_index_picks_among_the_drawn_cases(nishishinjuku: Any) -> None:
    every = expand_sweep(_sweep(samples=2), nishishinjuku, seed=3)
    one = expand_sweep(_sweep(samples=2, case_index=3), nishishinjuku, seed=3)
    assert one == [every[3]]
