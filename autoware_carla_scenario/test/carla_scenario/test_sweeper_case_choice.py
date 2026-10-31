"""Which base case a drawn ODD case is put on (``sweeper.case_choice``, #799),
against what ``sweeper.expand`` computed before it was split out, and that
expand still computes the same through it."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pytest

from autoware_carla_scenario.sweeper import expand
from autoware_carla_scenario.sweeper.case_choice import (
    case_lanelet,
    choose_cases,
    fits,
)

# -- What sweeper.expand computed before #799, verbatim ----------------------


def _master_case_lanelet(case: Sequence[str]) -> Optional[int]:
    for item in case:
        key, _, value = item.lstrip("+").partition("=")
        if key.endswith("lanelet_id"):
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _master_fits(
    sample: Any, buckets: dict[str, str], attributes: Sequence[str]
) -> bool:
    return all(
        buckets.get(a) == sample.buckets[a] for a in attributes if a in sample.buckets
    )


def _master_choose_cases(
    samples: Sequence[Any],
    buckets: list[dict[str, str]],
    attributes: Sequence[str],
) -> tuple[list[int], int]:
    chosen: list[int] = []
    uses = [0] * len(buckets)
    full = 0
    for sample in samples:
        wanted = {a: sample.buckets[a] for a in attributes if a in sample.buckets}
        scores = [
            sum(1 for a, label in wanted.items() if b.get(a) == label) for b in buckets
        ]
        best = max(scores) if scores else 0
        if best == len(wanted):
            full += 1
        index = min(
            (i for i, score in enumerate(scores) if score == best),
            key=lambda i: (uses[i], i),
        )
        uses[index] += 1
        chosen.append(index)
    return chosen, full


@dataclass
class _Sample:
    """What expand reads of a drawn case: its buckets."""

    buckets: dict[str, str] = field(default_factory=dict)


# -- case_lanelet ------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        [],
        ["ego.spawn_lanelet_id=242", "ego.spawn_s=18.6"],
        ["ego.spawn_s=1", "++npc.lanelet_id=7", "ego.goal_lanelet_id=9"],
        ["+ego.spawn_lanelet_id=-3"],
        ["ego.spawn_lanelet_id=abc", "ego.goal_lanelet_id=9"],
        ["ego.spawn_lanelet_id=", "x.lanelet_id=1"],
        ["ego.spawn_lanelet_id= 12 "],
        ["ego.spawn_lanelet_id=1_000"],
        ["ego.spawn_lanelet_id=1.5"],
        ["lanelet_id=5"],
        ["ego.lanelet_ids=5"],
        ["ego.spawn_s=1", "scenario.name=a=b"],
        ["ego.spawn_lanelet_id=4=5"],
    ],
)
def test_case_lanelet_matches_master(case: list[str]) -> None:
    assert case_lanelet(case) == _master_case_lanelet(case)
    assert expand._case_lanelet(case) == _master_case_lanelet(case)


# -- fits and choose_cases ---------------------------------------------------

_ATTRIBUTES = ("junction", "lanes", "limit")
_LABELS = {"junction": ["yes", "no"], "lanes": ["1", "2", "3"], "limit": ["30", "50"]}


def _random_buckets(rng: random.Random) -> dict[str, str]:
    return {a: rng.choice(_LABELS[a]) for a in _ATTRIBUTES if rng.random() < 0.8}


@pytest.mark.parametrize("seed", range(40))
def test_fits_matches_master(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        drawn = _random_buckets(rng)
        buckets = _random_buckets(rng)
        attributes = rng.sample(
            [*_ATTRIBUTES, "other"], rng.randint(0, len(_ATTRIBUTES) + 1)
        )
        assert fits(drawn, buckets, attributes) == _master_fits(
            _Sample(drawn), buckets, attributes
        )


@pytest.mark.parametrize("seed", range(40))
def test_choose_cases_matches_master(seed: int) -> None:
    rng = random.Random(seed)
    buckets = [_random_buckets(rng) for _ in range(rng.randint(1, 8))]
    samples = [_Sample(_random_buckets(rng)) for _ in range(rng.randint(0, 30))]
    attributes = rng.sample(list(_ATTRIBUTES), rng.randint(1, len(_ATTRIBUTES)))
    expected = _master_choose_cases(samples, buckets, attributes)
    assert choose_cases([s.buckets for s in samples], buckets, attributes) == expected
    assert (
        expand._choose_cases([s.buckets for s in samples], buckets, attributes)
        == expected
    )


def test_choose_cases_breaks_ties_by_uses_then_index() -> None:
    # Every base case is in the same buckets: the cases go round them in
    # order, the least used first and the first of equals.
    buckets = [{"lanes": "2"}] * 3
    samples = [_Sample({"lanes": "2"}) for _ in range(7)]
    expected = _master_choose_cases(samples, buckets, ["lanes"])
    assert expected == ([0, 1, 2, 0, 1, 2, 0], 7)
    assert choose_cases([s.buckets for s in samples], buckets, ["lanes"]) == expected


def test_choose_cases_prefers_the_best_score_before_the_least_used() -> None:
    buckets = [
        {"junction": "no", "lanes": "1"},
        {"junction": "yes", "lanes": "2"},
        {"junction": "yes", "lanes": "1"},
        {"junction": "yes", "lanes": "2"},
    ]
    drawn = [
        {"junction": "yes", "lanes": "2"},
        {"junction": "yes", "lanes": "2"},
        {"junction": "yes", "lanes": "2"},
        {"junction": "no", "lanes": "3"},
        {"junction": "no", "lanes": "3"},
        {},
        {"limit": "30"},
    ]
    attributes = ["junction", "lanes"]
    expected = _master_choose_cases([_Sample(d) for d in drawn], buckets, attributes)
    assert choose_cases(drawn, buckets, attributes) == expected
    assert expected == ([1, 3, 1, 0, 0, 2, 2], 5)


def test_choose_cases_exhaustively_on_small_inputs() -> None:
    # Every combination of a few base cases and drawn cases over one
    # attribute of two labels, present or not.
    options: list[dict[str, str]] = [{}, {"a": "x"}, {"a": "y"}]
    for n_buckets in range(1, 4):
        for buckets_t in itertools.product(options, repeat=n_buckets):
            buckets = list(buckets_t)
            for n_drawn in range(0, 4):
                for drawn_t in itertools.product(options, repeat=n_drawn):
                    drawn = list(drawn_t)
                    expected = _master_choose_cases(
                        [_Sample(d) for d in drawn], buckets, ["a"]
                    )
                    assert choose_cases(drawn, buckets, ["a"]) == expected


def test_choose_cases_raises_without_base_cases_as_master_did() -> None:
    with pytest.raises(ValueError):
        _master_choose_cases([_Sample({"a": "x"})], [], ["a"])
    with pytest.raises(ValueError):
        choose_cases([{"a": "x"}], [], ["a"])
    assert choose_cases([], [], ["a"]) == _master_choose_cases([], [], ["a"])


# -- expand's caller -----------------------------------------------------------


class _Sampler:
    def __init__(self, drawn: list[_Sample]) -> None:
        self._drawn = drawn

    def sample(self, count: int) -> list[_Sample]:
        return self._drawn[:count]


@pytest.mark.parametrize("seed", range(10))
def test_on_some_case_keeps_what_master_kept(seed: int) -> None:
    rng = random.Random(seed)
    buckets = [_random_buckets(rng) for _ in range(rng.randint(1, 5))]
    drawn = [_Sample(_random_buckets(rng)) for _ in range(60)]
    attributes = list(_ATTRIBUTES)
    wanted = rng.randint(1, 3)
    pool = drawn[: wanted * expand._DRAWS_PER_CASE]
    kept = [s for s in pool if any(_master_fits(s, b, attributes) for b in buckets)]
    passed = [s for s in pool if s not in kept]
    expected = (
        kept[:wanted] if len(kept) >= wanted else kept + passed[: wanted - len(kept)]
    )
    got = expand._on_some_case(_Sampler(drawn), wanted, buckets, attributes)
    assert [id(s) for s in got] == [id(s) for s in expected]
