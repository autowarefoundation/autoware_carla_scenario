"""What ``scenario-odd route`` reads off its command line and prints, pinned.

Pins, against verbatim copies of them as they were in ``odd.cli`` (the
``_old_*`` oracles below), how ``_scenario_configs()`` splits its arguments
into scenario configs (a ``scenario=`` one or a plain name, globs expanded)
and Hydra overrides (any other with an ``=``), and how ``_format_buckets()``
writes the metres of a route expected in each bucket: what each returns
(type and value) or raises (type and args), for random arguments and buckets.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from autoware_carla_scenario.odd import cli as odd_cli


def _old_scenario_configs(items: Sequence[str]) -> tuple[list[str], list[str]]:
    """Scenario config names (globs expanded) and the Hydra overrides in *items*."""
    from autoware_carla_scenario.examples import run  # noqa: PLC0415
    from autoware_carla_scenario.registry import get_conf_dirs  # noqa: PLC0415
    from autoware_carla_scenario.scenario_glob import (  # noqa: PLC0415
        resolve_scenario_glob,
    )

    patterns: list[str] = []
    overrides: list[str] = []
    for item in items:
        if item.startswith("scenario="):
            patterns.append(item[len("scenario=") :])
        elif "=" in item:
            overrides.append(item)
        else:
            patterns.append(item)
    names: list[str] = []
    for pattern in patterns:
        if run._is_glob_pattern(pattern):
            # An invalid pattern raises ValueError, which the caller reports.
            matched = resolve_scenario_glob(pattern, get_conf_dirs())
            if not matched:
                raise ValueError(f"no scenario configs match {pattern!r}")
            names += matched
        else:
            names.append(pattern)
    # Since #2577, a config named twice (overlapping patterns) is kept once,
    # where it is first named.
    return list(dict.fromkeys(names)), overrides


def _old_format_buckets(buckets: dict[str, float]) -> str:
    return ", ".join(f"{label} {metres:.1f} m" for label, metres in buckets.items())


def _outcome(function: Callable[..., Any], *args: Any) -> Tuple[Any, ...]:
    try:
        value = function(*args)
    except Exception as exc:  # noqa: BLE001 - the oracle's exceptions are compared
        return ("raised", type(exc), exc.args)
    return ("returned", type(value), value)


_ITEMS = [
    "scenario=",
    "scenario=a",
    "scenario=a=b",
    "scenario==",
    "Scenario=a",
    " scenario=a",
    "scenarioa",
    "scenario",
    "=",
    "=x",
    "x=",
    "map=nishishinjuku_map",
    "+x=1",
    "~x",
    "a",
    "intersection_passing/straight",
    "",
    "ü=1",
    "a\nb",
]
#: Patterns that are globs: one that matches nothing, a malformed one.
_GLOBS = ["no_such_*", "scenario=no_such_?", "scenario=x/**a", "x/**a"]


def test_scenario_configs_split_as_before() -> None:
    rng = random.Random(2135)
    lists: List[List[Any]] = [[], list(_ITEMS), [*_ITEMS, *_GLOBS]]
    for _ in range(500):
        pool = _ITEMS + (_GLOBS if rng.random() < 0.2 else [])
        lists.append([rng.choice(pool) for _ in range(rng.randrange(0, 6))])
    # An item that is no str fails at the first str method, as before.
    lists += [[None], ["a", 1], [b"scenario=a"], ["x=1", None, "y"]]
    for items in lists:
        assert _outcome(odd_cli._scenario_configs, items) == _outcome(
            _old_scenario_configs, items
        ), items
    # A tuple and a generator are read as the list is.
    assert odd_cli._scenario_configs(tuple(_ITEMS)) == _old_scenario_configs(_ITEMS)
    from_iterator = odd_cli._scenario_configs(iter(_ITEMS))  # type: ignore[arg-type]
    assert from_iterator == _old_scenario_configs(_ITEMS)


def test_overlapping_patterns_name_each_config_once() -> None:
    # #2577: left_turn matched both patterns and was planned twice.
    names, overrides = odd_cli._scenario_configs(
        [
            "intersection_passing/left*",
            "map=nishishinjuku_map",
            "intersection_passing/*turn",
            "intersection_passing/left_turn",
        ]
    )
    assert overrides == ["map=nishishinjuku_map"]
    assert len(names) == len(set(names))
    assert names[0] == "intersection_passing/left_turn"
    assert "intersection_passing/right_turn" in names


def test_format_buckets_is_as_before() -> None:
    rng = random.Random(2135)
    values = [
        0.0,
        -0.0,
        0.04,
        0.05,
        0.15,
        0.25,
        1.0,
        12.345,
        1e20,
        -3.75,
        math.inf,
        -math.inf,
        math.nan,
        1,
        True,
    ]
    labels = ["[0, 10)", "", "dry", "ü", "a, b", "x m"]
    buckets: List[Dict[Any, Any]] = [{}]
    for _ in range(500):
        buckets.append(
            {
                rng.choice(labels): rng.choice(values + [rng.uniform(-1e3, 1e3)])
                for _ in range(rng.randrange(0, 5))
            }
        )
    # A value that is no number fails at its format spec, as before.
    buckets += [{"a": "1.0"}, {"a": None}, {1: 2.0}]
    for bucket in buckets:
        assert _outcome(odd_cli._format_buckets, bucket) == _outcome(
            _old_format_buckets, bucket
        ), bucket


def _old_split(items: Sequence[str]) -> tuple[list[str], list[str]]:
    """The split of _scenario_configs() as it was, before any glob is read."""
    patterns: list[str] = []
    overrides: list[str] = []
    for item in items:
        if item.startswith("scenario="):
            patterns.append(item[len("scenario=") :])
        elif "=" in item:
            overrides.append(item)
        else:
            patterns.append(item)
    return patterns, overrides


def test_the_checked_split_is_the_loop_it_replaces() -> None:
    from autoware_carla_scenario.odd import route_text

    rng = random.Random(2135)
    lists: List[List[Any]] = [[], [*_ITEMS, *_GLOBS], [None], ["a", 1, "b=1"]]
    for _ in range(500):
        lists.append([rng.choice(_ITEMS + _GLOBS) for _ in range(rng.randrange(0, 6))])
    for items in lists:
        assert _outcome(route_text._split_items, items) == _outcome(
            _old_split, items
        ), items


def test_odd_cli_re_exports_the_checked_format_buckets() -> None:
    # odd.cli's _format_buckets is odd.route_text's object (#2135), and
    # _scenario_configs() splits its arguments with odd.route_text's
    # _split_items(); no test patches either in odd.cli's namespace.
    from autoware_carla_scenario.odd import route_text

    assert odd_cli._format_buckets is route_text._format_buckets
    assert odd_cli._split_items is route_text._split_items


def _old_format_route(coverage: Any) -> list[str]:
    ids = [str(i) for i in coverage.lanelet_ids]
    if len(ids) > 12:
        ids = [*ids[:6], "...", *ids[-5:]]
    lines = [
        f"{coverage.name}: lanelets {' -> '.join(ids)} ({coverage.length_m:.1f} m)",
        f"  inside {coverage.inside_m:.1f} m, undecided {coverage.undecided_m:.1f} m, "
        f"outside {coverage.outside_m:.1f} m",
    ]
    for lanelet in coverage.outside():
        lines.append(
            f"  OUTSIDE lanelet {lanelet.lanelet_id} ({lanelet.length_m:.1f} m): "
            f"modules {', '.join(lanelet.failing_modules) or '-'}"
        )
    undecided = coverage.undecided()
    if undecided:
        lines.append(
            "  inside only by assumption: lanelets "
            + ", ".join(str(ll.lanelet_id) for ll in undecided)
        )
    return lines


def _old_own_package(spec: Any) -> Any:
    if not isinstance(spec, str) or ":" not in spec:
        return None
    top = spec.split(":", 1)[0].split(".", 1)[0]
    return top if top and top not in ("openodd", "autoware_carla_scenario") else None


class _Lanelet:
    def __init__(self, lanelet_id: Any, length_m: Any, failing_modules: Any) -> None:
        self.lanelet_id = lanelet_id
        self.length_m = length_m
        self.failing_modules = failing_modules


class _Coverage:
    """What _format_route() reads of openodd's route coverage."""

    def __init__(self, rng: random.Random) -> None:
        def metres() -> Any:
            return rng.choice(
                [0.0, -0.0, 0.05, 0.15, 12.345, 1e20, math.inf, math.nan, 3]
                + [rng.uniform(-1e4, 1e4)]
            )

        def an_id() -> Any:
            return rng.choice([rng.randrange(0, 100000), -1, 0, "x", "ü", 10**30])

        self.name = rng.choice(["route", "", "ü", "a: b", "r#1"])
        self.lanelet_ids = [
            an_id() for _ in range(rng.choice([0, 1, 5, 11, 12, 13, 30]))
        ]
        self.length_m = metres()
        self.inside_m = metres()
        self.undecided_m = metres()
        self.outside_m = metres()
        modules = ["planning", "", "ü", "a, b"]
        self._outside = [
            _Lanelet(
                an_id(),
                metres(),
                [rng.choice(modules) for _ in range(rng.randrange(0, 4))],
            )
            for _ in range(rng.randrange(0, 4))
        ]
        self._undecided = [
            _Lanelet(an_id(), metres(), []) for _ in range(rng.randrange(0, 15))
        ]

    def outside(self) -> list[_Lanelet]:
        return self._outside

    def undecided(self) -> list[_Lanelet]:
        return self._undecided


def test_format_route_is_as_before() -> None:
    # odd.cli's _format_route() reads openodd's coverage into the plain
    # values the checked odd.route_text.format_route_lines() formats (#2386).
    rng = random.Random(2386)
    for _ in range(1000):
        coverage = _Coverage(rng)
        assert _outcome(odd_cli._format_route, coverage) == _outcome(
            _old_format_route, coverage
        ), vars(coverage)


def test_own_package_is_as_before() -> None:
    rng = random.Random(2386)
    pieces = ["openodd", "autoware_carla_scenario", "pkg", "", ".", ":", "a.b", "ü"]
    specs: List[Any] = [None, 1, b"a:b", "", ":", "a:", ":a", "openodd.x:f"]
    for _ in range(1000):
        specs.append("".join(rng.choice(pieces) for _ in range(rng.randrange(0, 5))))
    for spec in specs:
        assert _outcome(odd_cli._own_package, spec) == _outcome(
            _old_own_package, spec
        ), spec
