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
    return names, overrides


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
