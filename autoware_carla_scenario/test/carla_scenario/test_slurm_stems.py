"""slurm.stems' config_key and sequential_stems against the functions they
were moved from.

``_original_config_key`` and ``_original_sequential_stems`` are
``slurm.manifest``'s functions as they were before the move (#846), copied
verbatim; the moved ones, and slurm.manifest's re-exports of them, must give
the same answers.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from autoware_carla_scenario.slurm import manifest, stems


def _original_config_key(prefix: str, key: Any) -> str:
    """The name of *key* under *prefix* in a config, as the worker reports a
    difference (``a.b``, ``a.list[2]``)."""
    if isinstance(key, int):
        return f"{prefix}[{key}]"
    return f"{prefix}.{key}" if prefix else str(key)


def _original_sequential_stems(scenario_classes: list[str]) -> list[str]:
    """The file stem each case of a batch is named by, in a local batch's way.

    A local batch names a run's files by its scenario class, and the next run
    of the same class ``{class}-1``, ``{class}-2`` and so on, in the order they
    ran (``scenario_runner._run_file_stem``).  A case keeps that name whichever
    task runs it, and when.
    """
    seen: dict[str, int] = {}
    stems: list[str] = []
    for name in scenario_classes:
        count = seen.get(name, 0)
        stems.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return stems


def test_manifest_re_exports_the_moved_functions() -> None:
    assert manifest.config_key is stems.config_key
    assert manifest.sequential_stems is stems.sequential_stems


@pytest.mark.parametrize("prefix", ["", "a", "a.b", "a.list[2]", "<root>"])
@pytest.mark.parametrize(
    "key",
    [0, 3, -1, True, False, "x", "", "a.b", 1.5, None, ("t", 1)],
)
def test_config_key_matches_the_original(prefix: str, key: Any) -> None:
    assert stems.config_key(prefix, key) == _original_config_key(prefix, key)


def test_sequential_stems_match_the_original() -> None:
    rng = random.Random(846)
    names = ["A", "B", "C", "A-1", "", "Scenario"]
    for _ in range(2000):
        classes = [rng.choice(names) for _ in range(rng.randrange(0, 12))]
        assert stems.sequential_stems(classes) == _original_sequential_stems(classes)
