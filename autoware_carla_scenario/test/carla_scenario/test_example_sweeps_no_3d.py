"""No shipped example sweep expands onto a lanelet CARLA has no 3D model for (#1069).

The map config lists those lanelets (``map.no_3d_model_lanelet_ids``); a case
that spawns on one, picks one, or routes through one cannot be driven in
CARLA. Every example sweep is expanded on the Nishishinjuku fixture map, as
``scenario-expand`` expands it, and every lanelet ID a case sets is checked.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

import autoware_carla_scenario.examples as examples_package

_SCENARIO_DIR = Path(examples_package.__file__).parent / "conf" / "scenario"


def _sweep_scenarios() -> list[str]:
    """The example scenarios with ``sweep.constraints``: the ones that expand
    per lanelet."""
    names = []
    for path in sorted(_SCENARIO_DIR.rglob("*.yaml")):
        sweep = (yaml.safe_load(path.read_text()) or {}).get("sweep") or {}
        if sweep.get("constraints"):
            names.append(path.relative_to(_SCENARIO_DIR).with_suffix("").as_posix())
    return names


def _lanelet_ids(case: list[str]) -> set[int]:
    """Every lanelet ID a case's overrides set: the spawn lanelet, the route
    (whose first entry is the picked lanelet), a goal, ..."""
    ids: set[int] = set()
    for override in case:
        key, _, value = override.partition("=")
        if "lanelet_id" not in key:
            continue
        parsed: Any = yaml.safe_load(value)
        ids.update(int(v) for v in (parsed if isinstance(parsed, list) else [parsed]))
    return ids


_SWEEPS = _sweep_scenarios()


def test_the_example_sweeps_are_found() -> None:
    assert "intersection_passing/right_turn_sweep" in _SWEEPS
    assert len(_SWEEPS) >= 10


@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize("scenario", _SWEEPS)
def test_no_case_touches_a_lanelet_without_a_3d_model(
    scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.examples.compose_errors import compose_or_report
    from autoware_carla_scenario.registry import load_scenario_plugins
    from autoware_carla_scenario.sweeper.expand import expand_config

    # The picks a binding cannot route from are dropped with a warning each;
    # they would bury a failure's message.
    caplog.set_level(logging.ERROR)
    load_scenario_plugins()
    cfg = compose_or_report(scenario, [])
    assert cfg is not None, f"{scenario} does not compose"
    no_3d = set(OmegaConf.select(cfg, "map.no_3d_model_lanelet_ids") or [])
    assert no_3d, "the fixture map lists lanelets without a 3D model"

    cases = expand_config(cfg)
    assert cases and cases != [[]], f"{scenario} expands to no concrete case"
    offending = [
        (case, sorted(_lanelet_ids(case) & no_3d))
        for case in cases
        if _lanelet_ids(case) & no_3d
    ]
    assert not offending, (
        f"{scenario}: {len(offending)} case(s) on lanelets without a 3D model: "
        f"{offending}"
    )


@pytest.mark.usefixtures("at_repo_root")
def test_the_odd_reads_the_right_turns_on_their_picks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every right-turn case is a junction case to the ODD, although its
    stop_line_offset walks its spawn back onto the lane before the junction:
    the road a case is on is the constraints' pick (docs/odd.md, #2454)."""
    from autoware_carla_scenario.examples.compose_errors import compose_or_report
    from autoware_carla_scenario.odd.sampler import ODD_CASE_KEY, decode_case
    from autoware_carla_scenario.registry import load_scenario_plugins
    from autoware_carla_scenario.sweeper.expand import expand_config

    caplog.set_level(logging.ERROR)
    load_scenario_plugins()
    cfg = compose_or_report(
        "intersection_passing/right_turn_sweep",
        ["++sweep.odd_sample.count=20", "++sweep.odd_sample.seed=1"],
    )
    assert cfg is not None

    cases = expand_config(cfg)

    assert len(cases) == 20
    junctions = []
    for case in cases:
        encoded = next(o for o in case if o.startswith(f"++{ODD_CASE_KEY}="))
        sample, _space = decode_case(encoded.split("=", 1)[1])
        junctions.append(
            {k: v for k, v in sample.buckets.items() if k.endswith("junction_type")}
        )
    assert all(j and set(j.values()) == {"intersection"} for j in junctions), junctions
