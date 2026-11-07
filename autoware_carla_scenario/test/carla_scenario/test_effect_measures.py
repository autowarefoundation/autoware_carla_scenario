"""A drawn case's effects measured in the run, beside the ODD (openodd's
``effects``): the measure each control is keyed by reads what the case drew."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

from autoware_carla_scenario.coverage import CoverageCollector
from autoware_carla_scenario.examples.run import add_effect_measures
from autoware_carla_scenario.measures import set_measured_scenario
from autoware_carla_scenario.scenario_runner import _effect_measures

CONF = Path(__file__).resolve().parents[2] / (
    "src/autoware_carla_scenario/examples/conf/scenario"
)


@pytest.fixture(autouse=True)
def _no_scenario() -> Iterator[None]:
    set_measured_scenario(None)
    yield
    set_measured_scenario(None)


def _controls(path: Path) -> Any:
    return OmegaConf.create({"controls": yaml.safe_load(path.read_text())["controls"]})


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            "cut_in/left.yaml",
            {
                "vehicle_ahead_gap": "vehicle_ahead_gap_m",
                "vehicle_ahead_relative_speed": "vehicle_ahead_relative_speed_kph",
            },
        ),
        (
            "pedestrian_dart_out/pedestrian_dart_out.yaml",
            {
                "crossing_pedestrian_gap": "crossing_pedestrian_gap_m",
                "crossing_pedestrian_speed": "crossing_pedestrian_speed_ms",
            },
        ),
    ],
)
def test_each_effect_is_measured_by_the_measure_its_control_sets(
    config: str, expected: dict[str, str]
) -> None:
    scenario = SimpleNamespace(_effect_measures={})
    add_effect_measures(_controls(CONF / config), scenario)  # type: ignore[arg-type]
    assert scenario._effect_measures == expected


def test_a_control_without_a_quantity_measures_no_effect() -> None:
    scenario = SimpleNamespace(_effect_measures={"stale": "x"})
    cfg = OmegaConf.create(
        {"controls": {"vehicle_ahead_gap_m": {"key": "scenario.gap"}}}
    )
    add_effect_measures(cfg, scenario)  # type: ignore[arg-type]
    assert scenario._effect_measures == {}
    add_effect_measures(OmegaConf.create({}), scenario)  # type: ignore[arg-type]
    assert scenario._effect_measures == {}


def test_a_run_measures_the_effects_its_case_drew() -> None:
    scenario = SimpleNamespace(
        _effect_measures={"vehicle_ahead_gap": "vehicle_ahead_gap_m", "other": "x_m"}
    )
    case = (SimpleNamespace(quantities={"vehicle_ahead_gap": 12.0}), None)
    assert _effect_measures(scenario, case) == {  # type: ignore[arg-type]
        "vehicle_ahead_gap": "vehicle_ahead_gap_m"
    }
    # A run no sweep drew measures none.
    assert _effect_measures(scenario, None) == {}  # type: ignore[arg-type]


def test_the_collector_writes_what_the_measure_read() -> None:
    """The scenario's measure, read on each tick of the run's first seconds,
    into ``effects.samples``: no cover item, no grade."""
    readings = iter([11.5, 11.8, 12.1])
    set_measured_scenario(
        SimpleNamespace(measure=lambda key, world: next(readings))  # type: ignore[arg-type]
    )
    collector = CoverageCollector(
        [], effect_measures={"vehicle_ahead_gap": "vehicle_ahead_gap_m"}
    )
    world = object()
    collector.start(world, 0.0)
    for t in (0.5, 1.0, 1.5):
        collector.tick(world, t)
    collector.end(world, 2.0)
    doc = collector.to_dict("cut_in")
    samples = doc["effects"]["samples"]
    assert samples["quantities"] == ["vehicle_ahead_gap"]
    assert samples["rows"] == [[0.5, 11.5], [1.0, 11.8], [1.5, 12.1]]
    assert doc["items"] == []
