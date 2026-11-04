"""Every example scenario YAML lists every field of its config class (#1317).

Hydra composes ``examples/conf`` in struct mode with no schema registered for
``scenario``, so a ``scenario.<key>=`` override of a field the YAML leaves out
fails ("Could not override ... use +") although the config class has it and
the docs tell users to set it.  Each ``conf/scenario/**/*.yaml`` therefore
lists every field of the config class its ``scenario.name`` names
(``examples/configs.py``), and the overrides the docs show compose.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import autoware_carla_scenario.examples as examples_package
from autoware_carla_scenario.examples import configs

_CONF = Path(examples_package.__file__).parent / "conf"
_SCENARIO_DIR = _CONF / "scenario"

#: Each example config class by the ``name`` it defaults to, which is the
#: name it is registered under (``examples/run.py``).
_CONFIG_CLASSES: dict[str, type] = {
    str(cls.__dataclass_fields__["name"].default): cls
    for _, cls in inspect.getmembers(configs, inspect.isclass)
    if dataclasses.is_dataclass(cls)
    and cls.__module__ == configs.__name__
    and "name" in cls.__dataclass_fields__
}

_SCENARIOS = sorted(
    path.relative_to(_SCENARIO_DIR).with_suffix("").as_posix()
    for path in _SCENARIO_DIR.rglob("*.yaml")
)


def _scenario_block(scenario: str) -> dict[str, Any]:
    document = yaml.safe_load((_SCENARIO_DIR / f"{scenario}.yaml").read_text())
    block = document["scenario"]
    assert isinstance(block, dict)
    return block


def _compose(scenario: str, overrides: list[str]) -> Any:
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(_CONF), version_base=None):
            return compose(
                config_name="config", overrides=[f"scenario={scenario}", *overrides]
            )
    finally:
        GlobalHydra.instance().clear()


def test_the_example_scenarios_and_config_classes_are_found() -> None:
    assert "pedestrian_dart_out/pedestrian_dart_out" in _SCENARIOS
    assert len(_SCENARIOS) >= 20
    assert {
        "intersection_passing",
        "traffic_light_compliance",
        "lane_change",
        "pedestrian_dart_out",
        "cut_in",
        "temporary_stop",
        "adversity",
    } <= set(_CONFIG_CLASSES)


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_every_config_field_is_in_the_yaml(scenario: str) -> None:
    block = _scenario_block(scenario)
    config_cls = _CONFIG_CLASSES[block["name"]]
    fields = [f.name for f in dataclasses.fields(config_cls)]
    missing = [name for name in fields if name not in block]
    unknown = [key for key in block if key not in fields]
    assert not missing, (
        f"{scenario}.yaml leaves out {config_cls.__name__} fields {missing}: "
        "a scenario.<key>= override of them does not compose. List each with "
        "its default (null for an Optional one)."
    )
    assert not unknown, f"{scenario}.yaml sets unknown keys {unknown}"


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_the_composed_scenario_builds_its_config(scenario: str) -> None:
    cfg = _compose(scenario, [])
    block = OmegaConf.to_container(cfg.scenario, resolve=True)
    assert isinstance(block, dict)
    _CONFIG_CLASSES[str(block["name"])](**{str(k): v for k, v in block.items()})


#: The ``scenario.<key>=`` overrides the docs and tools show for fields the
#: YAMLs used to leave out (#1317).
_DOCUMENTED_OVERRIDES: list[tuple[str, list[str]]] = [
    (
        "pedestrian_dart_out/pedestrian_dart_out",
        [
            "scenario.turn_back_ttc_s=1.5",
            "scenario.turn_back_distance_m=8.0",
            "scenario.turn_back_within_m=1.0",
            "scenario.walker_type=walker.pedestrian.0001",
        ],
    ),
    ("adversity/brake_check", ["scenario.ego_speed_kmh=30"]),
    (
        "intersection_passing/left_turn",
        [
            "scenario.turn_indicator_before_m=30",
            "scenario.turn_indicator_grace_seconds=1.0",
            "scenario.max_steering_reversals=3",
            "scenario.max_turn_indicator_changes=3",
            "scenario.min_speed_kmh=3.0",
        ],
    ),
    (
        "traffic_light_compliance/traffic_light_compliance",
        [
            "scenario.stop_line_distance_m=10",
            "scenario.timeout_after_green_seconds=10",
            "scenario.moving_speed_kmh=15.0",
        ],
    ),
    ("cut_in/left", ["scenario.npc_vehicle_type=vehicle.tesla.model3"]),
]


@pytest.mark.parametrize(("scenario", "overrides"), _DOCUMENTED_OVERRIDES)
def test_the_documented_overrides_compose(scenario: str, overrides: list[str]) -> None:
    cfg = _compose(scenario, overrides)
    for override in overrides:
        key, _, value = override.partition("=")
        assert OmegaConf.select(cfg, key) == yaml.safe_load(value), override
