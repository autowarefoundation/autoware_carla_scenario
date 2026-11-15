"""The digest of a concrete scenario (autoware_carla_scenario.scenario_hash)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typesafe_carla.carla as carla
from omegaconf import OmegaConf

from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform
from autoware_carla_scenario.scenario_hash import SCENARIO_SECTIONS, scenario_hash

CLASS = "pkg.CutInScenario"


def _config(**changes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed": 0,
        "server": {"host": "localhost", "port": 2000},
        "traffic_manager": {"port": 8100},
        "rerun": {"enabled": False},
        "scenario": {"name": "cut_in", "cut_in_distance_m": 12.0},
        "map": {"name": "Town10HD_Opt"},
        "ego": {"spawn_lanelet_id": 242, "spawn_s": 25.0},
        "background_traffic": {"seed": 5, "source": {"initial_vehicles": 3}},
        "traffic": {"backend": "sumo", "options": {"tm_port": 8100}},
        "autoware": {"address": "localhost:50051"},
    }
    config.update(changes)
    return config


def test_the_hash_is_a_sha256_hex_digest() -> None:
    digest = scenario_hash(_config(), CLASS)
    assert len(digest) == 64
    assert int(digest, 16) >= 0


def test_the_seed_is_not_part_of_it() -> None:
    assert "seed" not in SCENARIO_SECTIONS
    assert scenario_hash(_config(seed=1), CLASS) == scenario_hash(
        _config(seed=2), CLASS
    )


def test_where_a_run_connects_is_not_part_of_it() -> None:
    # What a SLURM task moves to its own slot: ports, addresses, ROS domains.
    moved = _config(
        server={"host": "node7", "port": 2004},
        traffic_manager={"port": 8104},
        traffic={"backend": "sumo", "options": {"tm_port": 8104}},
        autoware={"address": "localhost:50055"},
        environment={"launcher": {"ros_domain_ids": [3], "ROS_DOMAIN_ID": "3"}},
    )
    base = _config(
        environment={"launcher": {"ros_domain_ids": [0], "ROS_DOMAIN_ID": "0"}}
    )
    assert scenario_hash(moved, CLASS) == scenario_hash(base, CLASS)


def test_how_a_run_is_recorded_is_not_part_of_it() -> None:
    assert scenario_hash(_config(rerun={"enabled": True}), CLASS) == scenario_hash(
        _config(), CLASS
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"scenario": {"name": "cut_in", "cut_in_distance_m": 15.0}},
        {"map": {"name": "Town01"}},
        {"ego": {"spawn_lanelet_id": 242, "spawn_s": 30.0}},
        {"background_traffic": {"seed": 6, "source": {"initial_vehicles": 3}}},
        {"traffic": {"backend": "traffic_manager", "options": {"tm_port": 8100}}},
        {"odd_case": "case-3"},
        {"environment": {"fog_density": 40.0}},
        {"controls": {"npc_speed": 30.0}},
    ],
)
def test_what_is_simulated_is_part_of_it(changes: dict[str, Any]) -> None:
    assert scenario_hash(_config(**changes), CLASS) != scenario_hash(_config(), CLASS)


def test_the_scenarios_class_is_part_of_it() -> None:
    assert scenario_hash(_config(), CLASS) != scenario_hash(_config(), "pkg.Other")


def test_the_order_the_config_holds_its_keys_in_does_not_matter() -> None:
    config = _config()
    reordered = {key: config[key] for key in reversed(config)}
    reordered["scenario"] = {"cut_in_distance_m": 12.0, "name": "cut_in"}
    assert scenario_hash(reordered, CLASS) == scenario_hash(config, CLASS)


def test_a_value_json_has_no_type_for_is_written_as_text() -> None:
    config = _config(map={"lanelet2_path": Path("/maps/a.osm")})
    assert scenario_hash(config, CLASS) == scenario_hash(
        _config(map={"lanelet2_path": "/maps/a.osm"}), CLASS
    )


class _Scenario(BaseScenario):
    def setup(self) -> None: ...

    def is_done(self) -> bool:
        return True


def test_build_scenario_gives_the_scenario_the_hash_of_its_config() -> None:
    from autoware_carla_scenario.examples.run import _set_scenario_hash

    cfg = OmegaConf.create(_config(seed=3))
    scenario = _Scenario(
        EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0, y=0, z=0))
            ),
            vehicle_type="vehicle.mini.cooper",
        )
    )
    assert scenario.scenario_hash is None
    _set_scenario_hash(cfg, scenario)
    assert scenario.scenario_hash == scenario_hash(
        _config(seed=3), f"{__name__}._Scenario"
    )
