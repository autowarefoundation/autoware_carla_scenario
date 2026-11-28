"""The digest of a concrete scenario (autoware_carla_scenario.scenario_hash)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typesafe_carla.carla as carla
from omegaconf import OmegaConf

from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform
from autoware_carla_scenario.scenario_hash import (
    OTHER_SECTIONS,
    SCENARIO_SECTIONS,
    SCENARIO_SETTINGS,
    scenario_hash,
)

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


def test_every_section_of_the_runs_config_is_hashed_or_not() -> None:
    # A section added to the config has to be put on one side: left out of
    # both, runs differing in it would pass as rollouts of one scenario.
    import autoware_carla_scenario.examples as examples

    config = OmegaConf.load(Path(examples.__file__).parent / "conf" / "config.yaml")
    sections = {str(key) for key in config if key != "defaults"}
    for default in config.defaults:
        if isinstance(default, str):
            continue  # _self_
        # Hydra's own groups (`override hydra/help: ...`) are no section.
        groups = (str(key).removeprefix("override ") for key in default)
        sections.update(group for group in groups if not group.startswith("hydra"))
    assert not set(SCENARIO_SECTIONS) & set(OTHER_SECTIONS)
    assert sections <= set(SCENARIO_SECTIONS) | set(OTHER_SECTIONS), sorted(
        sections - set(SCENARIO_SECTIONS) - set(OTHER_SECTIONS)
    )


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


def test_the_fog_density_scale_is_part_of_it_once_set() -> None:
    """server.fog_density_scale scales how thick the environment's fog
    renders, LiDAR fog included (#2300): runs that differ in it are not
    rollouts of one scenario.  The rest of server is still left out, and an
    unset scale (null, the server's own) keeps the digest it had before."""
    assert ("server", "fog_density_scale") in SCENARIO_SETTINGS
    base = scenario_hash(_config(), CLASS)
    unset = {"host": "localhost", "port": 2000, "fog_density_scale": None}
    assert scenario_hash(_config(server=unset), CLASS) == base

    def scaled(scale: object, **server: object) -> str:
        return scenario_hash(
            _config(server={"fog_density_scale": scale, **server}), CLASS
        )

    assert scaled(0.01) != base
    assert scaled(0.01) != scaled(0.001)
    assert scaled(0.01) == scaled(1e-2, host="node7", world_reset="soft")
    assert scaled(1) == scaled(1.0)


def test_the_scenarios_class_is_part_of_it() -> None:
    assert scenario_hash(_config(), CLASS) != scenario_hash(_config(), "pkg.Other")


def test_the_order_the_config_holds_its_keys_in_does_not_matter() -> None:
    config = _config()
    reordered = {key: config[key] for key in reversed(config)}
    reordered["scenario"] = {"cut_in_distance_m": 12.0, "name": "cut_in"}
    assert scenario_hash(reordered, CLASS) == scenario_hash(config, CLASS)


def test_a_value_json_has_no_type_for_is_written_as_text() -> None:
    config = _config(scenario={"name": "cut_in", "route": Path("/routes/a.yaml")})
    assert scenario_hash(config, CLASS) == scenario_hash(
        _config(scenario={"name": "cut_in", "route": "/routes/a.yaml"}), CLASS
    )


@pytest.mark.parametrize(
    ("whole", "spelled"),
    [
        ({"timeout_seconds": 15}, {"timeout_seconds": 15.0}),
        ({"speeds": [10, 20]}, {"speeds": [10.0, 20.0]}),
        ({"nested": {"gap": {"m": 0}}}, {"nested": {"gap": {"m": -0.0}}}),
    ],
)
def test_equal_numbers_digest_alike(
    whole: dict[str, Any], spelled: dict[str, Any]
) -> None:
    """scenario.timeout_seconds=15 against the YAML's 15.0 is one scenario
    (#2481): a whole float is digested as the integer it equals, at any
    depth of a hashed section."""

    def scenario(**values: Any) -> str:
        return scenario_hash(_config(scenario={"name": "cut_in", **values}), CLASS)

    assert scenario(**whole) == scenario(**spelled)
    # A number that differs still differs, and a flag is not a number.
    assert scenario(timeout_seconds=15) != scenario(timeout_seconds=15.5)
    assert scenario(flag=True) != scenario(flag=1)
    assert scenario(flag=False) != scenario(flag=0.0)


def test_where_the_maps_files_sit_is_not_part_of_it() -> None:
    """The same map read from another directory -- an absolute path, another
    checkout, NISHISHINJUKU_XODR_PATH -- is one scenario (#2481); its name,
    source and pack still tell maps apart."""
    relative = _config(
        map={
            "name": "NishishinjukuMap",
            "xodr_path": "data/nishishinjuku_carla.xodr",
            "lanelet2_path": "data/nishishinjuku.osm",
            "overwrite_xodr": True,
        }
    )
    absolute = _config(
        map={
            "name": "NishishinjukuMap",
            "xodr_path": "/home/a/data/nishishinjuku_carla.xodr",
            "lanelet2_path": "/home/a/data/nishishinjuku.osm",
            "directory": "/home/a/maps",
            "overwrite_xodr": True,
        }
    )
    assert scenario_hash(absolute, CLASS) == scenario_hash(relative, CLASS)
    assert scenario_hash(absolute, CLASS) != scenario_hash(
        _config(map={**absolute["map"], "name": "Town01"}), CLASS
    )
    for key, value in (
        ("source", "git+https://example.com/maps@abc#Town10HD_Opt"),
        ("pack", "owner/repo/pack@1.0"),
        ("overwrite_xodr", False),
    ):
        assert scenario_hash(
            _config(map={**absolute["map"], key: value}), CLASS
        ) != scenario_hash(absolute, CLASS), key
    # Only the map's own paths: a path elsewhere is the scenario's.
    assert scenario_hash(
        _config(scenario={"name": "cut_in", "route_path": "a.yaml"}), CLASS
    ) != scenario_hash(
        _config(scenario={"name": "cut_in", "route_path": "b.yaml"}), CLASS
    )


def test_where_the_traffic_backend_keeps_its_files_is_not_part_of_it() -> None:
    """A SLURM node with a network cache of its own, another SUMO executable
    or SUMO's floating car data recorded is one scenario (#2620); the
    network and the demand the options name still tell scenarios apart."""
    options = {"tm_port": 8100, "net_path": None, "route_path": None}
    base = _config(traffic={"backend": "sumo", "options": options})
    local = _config(
        traffic={
            "backend": "sumo",
            "options": {
                **options,
                "cache_dir": "/scratch/node7/sumo",
                "binary": "/opt/sumo/bin/sumo-gui",
                "fcd_output": True,
            },
        }
    )
    assert scenario_hash(local, CLASS) == scenario_hash(base, CLASS)
    for key in ("net_path", "route_path"):
        changed = _config(
            traffic={"backend": "sumo", "options": {**options, key: "a.xml"}}
        )
        assert scenario_hash(changed, CLASS) != scenario_hash(base, CLASS), key
    # Only the backend's options: a `binary` elsewhere is the scenario's.
    assert scenario_hash(
        _config(scenario={"name": "cut_in", "binary": "a"}), CLASS
    ) != scenario_hash(_config(scenario={"name": "cut_in", "binary": "b"}), CLASS)


def test_an_odd_path_counts_as_written() -> None:
    # Documented (module docstring, docs/wosac.md): the file is not read.
    assert scenario_hash(_config(odd="odd.yaml"), CLASS) != scenario_hash(
        _config(odd="/home/a/odd.yaml"), CLASS
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


def test_build_scenario_hashes_the_fog_density_scale() -> None:
    from autoware_carla_scenario.examples.run import _set_scenario_hash

    def built(scale: object) -> str | None:
        server = {"host": "localhost", "port": 2000, "fog_density_scale": scale}
        scenario = _Scenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0, y=0, z=0))
                ),
                vehicle_type="vehicle.mini.cooper",
            )
        )
        _set_scenario_hash(OmegaConf.create(_config(server=server)), scenario)
        return scenario.scenario_hash

    name = f"{__name__}._Scenario"
    assert built(0.01) == scenario_hash(
        _config(server={"fog_density_scale": 0.01}), name
    )
    assert built(None) == scenario_hash(_config(), name)
    assert built(0.01) != built(None)
