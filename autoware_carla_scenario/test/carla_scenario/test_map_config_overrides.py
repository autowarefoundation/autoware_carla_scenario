"""Every ``MapConfig`` field is a plain override on the shipped map groups.

Hydra composes in struct mode, so a key the composed ``map`` block does not
declare is refused without a ``+`` -- ``map.projector_type=utm`` (docs/maps.md)
and ``map.overwrite_xodr=true`` once were.
"""

from __future__ import annotations

import dataclasses

import pytest

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.scenario_config import MapConfig


@pytest.mark.parametrize("map_name", ["town10hd_opt", "nishishinjuku"])
def test_every_map_config_field_is_declared(map_name: str) -> None:
    cfg = run._compose_config("adversity/nade", [f"map={map_name}"])
    missing = {field.name for field in dataclasses.fields(MapConfig)} - set(cfg.map)
    assert not missing


@pytest.mark.parametrize("map_name", ["town10hd_opt", "nishishinjuku"])
def test_projector_type_and_overwrite_xodr_are_plain_overrides(map_name: str) -> None:
    cfg = run._compose_config(
        "adversity/nade",
        [f"map={map_name}", "map.projector_type=utm", "map.overwrite_xodr=true"],
    )
    assert cfg.map.projector_type == "utm"
    assert cfg.map.overwrite_xodr is True


def test_the_map_groups_keep_their_own_values() -> None:
    assert run._compose_config(
        "adversity/nade", ["map=nishishinjuku"]
    ).map.overwrite_xodr
    town = run._compose_config("adversity/nade", ["map=town10hd_opt"]).map
    assert town.overwrite_xodr is False
    assert town.projector_type is None
