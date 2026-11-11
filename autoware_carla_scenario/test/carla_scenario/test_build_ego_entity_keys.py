"""``build_ego_entity`` refuses keys its groups do not define, and reads a null
``driver.control`` as the default gains (issue #1408)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from autoware_carla_scenario.autoware_bridge import AutowareBridgeConfig
from autoware_carla_scenario.driver.control import ControlConfig

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


def _composed(overrides: List[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ("+autoware.timout_s=30", "timout_s"),
        ("+autoware.base_link_offset=-1.4", "base_link_offset"),
    ],
)
def test_an_unknown_autoware_key_is_refused(override: str, key: str) -> None:
    from autoware_carla_scenario.examples.run import build_ego_entity

    with pytest.raises(ValueError, match=f"^autoware: .*'{key}'") as info:
        build_ego_entity(_composed(["ego.entity=autoware", override]))
    # The launcher is the group's too, not a key the bridge refuses.
    assert "'launcher'" not in str(info.value).split("Known keys")[0]


def test_every_known_autoware_key_is_applied() -> None:
    from autoware_carla_scenario.entity import AutowareEgoEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(
        _composed(
            [
                "ego.entity=autoware",
                "autoware.address=localhost:50099",
                "autoware.ready_timeout_ticks=600",
                "autoware.boot_timeout_s=120",
                "autoware.liveness_timeout_s=12",
                "+autoware.timeout_s=30",
                "+autoware.boot_tick_period_s=0.1",
                "+autoware.base_link_offset_m=-1.4",
                "autoware.launcher.type=none",
            ]
        )
    )
    assert isinstance(entity, AutowareEgoEntity)
    assert entity.config == AutowareBridgeConfig(
        address="localhost:50099",
        timeout_s=30.0,
        ready_timeout_ticks=600,
        base_link_offset_m=-1.4,
        boot_timeout_s=120.0,
        boot_tick_period_s=0.1,
        liveness_timeout_s=12.0,
    )


def test_a_null_driver_control_uses_the_default_gains() -> None:
    from autoware_carla_scenario import CarlaDriverEntity
    from autoware_carla_scenario.examples.run import build_ego_entity

    entity = build_ego_entity(
        _composed(["ego.entity=carla_driver", "driver.control=null"])
    )
    assert isinstance(entity, CarlaDriverEntity)
    assert entity._follower._config == ControlConfig()
