"""utils.config.checked_options: unknown keys refused with the config's name."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.utils.config import checked_options


@dataclass
class _Config:
    a: int = 1


def test_unknown_keys_that_mix_ints_and_strings_name_the_config() -> None:
    """YAML reads ``1:`` as an int key; sorting it among str ones raised a
    bare ``TypeError`` that never named the config (#2507)."""
    mapping = OmegaConf.create("{a: 2, 1: x, b: 3}")
    with pytest.raises(ValueError, match=r"Unknown _Config key\(s\): \[1, 'b'\]"):
        checked_options(_Config, mapping)  # type: ignore[arg-type]


def test_known_keys_pass_and_ignored_ones_are_dropped() -> None:
    assert checked_options(_Config, {"a": 2, "alias": 3}, ignore={"alias"}) == {"a": 2}
