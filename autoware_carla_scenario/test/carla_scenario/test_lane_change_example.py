"""The lane-change example's config (#1074), without CARLA.

``lane_change.py`` checks the failure outcome only for ``expect: fail``, so an
unknown ``expect`` has to be refused rather than checked as a success.
"""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.examples.configs import LaneChangeConfig


@pytest.mark.parametrize("pattern", ["lane_change/*", "lane_change_fail/*"])
def test_every_shipped_case_is_a_valid_config(pattern: str) -> None:
    cases = run._resolve_scenario_glob(pattern)
    assert cases
    for case in cases:
        cfg = run._compose_config(case, [])
        scenario = OmegaConf.to_container(cfg.scenario, resolve=True)
        assert isinstance(scenario, dict)
        config = LaneChangeConfig(**{str(k): v for k, v in scenario.items()})
        expected = "fail" if pattern.startswith("lane_change_fail") else "success"
        assert config.expect == expected


@pytest.mark.parametrize("expect", ["success", "fail"])
def test_a_known_outcome_is_accepted(expect: str) -> None:
    assert LaneChangeConfig(expect=expect).expect == expect


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"expect": "fial"}, r"expect must be 'success' or 'fail', not 'fial'"),
        ({"expect": "Fail"}, r"expect must be 'success' or 'fail'"),
        ({"expect": ""}, r"expect must be 'success' or 'fail'"),
        ({"direction": "up"}, r"direction must be 'left' or 'right', not 'up'"),
    ],
)
def test_an_unknown_value_is_refused(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LaneChangeConfig(**config)
