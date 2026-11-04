"""The adversity examples (#404): what the shipped cases say, without CARLA.

That each one fires and is carried out is checked on a CARLA server
(docs/scenarios/adversity.md); here, that the cases compose into the kinds
the scenario knows and that it refuses one it does not.
"""

from __future__ import annotations

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import EgoConfig, Lanelet2Pose, SpawnTransform
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.examples.adversity import (
    VEHICLE_KINDS,
    AdversityScenario,
)
from autoware_carla_scenario.examples.configs import AdversityConfig


def test_there_is_a_case_for_each_phase_1_kind_and_for_nade() -> None:
    cases = run._resolve_scenario_glob("adversity/*")
    kinds = set()
    for case in cases:
        cfg = run._compose_config(case, ["map=town10hd_opt"])
        assert cfg.scenario.name == "adversity"
        assert case == f"adversity/{cfg.scenario.kind}"
        kinds.add(str(cfg.scenario.kind))
    assert kinds == {*VEHICLE_KINDS, "jaywalking", "nade"}


def test_nade_runs_among_background_traffic() -> None:
    cfg = run._compose_config("adversity/nade", ["map=town10hd_opt"])
    assert cfg.background_traffic.enabled is True
    assert cfg.scenario.nade_is_probability > cfg.scenario.nade_probability


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("nade_is_probability", "1.5"),
        ("nade_is_probability", "-0.1"),
        ("nade_probability", "2.0"),
        ("nade_cooldown", "-2.0"),
        ("nade_seconds", "0.0"),
    ],
)
def test_a_nade_value_out_of_bounds_is_refused_before_the_run(
    key: str, value: str
) -> None:
    """The batch check builds the scenario's config, so a value NadeSettings
    would refuse once CARLA is up is refused first, naming its key (#1458)."""
    cfg = run._compose_config(
        "adversity/nade", ["map=town10hd_opt", f"scenario.{key}={value}"]
    )
    with pytest.raises(run.RunConfigError, match=rf"scenario\.{key} must"):
        run.check_batch_configs([cfg])


def test_the_shipped_adversity_cases_pass_the_batch_check() -> None:
    for case in run._resolve_scenario_glob("adversity/*"):
        run.check_batch_configs([run._compose_config(case, ["map=town10hd_opt"])])


def test_an_unknown_kind_is_refused_before_anything_is_spawned() -> None:
    scenario = AdversityScenario(
        EgoConfig(spawn_location=SpawnTransform(carla.Transform())),
        Lanelet2Pose(lanelet_id=324, s=5.0),
        AdversityConfig(kind="drift"),
    )
    with pytest.raises(ValueError, match="adversity kind must be one of"):
        scenario.setup()
