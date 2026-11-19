"""The adversity examples (#404): what the shipped cases say, without CARLA.

That each one fires and is carried out is checked on a CARLA server
(docs/scenarios/adversity.md); here, that the cases compose into the kinds
the scenario knows and that it refuses one it does not.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla
from omegaconf import OmegaConf

from autoware_carla_scenario import (
    EgoConfig,
    Lanelet2Pose,
    SpawnTransform,
    TimeoutCondition,
)
from autoware_carla_scenario.examples import adversity, run
from autoware_carla_scenario.examples.adversity import (
    VEHICLE_KINDS,
    AdversityScenario,
)
from autoware_carla_scenario.examples.configs import (
    ADVERSITY_KINDS,
    ADVERSITY_VEHICLE_KINDS,
    NADE_TIMEOUT_MARGIN_SECONDS,
    AdversityConfig,
)
from autoware_carla_scenario.scenario_runner import _register_default_timeout


def _scenario(config: AdversityConfig) -> AdversityScenario:
    return AdversityScenario(
        EgoConfig(spawn_location=SpawnTransform(carla.Transform())),
        Lanelet2Pose(lanelet_id=324, s=5.0),
        config,
    )


def _shipped_config(case: str, *overrides: str) -> AdversityConfig:
    cfg = run._compose_config(case, ["map=town10hd_opt", *overrides])
    settings: Any = OmegaConf.to_container(cfg.scenario, resolve=True)
    return AdversityConfig(**settings)


def _timeouts(scenario: AdversityScenario) -> list[float]:
    return [
        c.timeout_seconds
        for c in scenario._fail_conditions
        if isinstance(c, TimeoutCondition)
    ]


@pytest.mark.parametrize("case", run._resolve_scenario_glob("adversity/*"))
def test_nade_on_every_shipped_case_times_out_only_after_it_passes(case: str) -> None:
    """#2202: NADE registered no timeout, so the run's 30 s fail-safe failed a
    45 s NADE run before it could pass, on every case but nade.yaml."""
    config = _shipped_config(case, "scenario.kind=nade")
    scenario = _scenario(config)
    scenario._setup_nade()
    (timeout,) = _timeouts(scenario)
    assert timeout == config.effective_timeout_seconds()
    assert timeout >= config.nade_seconds + NADE_TIMEOUT_MARGIN_SECONDS
    assert timeout > config.nade_seconds
    # The runner adds its default -- the run's scenario.timeout_seconds,
    # shorter than NADE's run on these cases -- only to a scenario with no
    # timeout of its own, so this one is the run's.
    assert not _register_default_timeout(scenario, config.timeout_seconds, "adversity")
    assert _timeouts(scenario) == [timeout]


def test_the_default_config_s_nade_times_out_only_after_it_passes() -> None:
    config = AdversityConfig(kind="nade")
    assert config.timeout_seconds < config.nade_seconds
    assert config.effective_timeout_seconds() == (
        config.nade_seconds + NADE_TIMEOUT_MARGIN_SECONDS
    )


def test_a_longer_nade_timeout_is_kept() -> None:
    config = AdversityConfig(kind="nade", nade_seconds=10.0, timeout_seconds=90.0)
    assert config.effective_timeout_seconds() == 90.0


@pytest.mark.parametrize("kind", [k for k in ADVERSITY_KINDS if k != "nade"])
def test_a_single_adversity_keeps_its_own_timeout(kind: str) -> None:
    config = AdversityConfig(kind=kind, timeout_seconds=12.0, nade_seconds=45.0)
    assert config.effective_timeout_seconds() == 12.0


@pytest.mark.parametrize(
    ("rule", "offset", "t", "heading"),
    [
        ("RHT", 3.0, -3.0, 1),
        ("RHT", -3.0, -3.0, 1),
        ("LHT", 3.0, 3.0, -1),
        ("LHT", -3.0, 3.0, -1),
    ],
)
def test_the_jaywalker_waits_at_the_kerb_the_road_s_traffic_keeps_to(
    monkeypatch: pytest.MonkeyPatch, rule: str, offset: float, t: float, heading: int
) -> None:
    """As the dart-out pedestrian does: the sign of npc_offset_m used to pick
    the side, whatever the road's traffic rule."""
    kerbs: list[Lanelet2Pose] = []

    def spawn_transform(kerb: Lanelet2Pose) -> carla.Transform:
        kerbs.append(kerb)
        return carla.Transform()

    monkeypatch.setattr(adversity, "road_traffic_rule", lambda road_id: rule)
    monkeypatch.setattr(adversity, "to_opendrive", lambda pose: MagicMock(road_id="1"))
    monkeypatch.setattr(adversity, "walker_spawn_transform", spawn_transform)
    monkeypatch.setattr(adversity, "PedestrianEntity", MagicMock())
    scenario = _scenario(
        AdversityConfig(kind="jaywalking", npc_ahead_m=35.0, npc_offset_m=offset)
    )
    scenario.set_client(MagicMock())
    monkeypatch.setattr(scenario, "register_pedestrian", MagicMock())
    scenario._spawn_pedestrian(Lanelet2Pose(lanelet_id=324, s=5.0))
    (kerb,) = kerbs
    assert (kerb.lanelet_id, kerb.s, kerb.t) == (324, 40.0, t)
    assert kerb.heading == pytest.approx(heading * math.pi / 2)


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


def test_the_nade_sampler_follows_the_run_s_seed_by_default() -> None:
    """No shipped case pins the sampler's seed, so a seed sweep draws other
    adversities in each case, as the scenario's other adversarial actions do
    (docs/usage.md#one-seed-for-the-whole-run)."""
    assert AdversityConfig().nade_seed is None
    for case in run._resolve_scenario_glob("adversity/*"):
        cfg = run._compose_config(case, ["map=town10hd_opt"])
        assert cfg.scenario.nade_seed is None, case


def test_a_nade_seed_of_its_own_is_kept() -> None:
    cfg = run._compose_config(
        "adversity/nade", ["map=town10hd_opt", "scenario.nade_seed=7"]
    )
    assert cfg.scenario.nade_seed == 7


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


def test_an_unknown_kind_is_refused_when_the_config_is_built() -> None:
    """#1816: when the config is built, so a batch refuses it before anything
    starts, naming its key."""
    with pytest.raises(
        ValueError,
        match=(
            r"^scenario\.kind must be one of brake_check, ignore_leader, cut_in, "
            r"traffic_rule, jaywalking, nade, not 'drift'$"
        ),
    ):
        AdversityConfig(kind="drift")


@pytest.mark.parametrize("kind", ADVERSITY_KINDS)
def test_every_kind_is_accepted(kind: str) -> None:
    assert AdversityConfig(kind=kind).kind == kind


def test_the_vehicle_kinds_are_the_shared_ones() -> None:
    assert VEHICLE_KINDS == ADVERSITY_VEHICLE_KINDS
    assert set(ADVERSITY_KINDS) == {*VEHICLE_KINDS, "jaywalking", "nade"}


def test_an_unknown_kind_is_refused_before_anything_is_spawned() -> None:
    """The setup's backstop, for a config changed after it was built."""
    config = AdversityConfig()
    config.kind = "drift"
    scenario = _scenario(config)
    with pytest.raises(ValueError, match=r"^scenario\.kind must be one of"):
        scenario.setup()
