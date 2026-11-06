"""The built-in scenario configs refuse a value out of its bounds when they
are built, naming its key as ``scenario.<key>`` (#1815, #1816).

A threshold at 0 or below makes the check that reads it vacuous (#1815), and
the other values here were refused only by a lower-level class once CARLA was
up (#1816).  ``check_batch_configs`` builds the config
(``_check_scenario_config``), so a batch refuses them before anything starts.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.examples.configs import (
    AdversityConfig,
    CutInConfig,
    IntersectionPassingConfig,
    LaneChangeConfig,
    PedestrianDartOutConfig,
    TemporaryStopConfig,
    TrafficLightComplianceConfig,
)

#: Finite and above 0.
POSITIVE_BAD = [0.0, -1.0, math.nan, math.inf, -math.inf]
#: Finite and at least 0.
NOT_NEGATIVE_BAD = [-0.5, -5.0, math.nan, math.inf, -math.inf]

#: (config class, field) that must be finite and positive.
POSITIVE_FIELDS: list[tuple[type, str]] = [
    (IntersectionPassingConfig, "timeout_seconds"),
    (IntersectionPassingConfig, "min_speed_kmh"),
    (IntersectionPassingConfig, "speed_check_delay_seconds"),
    (IntersectionPassingConfig, "turn_indicator_grace_seconds"),
    (TrafficLightComplianceConfig, "timeout_seconds"),
    (TrafficLightComplianceConfig, "light_switch_delay_seconds"),
    (TrafficLightComplianceConfig, "moving_speed_kmh"),
    (TrafficLightComplianceConfig, "departure_speed_kmh"),
    (TrafficLightComplianceConfig, "timeout_after_green_seconds"),
    (TrafficLightComplianceConfig, "standstill_seconds"),
    (LaneChangeConfig, "timeout_seconds"),
    (PedestrianDartOutConfig, "trigger_distance_m"),
    (PedestrianDartOutConfig, "walk_speed_ms"),
    (PedestrianDartOutConfig, "timeout_seconds"),
    (CutInConfig, "cut_in_after_seconds"),
    (CutInConfig, "hold_seconds"),
    (CutInConfig, "timeout_seconds"),
    (TemporaryStopConfig, "s_margin"),
    (TemporaryStopConfig, "stop_duration"),
    (TemporaryStopConfig, "restart_speed_kmh"),
    (TemporaryStopConfig, "timeout_seconds"),
    (AdversityConfig, "timeout_seconds"),
    (AdversityConfig, "nade_seconds"),
]

#: (config class, field) that must be finite and not negative.
NOT_NEGATIVE_FIELDS: list[tuple[type, str]] = [
    (TrafficLightComplianceConfig, "stop_line_tolerance_m"),
    (TrafficLightComplianceConfig, "standstill_speed_mps"),
    (PedestrianDartOutConfig, "pass_beyond_m"),
    (PedestrianDartOutConfig, "turn_back_within_m"),
    (PedestrianDartOutConfig, "turn_back_ttc_s"),
    (PedestrianDartOutConfig, "turn_back_distance_m"),
    (CutInConfig, "npc_initial_speed_kmh"),
    (TemporaryStopConfig, "speed_threshold"),
    (AdversityConfig, "npc_speed_kmh"),
    (AdversityConfig, "ego_speed_kmh"),
    (AdversityConfig, "nade_cooldown"),
]


def _message(key: str, rest: str) -> str:
    return rf"^scenario\.{re.escape(key)} must be {rest}"


@pytest.mark.parametrize("value", POSITIVE_BAD)
@pytest.mark.parametrize(
    ("cls", "key"),
    POSITIVE_FIELDS,
    ids=[f"{c.__name__}.{k}" for c, k in POSITIVE_FIELDS],
)
def test_a_value_not_positive_is_refused(cls: type, key: str, value: float) -> None:
    with pytest.raises(ValueError, match=_message(key, "finite and positive, not ")):
        cls(**{key: value})


@pytest.mark.parametrize("value", NOT_NEGATIVE_BAD)
@pytest.mark.parametrize(
    ("cls", "key"),
    NOT_NEGATIVE_FIELDS,
    ids=[f"{c.__name__}.{k}" for c, k in NOT_NEGATIVE_FIELDS],
)
def test_a_negative_value_is_refused(cls: type, key: str, value: float) -> None:
    with pytest.raises(
        ValueError, match=_message(key, "finite and not negative, not ")
    ):
        cls(**{key: value})


@pytest.mark.parametrize("value", [1e-6, 0.3, 1000.0])
@pytest.mark.parametrize(
    ("cls", "key"),
    POSITIVE_FIELDS,
    ids=[f"{c.__name__}.{k}" for c, k in POSITIVE_FIELDS],
)
def test_a_small_positive_value_is_accepted(cls: type, key: str, value: float) -> None:
    assert getattr(cls(**{key: value}), key) == value


@pytest.mark.parametrize("value", [0.0, 1e-6, 25.0])
@pytest.mark.parametrize(
    ("cls", "key"),
    NOT_NEGATIVE_FIELDS,
    ids=[f"{c.__name__}.{k}" for c, k in NOT_NEGATIVE_FIELDS],
)
def test_zero_is_accepted_where_not_negative(cls: type, key: str, value: float) -> None:
    assert getattr(cls(**{key: value}), key) == value


@pytest.mark.parametrize(
    ("cls", "key"),
    [
        (IntersectionPassingConfig, "min_speed_kmh"),
        (TrafficLightComplianceConfig, "departure_speed_kmh"),
        (TrafficLightComplianceConfig, "timeout_after_green_seconds"),
        (PedestrianDartOutConfig, "turn_back_ttc_s"),
        (PedestrianDartOutConfig, "turn_back_distance_m"),
        (AdversityConfig, "ego_speed_kmh"),
    ],
)
def test_none_still_turns_an_optional_check_off(cls: type, key: str) -> None:
    assert getattr(cls(**{key: None}), key) is None


def test_every_default_is_accepted() -> None:
    for cls in (
        IntersectionPassingConfig,
        TrafficLightComplianceConfig,
        LaneChangeConfig,
        PedestrianDartOutConfig,
        CutInConfig,
        TemporaryStopConfig,
        AdversityConfig,
    ):
        cls()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"stop_zone_min_m": 4.0, "stop_zone_max_m": 3.0},
            r"^scenario\.stop_zone_min_m \(4\.0\) is past scenario\.stop_zone_max_m "
            r"\(3\.0\)$",
        ),
        ({"stop_zone_min_m": math.nan}, r"^scenario\.stop_zone_min_m must be finite"),
        ({"stop_zone_max_m": math.inf}, r"^scenario\.stop_zone_max_m must be finite"),
    ],
)
def test_the_stop_zone_must_be_in_order(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrafficLightComplianceConfig(**config)


@pytest.mark.parametrize(
    "zone", [(0.0, 0.0), (-1.0, 3.0), (2.0, 2.0), (0.0, 30.0)], ids=str
)
def test_a_stop_zone_in_order_is_accepted(zone: tuple[float, float]) -> None:
    low, high = zone
    config = TrafficLightComplianceConfig(stop_zone_min_m=low, stop_zone_max_m=high)
    assert (config.stop_zone_min_m, config.stop_zone_max_m) == zone


# ---------------------------------------------------------------------------
# The batch check refuses them before anything starts
# ---------------------------------------------------------------------------

#: A shipped case of each built-in scenario, with the overrides it needs.
_CASES = {
    "intersection_passing": ("intersection_passing/straight", []),
    "traffic_light_compliance": (
        "traffic_light_compliance/traffic_light_compliance",
        [],
    ),
    "lane_change": ("lane_change/left", []),
    "pedestrian_dart_out": ("pedestrian_dart_out/pedestrian_dart_out", []),
    "cut_in": ("cut_in/left", []),
    "temporary_stop": ("temporary_stop/temporary_stop", []),
    "adversity": ("adversity/brake_check", ["map=town10hd_opt"]),
}


@pytest.mark.parametrize(
    ("scenario", "override"),
    [
        ("intersection_passing", "scenario.min_speed_kmh=0"),
        ("intersection_passing", "scenario.speed_check_delay_seconds=-1"),
        ("intersection_passing", "scenario.timeout_seconds=0"),
        ("traffic_light_compliance", "scenario.departure_speed_kmh=0"),
        ("traffic_light_compliance", "scenario.moving_speed_kmh=-1"),
        ("traffic_light_compliance", "scenario.timeout_after_green_seconds=-1"),
        ("traffic_light_compliance", "scenario.stop_line_tolerance_m=-0.5"),
        ("traffic_light_compliance", "scenario.stop_zone_min_m=5"),
        ("lane_change", "scenario.timeout_seconds=nan"),
        ("pedestrian_dart_out", "scenario.trigger_distance_m=-20"),
        ("pedestrian_dart_out", "scenario.walk_speed_ms=0"),
        ("pedestrian_dart_out", "scenario.pass_beyond_m=-1"),
        ("pedestrian_dart_out", "scenario.turn_back_within_m=nan"),
        ("cut_in", "scenario.npc_initial_speed_kmh=-5"),
        ("cut_in", "scenario.hold_seconds=0"),
        ("temporary_stop", "scenario.restart_speed_kmh=0"),
        ("temporary_stop", "scenario.s_margin=0"),
        ("temporary_stop", "scenario.speed_threshold=-0.1"),
        ("temporary_stop", "scenario.stop_duration=inf"),
        ("adversity", "scenario.kind=brake_chek"),
        ("adversity", "scenario.npc_speed_kmh=-5"),
        ("adversity", "scenario.ego_speed_kmh=nan"),
        ("adversity", "scenario.timeout_seconds=0"),
    ],
)
def test_the_batch_check_refuses_an_override(scenario: str, override: str) -> None:
    case, overrides = _CASES[scenario]
    cfg = run._compose_config(case, [*overrides, override])
    key = override.split("=", 1)[0]
    with pytest.raises(run.RunConfigError, match=rf"^{re.escape(key)}\b"):
        run.check_batch_configs([cfg])


@pytest.mark.parametrize("scenario", sorted(_CASES))
def test_the_shipped_case_passes_the_batch_check(scenario: str) -> None:
    case, overrides = _CASES[scenario]
    run.check_batch_configs([run._compose_config(case, overrides)])


@pytest.mark.parametrize(
    "pattern",
    [
        "intersection_passing/*",
        "traffic_light_compliance/*",
        "lane_change/*",
        "lane_change_fail/*",
        "pedestrian_dart_out/*",
        "cut_in/*",
        "temporary_stop/*",
    ],
)
def test_every_shipped_case_builds_its_config(pattern: str) -> None:
    cases = run._resolve_scenario_glob(pattern)
    assert cases
    for case in cases:
        run._check_scenario_config(run._compose_config(case, []))


@pytest.mark.parametrize(
    ("override", "value"),
    [
        # The sweep and ODD bounds the shipped cases draw from (controls:).
        ("scenario.trigger_distance_m", 8.0),
        ("scenario.walk_speed_ms", 0.8),
    ],
)
def test_the_shipped_sweep_bounds_are_accepted(override: str, value: float) -> None:
    case, overrides = _CASES["pedestrian_dart_out"]
    run.check_batch_configs(
        [run._compose_config(case, [*overrides, f"{override}={value}"])]
    )


def test_a_still_cut_in_npc_is_accepted() -> None:
    """``cut_in``'s relative-speed control limits the NPC's speed to 0."""
    case, overrides = _CASES["cut_in"]
    run.check_batch_configs(
        [run._compose_config(case, [*overrides, "scenario.npc_initial_speed_kmh=0"])]
    )


# ---------------------------------------------------------------------------
# The scenario package template
# ---------------------------------------------------------------------------

_TEMPLATE_SRC = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "scenario_package_template"
    / "src"
)


def _reach_goal_config() -> type:
    sys.path.insert(0, str(_TEMPLATE_SRC))
    try:
        from my_scenario_package.configs import ReachGoalConfig
    finally:
        sys.path.remove(str(_TEMPLATE_SRC))
    return ReachGoalConfig


@pytest.mark.parametrize("value", POSITIVE_BAD)
def test_the_template_refuses_a_timeout_not_positive(value: float) -> None:
    with pytest.raises(
        ValueError, match=_message("timeout_seconds", "finite and positive, not ")
    ):
        _reach_goal_config()(timeout_seconds=value)


@pytest.mark.parametrize("value", [1e-6, 8.0])
def test_the_template_accepts_a_positive_timeout(value: float) -> None:
    assert _reach_goal_config()(timeout_seconds=value).timeout_seconds == value
