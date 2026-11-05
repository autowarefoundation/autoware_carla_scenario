"""The traffic parameters' probe names, knob keys and limits, pinned (#1680).

The table moved from ``odd.traffic_probes`` into the checked
``odd.traffic_ranges``; what the probes and their knobs are called, and what
they draw and keep a value within, is unchanged.  ``EXPECTED`` is what
``odd.traffic_probes`` built before the move.
"""

from __future__ import annotations

import pytest

from autoware_carla_scenario.odd import traffic_probes, traffic_ranges
from autoware_carla_scenario.odd.traffic_probes import TRAFFIC_KNOBS, TRAFFIC_PROBES
from autoware_carla_scenario.odd.traffic_ranges import (
    PARAMETER_RANGES,
    knob_key,
    knob_limits,
    probe_name,
)

INF = float("inf")

#: (probe name, unit, knob key, knob range, knob limits), in the built order.
EXPECTED: list[tuple[str, str, str, tuple[float, float], tuple[float, float]]] = [
    (
        "vehicle_speed_factor_mean",
        "",
        "traffic.vehicles.speed_factor.mean",
        (0.5, 1.5),
        (0.05, INF),
    ),
    (
        "vehicle_speed_factor_std",
        "",
        "traffic.vehicles.speed_factor.std",
        (0.0, 0.3),
        (0.0, INF),
    ),
    (
        "vehicle_min_gap_m_mean",
        "m",
        "traffic.vehicles.min_gap_m.mean",
        (0.5, 10.0),
        (0.0, INF),
    ),
    (
        "vehicle_min_gap_m_std",
        "m",
        "traffic.vehicles.min_gap_m.std",
        (0.0, 3.0),
        (0.0, INF),
    ),
    (
        "vehicle_ignore_lights_mean",
        "",
        "traffic.vehicles.ignore_lights.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "vehicle_ignore_lights_std",
        "",
        "traffic.vehicles.ignore_lights.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_ignore_signs_mean",
        "",
        "traffic.vehicles.ignore_signs.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "vehicle_ignore_signs_std",
        "",
        "traffic.vehicles.ignore_signs.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_ignore_vehicles_mean",
        "",
        "traffic.vehicles.ignore_vehicles.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "vehicle_ignore_vehicles_std",
        "",
        "traffic.vehicles.ignore_vehicles.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_ignore_walkers_mean",
        "",
        "traffic.vehicles.ignore_walkers.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "vehicle_ignore_walkers_std",
        "",
        "traffic.vehicles.ignore_walkers.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_random_lane_changes_mean",
        "",
        "traffic.vehicles.random_lane_changes.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "vehicle_random_lane_changes_std",
        "",
        "traffic.vehicles.random_lane_changes.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_accel_mean",
        "m/s^2",
        "traffic.vehicles.accel.mean",
        (0.5, 5.0),
        (0.05, INF),
    ),
    (
        "vehicle_accel_std",
        "m/s^2",
        "traffic.vehicles.accel.std",
        (0.0, 1.5),
        (0.0, INF),
    ),
    (
        "vehicle_decel_mean",
        "m/s^2",
        "traffic.vehicles.decel.mean",
        (1.0, 9.0),
        (0.05, INF),
    ),
    (
        "vehicle_decel_std",
        "m/s^2",
        "traffic.vehicles.decel.std",
        (0.0, 2.0),
        (0.0, INF),
    ),
    (
        "vehicle_reaction_time_s_mean",
        "s",
        "traffic.vehicles.reaction_time_s.mean",
        (0.3, 3.0),
        (0.0, INF),
    ),
    (
        "vehicle_reaction_time_s_std",
        "s",
        "traffic.vehicles.reaction_time_s.std",
        (0.0, 1.0),
        (0.0, INF),
    ),
    (
        "vehicle_imperfection_mean",
        "",
        "traffic.vehicles.imperfection.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "vehicle_imperfection_std",
        "",
        "traffic.vehicles.imperfection.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_max_lateral_speed_mps_mean",
        "m/s",
        "traffic.vehicles.max_lateral_speed_mps.mean",
        (0.2, 2.0),
        (0.05, INF),
    ),
    (
        "vehicle_max_lateral_speed_mps_std",
        "m/s",
        "traffic.vehicles.max_lateral_speed_mps.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "vehicle_max_lateral_accel_mean",
        "m/s^2",
        "traffic.vehicles.max_lateral_accel.mean",
        (0.2, 3.0),
        (0.05, INF),
    ),
    (
        "vehicle_max_lateral_accel_std",
        "m/s^2",
        "traffic.vehicles.max_lateral_accel.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "pedestrian_walking_speed_mps_mean",
        "m/s",
        "traffic.pedestrians.walking_speed_mps.mean",
        (0.5, 2.5),
        (0.05, INF),
    ),
    (
        "pedestrian_walking_speed_mps_std",
        "m/s",
        "traffic.pedestrians.walking_speed_mps.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "pedestrian_min_gap_m_mean",
        "m",
        "traffic.pedestrians.min_gap_m.mean",
        (0.1, 2.0),
        (0.0, INF),
    ),
    (
        "pedestrian_min_gap_m_std",
        "m",
        "traffic.pedestrians.min_gap_m.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "pedestrian_red_light_violation_s_mean",
        "s",
        "traffic.pedestrians.red_light_violation_s.mean",
        (0.0, 60.0),
        (0.0, INF),
    ),
    (
        "pedestrian_red_light_violation_s_std",
        "s",
        "traffic.pedestrians.red_light_violation_s.std",
        (0.0, 30.0),
        (0.0, INF),
    ),
    (
        "bicycle_max_speed_mps_mean",
        "m/s",
        "traffic.bicycles.max_speed_mps.mean",
        (2.0, 8.0),
        (0.5, INF),
    ),
    (
        "bicycle_max_speed_mps_std",
        "m/s",
        "traffic.bicycles.max_speed_mps.std",
        (0.0, 1.5),
        (0.0, INF),
    ),
    (
        "bicycle_accel_mean",
        "m/s^2",
        "traffic.bicycles.accel.mean",
        (0.5, 2.0),
        (0.05, INF),
    ),
    (
        "bicycle_accel_std",
        "m/s^2",
        "traffic.bicycles.accel.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "bicycle_decel_mean",
        "m/s^2",
        "traffic.bicycles.decel.mean",
        (1.0, 4.0),
        (0.05, INF),
    ),
    (
        "bicycle_decel_std",
        "m/s^2",
        "traffic.bicycles.decel.std",
        (0.0, 1.0),
        (0.0, INF),
    ),
    (
        "bicycle_min_gap_m_mean",
        "m",
        "traffic.bicycles.min_gap_m.mean",
        (0.2, 2.0),
        (0.0, INF),
    ),
    (
        "bicycle_min_gap_m_std",
        "m",
        "traffic.bicycles.min_gap_m.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "bicycle_reaction_time_s_mean",
        "s",
        "traffic.bicycles.reaction_time_s.mean",
        (0.3, 2.0),
        (0.0, INF),
    ),
    (
        "bicycle_reaction_time_s_std",
        "s",
        "traffic.bicycles.reaction_time_s.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "bicycle_imperfection_mean",
        "",
        "traffic.bicycles.imperfection.mean",
        (0.0, 1.0),
        (0.0, 1.0),
    ),
    (
        "bicycle_imperfection_std",
        "",
        "traffic.bicycles.imperfection.std",
        (0.0, 0.3),
        (0.0, INF),
    ),
    (
        "bicycle_max_lateral_speed_mps_mean",
        "m/s",
        "traffic.bicycles.max_lateral_speed_mps.mean",
        (0.2, 2.0),
        (0.05, INF),
    ),
    (
        "bicycle_max_lateral_speed_mps_std",
        "m/s",
        "traffic.bicycles.max_lateral_speed_mps.std",
        (0.0, 0.5),
        (0.0, INF),
    ),
    (
        "bicycle_red_light_violation_s_mean",
        "s",
        "traffic.bicycles.red_light_violation_s.mean",
        (0.0, 30.0),
        (0.0, INF),
    ),
    (
        "bicycle_red_light_violation_s_std",
        "s",
        "traffic.bicycles.red_light_violation_s.std",
        (0.0, 10.0),
        (0.0, INF),
    ),
]


def test_the_probes_and_knobs_are_unchanged() -> None:
    built = []
    for name, (probe, unit) in TRAFFIC_PROBES.items():
        knob = TRAFFIC_KNOBS[probe]
        value_range, limits = knob.range, knob.limits
        assert value_range is not None and limits is not None, name
        built.append((name, unit, knob.key, tuple(value_range), tuple(limits)))
    assert built == EXPECTED
    assert len(TRAFFIC_KNOBS) == len(EXPECTED)


def test_the_helpers_give_the_pinned_names_keys_and_limits() -> None:
    expected = iter(EXPECTED)
    for (space, field), (unit, mean_range, std_range) in PARAMETER_RANGES.items():
        for statistic, value_range in (("mean", mean_range), ("std", std_range)):
            name, unit_, key, range_, limits = next(expected)
            assert probe_name(space, field, statistic) == name
            assert knob_key(space, field, statistic) == key
            assert knob_limits(space, field, statistic) == limits
            assert (unit, value_range) == (unit_, range_)
    assert next(expected, None) is None


def test_every_mean_range_lies_within_its_limits() -> None:
    for (space, field), (_, (low, high), _) in PARAMETER_RANGES.items():
        lower, upper = knob_limits(space, field, "mean")
        assert lower <= low <= high <= upper, (space, field)


def test_traffic_probes_re_exports_the_moved_names() -> None:
    assert traffic_probes.PARAMETER_RANGES is traffic_ranges.PARAMETER_RANGES
    assert traffic_probes._PREFIX is traffic_ranges._PREFIX


@pytest.mark.parametrize("space", ["vehicle", "", "trucks"])
def test_an_unknown_space_is_a_key_error(space: str) -> None:
    with pytest.raises(KeyError):
        knob_limits(space, "accel", "mean")
    with pytest.raises(KeyError):
        probe_name(space, "accel", "mean")
