"""Unit tests for the Chaos powertrain model.

The expected values come from CARLA 0.10.0 runs with a Lincoln MKZ on Town10, carried
to a speed and then left on a pedal (``vehicle.lincoln.mkz``, 20 Hz synchronous).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from autoware_carla_scenario.utils.powertrain import (
    CREEP_THROTTLE,
    ChaosPowertrain,
)

from ._vehicle_physics import mkz_physics, mkz_powertrain

_KMH30 = 8.33


def test_the_torque_curve_is_read_as_chaos_samples_it() -> None:
    """21 samples over [0, max_rpm], each normalised by the curve's largest key;
    CARLA's MKZ curve is stored unsorted and searched as if it were not."""
    graph = mkz_powertrain().torque_graph
    assert len(graph) == 21
    assert graph[0] == pytest.approx(500.0 / 677.0)
    assert max(graph) == pytest.approx(0.999, abs=1e-3)


def test_released_pedals_engine_brake_in_gear() -> None:
    """Measured: 8.33 -> 7.83 m/s over a second in first, throttle and brake at zero."""
    assert mkz_powertrain().acceleration(_KMH30, 0.0, 0.0, 1) == pytest.approx(-0.50, abs=0.03)


def test_any_throttle_takes_the_engine_braking_off() -> None:
    """Chaos engine-brakes only at a throttle of exactly zero; the creep is left to drag."""
    powertrain = mkz_powertrain()
    creep = powertrain.acceleration(_KMH30, CREEP_THROTTLE, 0.0, 1)
    assert -0.05 < creep < 0.0


@pytest.mark.parametrize(
    ("throttle", "measured"), [(0.2, 0.37), (0.4, 1.60), (0.7, 4.6)]
)
def test_throttle_goes_with_its_square(throttle: float, measured: float) -> None:
    """At 30 km/h in first, settled (the measured 0.7 still slewing in)."""
    got = mkz_powertrain().acceleration(_KMH30, throttle, 0.0, 1)
    assert got == pytest.approx(measured, rel=0.15)


def test_brakes_are_linear_in_the_pedal() -> None:
    powertrain = mkz_powertrain()
    a1 = powertrain.acceleration(_KMH30, CREEP_THROTTLE, 0.2, 1)
    a2 = powertrain.acceleration(_KMH30, CREEP_THROTTLE, 0.6, 1)
    assert a1 == pytest.approx(-1.35, abs=0.05)  # 4 x 1000 N·m x 0.2 / 0.355 m / 1696 kg, plus drag
    assert a2 - a1 == pytest.approx(2.0 * (a1 + 0.02), rel=0.02)


def test_traction_control_caps_a_full_throttle_pull_away() -> None:
    """Measured: an MKZ pulls at about 6.9 m/s² flat out in first, not the engine's 10."""
    assert mkz_powertrain().acceleration(_KMH30, 1.0, 0.0, 1) == pytest.approx(6.9, abs=0.3)


def test_a_higher_gear_pulls_less() -> None:
    powertrain = mkz_powertrain()
    first = powertrain.acceleration(_KMH30, 0.4, 0.0, 1)
    second = powertrain.acceleration(_KMH30, 0.4, 0.0, 2)
    assert 0.0 < second < first


def test_neutral_counts_as_first() -> None:
    """A throttle on a car in neutral puts it straight into first."""
    powertrain = mkz_powertrain()
    assert powertrain.pedals(1.0, _KMH30, 0) == powertrain.pedals(1.0, _KMH30, 1)


@pytest.mark.parametrize("acceleration", [-4.0, -1.0, -0.01, 0.0, 0.3, 1.5, 4.0])
@pytest.mark.parametrize("gear", [1, 2])
def test_pedals_settle_at_the_asked_acceleration(acceleration: float, gear: int) -> None:
    powertrain = mkz_powertrain()
    throttle, brake = powertrain.pedals(acceleration, _KMH30, gear)
    assert powertrain.acceleration(_KMH30, throttle, brake, gear) == pytest.approx(
        acceleration, abs=1e-6
    )


def test_slowing_down_is_left_to_the_brakes() -> None:
    """Below coasting, the throttle stays at the creep: in gear, engine braking off."""
    throttle, brake = mkz_powertrain().pedals(-0.3, _KMH30, 1)
    assert throttle == CREEP_THROTTLE
    assert 0.0 < brake < 0.1


def test_beyond_the_powertrain_the_pedals_saturate() -> None:
    powertrain = mkz_powertrain()
    assert powertrain.pedals(20.0, _KMH30, 1) == (1.0, 0.0)
    assert powertrain.pedals(-20.0, _KMH30, 1) == (CREEP_THROTTLE, 1.0)


def test_holding_a_speed_takes_a_light_throttle() -> None:
    """Holding 30 km/h is a matter of drag: a few percent of throttle, no brake."""
    throttle, brake = mkz_powertrain().pedals(0.0, _KMH30, 1)
    assert CREEP_THROTTLE < throttle < 0.1
    assert brake == 0.0


def test_the_gear_after_pulling_away() -> None:
    """First up to 4000 rpm in first -- 11.6 m/s -- then second."""
    powertrain = mkz_powertrain()
    assert [powertrain.gear_for(v) for v in (0.0, 8.33, 11.5, 11.7, 18.0)] == [1, 1, 1, 2, 2]


def test_the_physics_control_is_read_live() -> None:
    """A physics control changed through CARLA's API changes the model."""
    physics = mkz_physics()
    heavier = mkz_physics()
    heavier.mass = 2 * physics.mass
    base = ChaosPowertrain.from_physics_control(physics)
    loaded = ChaosPowertrain.from_physics_control(heavier)
    assert loaded.acceleration(_KMH30, 0.3, 0.0, 1) < base.acceleration(_KMH30, 0.3, 0.0, 1)


def test_the_differential_picks_the_driven_wheels() -> None:
    physics = mkz_physics()
    assert ChaosPowertrain.from_physics_control(physics).driven_wheels == (0, 1)
    physics.differential_type = 3
    assert ChaosPowertrain.from_physics_control(physics).driven_wheels == (2, 3)
    physics.differential_type = 1
    assert ChaosPowertrain.from_physics_control(physics).driven_wheels == (0, 1, 2, 3)


def test_autoware_maps_have_its_layout() -> None:
    """Rows per pedal, columns per speed; more pedal, more acceleration (or less)."""
    powertrain = mkz_powertrain()
    speeds = [0.0, 2.78, 5.56, 8.33, 11.11, 13.89]
    accel = powertrain.accel_map(speeds, [0.0, 0.2, 0.5, 1.0])
    brake = powertrain.brake_map(speeds, [0.0, 0.3, 1.0])
    assert accel.shape == (4, 6) and brake.shape == (3, 6)
    assert np.all(np.diff(accel[:, 1:], axis=0) > 0.0)
    assert np.all(np.diff(brake[:, 1:], axis=0) < 0.0)


def test_a_vehicle_without_a_drivetrain_is_refused() -> None:
    physics = mkz_physics()
    physics.forward_gear_ratios = []
    with pytest.raises(ValueError):
        ChaosPowertrain.from_physics_control(physics)


def test_the_model_is_a_value() -> None:
    assert dataclasses.is_dataclass(mkz_powertrain())
    assert mkz_powertrain() == mkz_powertrain()
