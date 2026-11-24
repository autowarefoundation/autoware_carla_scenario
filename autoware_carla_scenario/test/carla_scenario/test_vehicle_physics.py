"""Unit tests for the ego's steering parameters in CARLA's vehicle physics
(``driver.vehicle``, driver/vehicle_physics.py): the config, writing it to an
actor, its probe and its OpenODD knob."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, List

import pytest

from autoware_carla_scenario.driver.base import DriverClientConfig
from autoware_carla_scenario.driver.vehicle_physics import (
    VehicleSteeringConfig,
    apply_vehicle_steering,
)
from autoware_carla_scenario.odd import probes
from autoware_carla_scenario.odd.backend import BUILT_IN_PROBES, DEFAULT_KNOBS
from autoware_carla_scenario.scenario_config import DriverVehicleSpec


class _Physics:
    def __init__(self, wheels: List[Any]) -> None:
        self.wheels = wheels
        self.steering_curve: List[Any] = ["carla's"]


class _Vehicle:
    """A vehicle whose physics control is a copy, as CARLA's: a change shows only
    once it is applied."""

    id = 7

    def __init__(self, wheels: List[Any]) -> None:
        self._physics = _Physics(wheels)
        self.applied: List[_Physics] = []

    def get_physics_control(self) -> _Physics:
        physics = _Physics([dataclasses.replace(w) for w in self._physics.wheels])
        physics.steering_curve = list(self._physics.steering_curve)
        return physics

    def apply_physics_control(self, physics: _Physics) -> None:
        self.applied.append(physics)
        self._physics = physics


@dataclasses.dataclass
class _Wheel:
    max_steer_angle: float
    affected_by_steering: bool


def _car() -> _Vehicle:
    # CARLA 0.10's MKZ: the front wheels steer, the rear ones do not.
    return _Vehicle(
        [
            _Wheel(70.0, True),
            _Wheel(70.0, True),
            _Wheel(0.0, False),
            _Wheel(0.0, False),
        ]
    )


# -- config -----------------------------------------------------------------------


def test_the_default_keeps_the_blueprint() -> None:
    config = VehicleSteeringConfig.from_mapping({})
    assert config == VehicleSteeringConfig()
    assert not config.changes_anything


def test_the_schema_mirrors_the_config() -> None:
    spec = {f.name: f.default for f in dataclasses.fields(DriverVehicleSpec)}
    runtime = {f.name: f.default for f in dataclasses.fields(VehicleSteeringConfig)}
    assert spec == runtime


def test_a_curve_is_read_as_float_pairs() -> None:
    config = VehicleSteeringConfig.from_mapping(
        {"max_steer_angle_deg": 45, "steering_curve": [[0, 1], [120, 0.5]]}
    )
    assert config.max_steer_angle_deg == 45.0
    assert config.steering_curve == ((0.0, 1.0), (120.0, 0.5))
    assert all(isinstance(v, float) for point in config.steering_curve for v in point)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"max_steer": 45}, "^max_steer: Unknown VehicleSteeringConfig"),
        ({"max_steer_angle_deg": 0}, r"^max_steer_angle_deg must be in \(0, 90\]"),
        ({"max_steer_angle_deg": 91}, r"^max_steer_angle_deg must be in \(0, 90\]"),
        ({"max_steer_angle_deg": True}, "^max_steer_angle_deg must be a number"),
        ({"steering_curve": "flat"}, "^steering_curve must be a list of"),
        ({"steering_curve": [[0, 1, 2]]}, r"^steering_curve\[0\] must be a"),
        ({"steering_curve": [0, 1]}, r"^steering_curve\[0\] must be a"),
        ({"steering_curve": [[0, "1"]]}, r"^steering_curve\[0\] must be a number"),
        ({"steering_curve": []}, "^steering_curve must have at least one point"),
        ({"steering_curve": [[0, -1]]}, "^steering_curve's speeds and factors"),
        ({"steering_curve": [[60, 1], [30, 1]]}, "^steering_curve's speeds must"),
    ],
)
def test_the_config_names_what_it_refuses(mapping: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VehicleSteeringConfig.from_mapping(mapping)


def test_a_null_value_keeps_the_blueprint() -> None:
    config = VehicleSteeringConfig.from_mapping(
        {"max_steer_angle_deg": None, "steering_curve": None}
    )
    assert config == VehicleSteeringConfig()


def test_the_driver_client_ignores_the_vehicle_section() -> None:
    DriverClientConfig.from_mapping({"vehicle": {"max_steer_angle_deg": 45}})


def test_run_names_the_vehicle_section() -> None:
    from autoware_carla_scenario.examples.run import _vehicle_config

    assert _vehicle_config({"vehicle": None}) == VehicleSteeringConfig()
    with pytest.raises(ValueError, match="^driver.vehicle must be a mapping"):
        _vehicle_config({"vehicle": 45})
    with pytest.raises(ValueError, match="^driver.vehicle.max_steer_angle_deg must"):
        _vehicle_config({"vehicle": {"max_steer_angle_deg": -1}})


# -- writing it to the actor ------------------------------------------------------


def test_the_angle_goes_on_the_steered_wheels_only() -> None:
    car = _car()
    apply_vehicle_steering(car, VehicleSteeringConfig(max_steer_angle_deg=40.0))  # type: ignore[arg-type]
    assert len(car.applied) == 1
    angles = [w.max_steer_angle for w in car.get_physics_control().wheels]
    assert angles == [40.0, 40.0, 0.0, 0.0]
    # The curve is left as it was.
    assert car.get_physics_control().steering_curve == ["carla's"]


def test_the_curve_replaces_carlas() -> None:
    car = _car()
    apply_vehicle_steering(
        car,  # type: ignore[arg-type]
        VehicleSteeringConfig(steering_curve=((0.0, 1.0), (120.0, 1.0))),
    )
    curve = car.get_physics_control().steering_curve
    assert [(p.x, p.y) for p in curve] == [(0.0, 1.0), (120.0, 1.0)]
    angles = [w.max_steer_angle for w in car.get_physics_control().wheels]
    assert angles == [70.0, 70.0, 0.0, 0.0]


def test_nothing_is_applied_when_nothing_is_set() -> None:
    car = _car()
    apply_vehicle_steering(car, VehicleSteeringConfig())  # type: ignore[arg-type]
    assert car.applied == []


# -- the probe and its knob -------------------------------------------------------


def test_the_probe_reads_the_wheels_angle(monkeypatch: pytest.MonkeyPatch) -> None:
    car = _car()
    monkeypatch.setattr(probes, "_ego", lambda world: car)
    assert probes.ego_max_steer_angle_deg(SimpleNamespace()) == 70.0
    apply_vehicle_steering(car, VehicleSteeringConfig(max_steer_angle_deg=35.0))  # type: ignore[arg-type]
    assert probes.ego_max_steer_angle_deg(SimpleNamespace()) == 35.0


def test_the_probe_reads_nothing_without_a_steered_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probes, "_ego", lambda world: _Vehicle([_Wheel(0.0, False)]))
    assert probes.ego_max_steer_angle_deg(SimpleNamespace()) is None
    monkeypatch.setattr(probes, "_ego", lambda world: None)
    assert probes.ego_max_steer_angle_deg(SimpleNamespace()) is None


def test_the_knob_sets_driver_vehicle() -> None:
    probe, unit = BUILT_IN_PROBES["ego_max_steer_angle_deg"]
    assert unit == "deg"
    knob = DEFAULT_KNOBS[probe]
    assert knob.key == "driver.vehicle.max_steer_angle_deg"
    # Every value it draws is one the config takes.
    assert knob.range is not None
    low, high = knob.range
    VehicleSteeringConfig(max_steer_angle_deg=low)
    VehicleSteeringConfig(max_steer_angle_deg=high)
