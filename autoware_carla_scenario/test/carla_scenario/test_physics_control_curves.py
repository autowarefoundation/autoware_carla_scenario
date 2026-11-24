"""utils.vehicles.apply_physics_control: a physics control read off a CARLA 0.10
vehicle and applied back leaves its steering and torque curves as they were."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from autoware_carla_scenario.utils.vehicles import (
    CARLA_STEERING_CURVE_LEAD,
    CARLA_TORQUE_CURVE_LEAD,
    apply_physics_control,
)

from ._vehicle_physics import Carla010Curves


class _Vehicle:
    def __init__(self) -> None:
        self.curves = Carla010Curves()
        self.applied: List[Any] = []

    def get_physics_control(self) -> SimpleNamespace:
        return SimpleNamespace(**self.curves.read())

    def apply_physics_control(self, physics: SimpleNamespace) -> None:
        self.applied.append(physics)
        self.curves.write(physics)


def _points(pairs: Any) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y) for x, y in pairs]


def test_the_fake_server_doubles_the_keys_of_a_plain_round_trip() -> None:
    # What the helper is for: CARLA's own apply, on a curve as read.
    vehicle = _Vehicle()
    vehicle.apply_physics_control(vehicle.get_physics_control())
    lead = list(CARLA_STEERING_CURVE_LEAD)
    assert vehicle.curves.steering[: 2 * len(lead)] == lead + lead


def test_a_round_trip_leaves_the_curves_as_they_were() -> None:
    vehicle = _Vehicle()
    before = (list(vehicle.curves.steering), list(vehicle.curves.torque))
    for _ in range(3):
        apply_physics_control(vehicle, vehicle.get_physics_control())  # type: ignore[arg-type]
    assert (vehicle.curves.steering, vehicle.curves.torque) == before
    assert len(vehicle.applied) == 3


def test_a_curve_the_caller_set_is_applied_as_it_is() -> None:
    vehicle = _Vehicle()
    physics = vehicle.get_physics_control()
    physics.steering_curve = _points([(0.0, 1.0), (120.0, 1.0)])
    apply_physics_control(vehicle, physics)  # type: ignore[arg-type]
    assert vehicle.curves.steering == list(CARLA_STEERING_CURVE_LEAD) + [
        (0.0, 1.0),
        (120.0, 1.0),
    ]
    # The torque curve, as read, went back without its leading keys.
    assert vehicle.curves.torque == (
        list(CARLA_TORQUE_CURVE_LEAD) + Carla010Curves.MKZ_TORQUE
    )


def test_only_the_exact_leading_keys_are_taken_off() -> None:
    applied: List[Any] = []
    vehicle = SimpleNamespace(apply_physics_control=applied.append)
    # A server that adds nothing: its curves start elsewhere, and stay whole.
    other = _points([(0.0, 1.0), (20.0, 0.9), (60.0, 0.8)])
    # The leading keys alone are not taken off: nothing would be left.
    lead_only = _points(CARLA_TORQUE_CURVE_LEAD)
    physics = SimpleNamespace(steering_curve=other, torque_curve=lead_only)
    apply_physics_control(vehicle, physics)  # type: ignore[arg-type]
    (sent,) = applied
    assert [(p.x, p.y) for p in sent.steering_curve] == [
        (0.0, 1.0),
        (20.0, 0.9),
        (60.0, 0.8),
    ]
    assert [(p.x, p.y) for p in sent.torque_curve] == list(CARLA_TORQUE_CURVE_LEAD)
