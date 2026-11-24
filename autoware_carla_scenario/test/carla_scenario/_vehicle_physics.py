"""A Lincoln MKZ's physics control as CARLA 0.10 reports it, for tests that need a powertrain."""

from __future__ import annotations

from types import SimpleNamespace

from autoware_carla_scenario.utils.powertrain import ChaosPowertrain


def mkz_physics() -> SimpleNamespace:
    """``vehicle.lincoln.mkz``'s ``get_physics_control()`` on CARLA 0.10.0, as read off one.

    The torque curve's keys are in the order CARLA hands them back, unsorted.
    """
    curve = [
        (0.0, 500.0),
        (5000.0, 500.0),
        (1000.0, 347.0),
        (1500.0, 523.0),
        (2000.0, 606.0),
        (2800.0, 670.0),
        (4300.0, 677.0),
        (5300.0, 607.0),
        (6500.0, 466.0),
    ]
    wheel = dict(
        wheel_radius=35.5, max_brake_torque=1000.0, friction_force_multiplier=3.5
    )
    return SimpleNamespace(
        torque_curve=[SimpleNamespace(x=x, y=y) for x, y in curve],
        max_torque=550.0,
        max_rpm=6500.0,
        idle_rpm=750.0,
        brake_effect=0.05,
        differential_type=2,
        mass=1696.0,
        forward_gear_ratios=[4.0, 2.5, 1.912, 1.612, 1.0, 0.746],
        final_ratio=3.21,
        transmission_efficiency=0.9,
        change_up_rpm=4000.0,
        change_down_rpm=1500.0,
        gear_change_time=0.1,
        drag_area=28080.0,
        drag_coefficient=0.3,
        wheels=[SimpleNamespace(**wheel) for _ in range(4)],
    )


def mkz_powertrain() -> ChaosPowertrain:
    """The MKZ's powertrain model."""
    return ChaosPowertrain.from_physics_control(mkz_physics())


class Carla010Curves:
    """A vehicle's steering and torque curves as a CARLA 0.10 server keeps them:
    a read hands back the stored keys, leading keys included
    (:data:`~autoware_carla_scenario.utils.vehicles.CARLA_STEERING_CURVE_LEAD`,
    ``CARLA_TORQUE_CURVE_LEAD``), and an apply stores those leading keys in front
    of whatever curve it is given -- so a read applied back unchanged has them
    twice.  Measured on a CARLA 0.10 server with every vehicle blueprint."""

    #: The MKZ's own keys behind the leading ones, as its server hands them back.
    MKZ_STEERING = [(0.0, 1.0), (20.0, 0.9), (60.0, 0.8), (120.0, 0.7)]
    MKZ_TORQUE = [(1000.0, 347.0), (1500.0, 523.0), (2000.0, 606.0), (6500.0, 466.0)]

    def __init__(self) -> None:
        from autoware_carla_scenario.utils.vehicles import (
            CARLA_STEERING_CURVE_LEAD,
            CARLA_TORQUE_CURVE_LEAD,
        )

        self._steering_lead = list(CARLA_STEERING_CURVE_LEAD)
        self._torque_lead = list(CARLA_TORQUE_CURVE_LEAD)
        self.steering = self._steering_lead + self.MKZ_STEERING
        self.torque = self._torque_lead + self.MKZ_TORQUE

    def read(self) -> dict:
        """The curves of a ``get_physics_control()``."""
        return {
            "steering_curve": [SimpleNamespace(x=x, y=y) for x, y in self.steering],
            "torque_curve": [SimpleNamespace(x=x, y=y) for x, y in self.torque],
        }

    def write(self, physics: SimpleNamespace) -> None:
        """Store *physics*'s curves as an ``apply_physics_control()`` does."""
        self.steering = self._steering_lead + [
            (float(p.x), float(p.y)) for p in physics.steering_curve
        ]
        self.torque = self._torque_lead + [
            (float(p.x), float(p.y)) for p in physics.torque_curve
        ]
