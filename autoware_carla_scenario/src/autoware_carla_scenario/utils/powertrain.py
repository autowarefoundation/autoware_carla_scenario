"""The longitudinal response of a CARLA 0.10 vehicle, worked out from its physics control.

CARLA 0.10 drives its vehicles with Unreal Engine 5.5's Chaos wheeled-vehicle model
(``ChaosVehicles`` in CARLA's UE fork).  What a pedal does there follows from the
vehicle's :class:`carla.VehiclePhysicsControl` alone, read off that source:

* **Engine.**  Torque is ``throttle**2`` times the torque curve at the engine's rpm
  (``FSimpleEngineSim``, fed the squared throttle by
  ``UChaosWheeledVehicleSimulation::ApplyInput``).  The curve is sampled at 21 even
  steps over ``[0, max_rpm]``, normalised by its largest key and scaled to
  ``max_torque`` (``FillEngineSetup``).  In gear, the rpm is the wheels' times the gear
  ratio, held within ``[idle_rpm, max_rpm]``.
* **Gearbox.**  Torque reaches the driven wheels times the gear's ratio, the final
  ratio and the transmission efficiency, shared evenly between them.  The automatic
  box changes up at ``change_up_rpm`` and down at ``change_down_rpm``, passing
  ``gear_change_time`` in neutral; a throttle on a car in neutral puts it straight into
  first.
* **Engine braking.**  Only with the throttle at exactly zero: each driven wheel is
  braked with ``rpm * brake_effect`` N·m.
* **Brakes.**  ``brake * max_brake_torque`` on every wheel.
* **Wheels.**  A torque pushes the car with ``torque / wheel_radius``; a wheel braked
  harder than it is driven only brakes.  Traction control holds a driven wheel's force
  to 0.98 of its grip -- its load times the road's friction times the wheel's
  ``friction_force_multiplier``.  There is no rolling resistance.
* **Air.**  ``0.5 * 1.225 * drag_area * drag_coefficient * v**2``.

Measured against a Lincoln MKZ on Town10 the model holds the speed a second after a
pedal step to under 0.1 m/s (median), from a crawl to 50 km/h and from full brake to
0.7 throttle.  It knows the rates the pedals are slewed at (6 per second up, 10
down) but answers for a pedal that has settled.

Other vehicle physics (CARLA's Chrono or CarSim integrations) answer differently; whoever
puts a vehicle on them knows it, and this model is not for that vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np

__all__ = ["CREEP_THROTTLE", "ChaosPowertrain", "DEFAULT_ROAD_FRICTION"]

#: Air density Chaos uses, kg/m³ (``RealWorldConsts::AirDensity``).
_AIR_DENSITY: float = 1.225

#: Traction control's share of the grip it lets a wheel use.
_TRACTION_SCALING: float = 0.98

#: The friction of Town10's road surface, fitted to an MKZ's full-throttle pull-away.
#: The road's physical material is not readable through CARLA's API.
DEFAULT_ROAD_FRICTION: float = 0.41

#: A throttle too small to push the car (its square is 1e-4) that still takes engine
#: braking off and keeps the gearbox in gear: Chaos engine-brakes only at a throttle
#: of exactly zero, and only a throttle above 1e-4 engages first from neutral.
CREEP_THROTTLE: float = 0.01

#: Chaos samples the torque curve at this many even steps over ``[0, max_rpm]``.
_TORQUE_SAMPLES: int = 20

#: Standard gravity, m/s².
_GRAVITY: float = 9.81

#: Which wheels a ``differential_type`` drives, by wheel index (front pair first):
#: CARLA passes UE's ``EVehicleDifferential`` through as is.
_DRIVEN_WHEELS = {0: "all", 1: "all", 2: "front", 3: "rear"}


def _rich_curve(keys: Sequence[Tuple[float, float]], rpm: float) -> float:
    """UE's ``FRichCurve::Eval`` over linear keys, in the order they are stored.

    It searches the keys as if sorted by time; CARLA's own MKZ curve is not, and the
    engine sees whatever this search finds, so the search is kept as it is.
    """
    if not keys:
        return 0.0
    if len(keys) < 2 or rpm < keys[0][0]:
        return keys[0][1]
    if rpm > keys[-1][0]:
        return keys[-1][1]
    first, count = 1, len(keys) - 2
    while count > 0:
        step = count // 2
        middle = first + step
        if rpm >= keys[middle][0]:
            first = middle + 1
            count -= step + 1
        else:
            count = step
    (t0, v0), (t1, v1) = keys[first - 1], keys[first]
    if t1 - t0 <= 0.0:
        return v0
    return v0 + (v1 - v0) * (rpm - t0) / (t1 - t0)


@dataclass(frozen=True)
class ChaosPowertrain:
    """A Chaos vehicle's pedals as forces on the car.  Build with :meth:`from_physics_control`.

    Attributes:
        mass_kg: The vehicle's mass.
        wheel_radius_m: Its wheels' radius.
        wheel_brake_torques_nm: Each wheel's brake torque at full brake.
        driven_wheels: Indices of the wheels the engine drives.
        torque_graph: The engine's torque curve as Chaos samples it, normalised.
        max_torque_nm: Engine torque at the curve's peak.
        max_rpm: The engine's top rpm, where its torque is cut.
        idle_rpm: The engine's lowest rpm.
        engine_brake_effect: N·m of engine braking per rpm, on each driven wheel.
        gear_ratios: Each forward gear's ratio times the final ratio.
        transmission_efficiency: Share of the engine torque that reaches the wheels.
        change_up_rpm: The automatic box changes up at this rpm.
        drag_n_per_mps2: Air drag per squared m/s.
        wheel_grip_n: A driven wheel's force under traction control.
    """

    mass_kg: float
    wheel_radius_m: float
    wheel_brake_torques_nm: Tuple[float, ...]
    driven_wheels: Tuple[int, ...]
    torque_graph: Tuple[float, ...]
    max_torque_nm: float
    max_rpm: float
    idle_rpm: float
    engine_brake_effect: float
    gear_ratios: Tuple[float, ...]
    transmission_efficiency: float
    change_up_rpm: float
    drag_n_per_mps2: float
    wheel_grip_n: float

    @classmethod
    def from_physics_control(
        cls, physics: Any, road_friction: float = DEFAULT_ROAD_FRICTION
    ) -> "ChaosPowertrain":
        """Read the model off a vehicle's ``carla.VehiclePhysicsControl``.

        Args:
            physics: What ``vehicle.get_physics_control()`` returns.
            road_friction: The friction of the road's surface material.

        Raises:
            ValueError: If the physics control has no wheels, gears or torque curve.
        """
        wheels = list(physics.wheels)
        keys = [(float(k.x), float(k.y)) for k in physics.torque_curve]
        gears = [float(r) for r in physics.forward_gear_ratios]
        if not wheels or not gears or not keys:
            raise ValueError(
                "a Chaos vehicle needs wheels, forward gears and a torque curve"
            )
        max_rpm = float(physics.max_rpm)
        peak = max(value for _, value in keys)
        graph = []
        # UE steps a float by max_rpm / 20 up to max_rpm; mirror it, not a linspace.
        step = np.float32(max_rpm / _TORQUE_SAMPLES)
        x = np.float32(0.0)
        while x <= max_rpm:
            graph.append(_rich_curve(keys, float(x)) / peak if peak else 0.0)
            x = np.float32(x + step)
        driven = _DRIVEN_WHEELS.get(int(physics.differential_type), "all")
        half = len(wheels) // 2
        indices = {
            "all": tuple(range(len(wheels))),
            "front": tuple(range(half)),
            "rear": tuple(range(half, len(wheels))),
        }[driven]
        radius = float(wheels[0].wheel_radius) / 100.0  # CARLA reports centimetres
        mass = float(physics.mass)
        load = mass * _GRAVITY / len(wheels)
        grip = (
            _TRACTION_SCALING
            * road_friction
            * float(wheels[0].friction_force_multiplier)
            * load
        )
        final = float(physics.final_ratio)
        return cls(
            mass_kg=mass,
            wheel_radius_m=radius,
            wheel_brake_torques_nm=tuple(float(w.max_brake_torque) for w in wheels),
            driven_wheels=indices,
            torque_graph=tuple(graph),
            max_torque_nm=float(physics.max_torque),
            max_rpm=max_rpm,
            idle_rpm=float(physics.idle_rpm),
            engine_brake_effect=float(physics.brake_effect),
            gear_ratios=tuple(g * final for g in gears),
            transmission_efficiency=float(physics.transmission_efficiency),
            change_up_rpm=float(physics.change_up_rpm),
            # drag_area is in cm², as CARLA hands UE's DragArea through.
            drag_n_per_mps2=0.5
            * _AIR_DENSITY
            * float(physics.drag_area)
            / 1e4
            * float(physics.drag_coefficient),
            wheel_grip_n=grip,
        )

    # -- the engine and gearbox ----------------------------------------------------

    def engine_torque(self, rpm: float) -> float:
        """Full-throttle engine torque at *rpm*, N·m (``FSimpleEngineSim::GetTorqueFromRPM``)."""
        if abs(rpm - self.max_rpm) < 1.0 or self.max_rpm <= 0.0:
            return 0.0
        rpm = min(max(rpm, self.idle_rpm), self.max_rpm)
        graph = self.torque_graph
        step = self.max_rpm / (len(graph) - 1)
        index = int(rpm / step)
        if index >= len(graph) - 1:
            value = graph[-1]
        else:
            value = (
                graph[index]
                + (graph[index + 1] - graph[index]) * (rpm - index * step) / step
            )
        return value * self.max_torque_nm

    def gear_for(self, speed_mps: float) -> int:
        """The gear the automatic box is in at *speed_mps* after pulling away from rest."""
        for gear in range(1, len(self.gear_ratios) + 1):
            if (
                self._wheel_rpm(speed_mps) * self.gear_ratios[gear - 1]
                < self.change_up_rpm
            ):
                return gear
        return len(self.gear_ratios)

    def _wheel_rpm(self, speed_mps: float) -> float:
        return abs(speed_mps) / self.wheel_radius_m * 60.0 / (2.0 * math.pi)

    def _ratio(self, gear: int) -> float:
        """The overall ratio in *gear*; neutral (0) counts as first, which a throttle selects."""
        gear = min(max(gear, 1), len(self.gear_ratios))
        return self.gear_ratios[gear - 1]

    def _engine_rpm(self, speed_mps: float, ratio: float) -> float:
        return min(max(self._wheel_rpm(speed_mps) * ratio, self.idle_rpm), self.max_rpm)

    def _drive_force(self, throttle: float, speed_mps: float, ratio: float) -> float:
        """Force the driven wheels push with at *throttle*, grip-limited, N."""
        if self._wheel_rpm(speed_mps) * ratio > self.max_rpm:
            return 0.0  # past the engine's top rpm Chaos cuts the drive
        torque = throttle**2 * self.engine_torque(self._engine_rpm(speed_mps, ratio))
        per_wheel = (
            torque * ratio * self.transmission_efficiency / len(self.driven_wheels)
        )
        return len(self.driven_wheels) * min(
            per_wheel / self.wheel_radius_m, self.wheel_grip_n
        )

    def _full_brake_force(self) -> float:
        return sum(self.wheel_brake_torques_nm) / self.wheel_radius_m

    # -- forward and inverse ---------------------------------------------------------

    def acceleration(
        self, speed_mps: float, throttle: float, brake: float, gear: int
    ) -> float:
        """Settled acceleration at *speed_mps* in *gear* with these pedals, m/s².

        Forward travel only; a standing car is taken to stay put under the brakes.
        """
        ratio = self._ratio(gear)
        drag = self.drag_n_per_mps2 * speed_mps * abs(speed_mps)
        if speed_mps <= 0.0:
            return (
                max(
                    self._drive_force(throttle, 0.0, ratio)
                    - brake * self._full_brake_force(),
                    0.0,
                )
                / self.mass_kg
            )
        engine_brake = (
            self._engine_rpm(speed_mps, ratio) * self.engine_brake_effect
            if throttle < 1e-8
            else 0.0
        )
        drive_per_wheel = self._drive_force(throttle, speed_mps, ratio) / len(
            self.driven_wheels
        )
        force = 0.0
        for index, brake_torque in enumerate(self.wheel_brake_torques_nm):
            driven = index in self.driven_wheels
            braking = brake * brake_torque + (engine_brake if driven else 0.0)
            pushing = drive_per_wheel * self.wheel_radius_m if driven else 0.0
            force += (
                -braking / self.wheel_radius_m
                if braking > pushing
                else pushing / self.wheel_radius_m
            )
        return (force - drag) / self.mass_kg

    def pedals(
        self, acceleration: float, speed_mps: float, gear: int
    ) -> Tuple[float, float]:
        """``(throttle, brake)`` that settle at *acceleration* at *speed_mps* in *gear*.

        Slowing down is left to the brakes: the throttle stays at
        :data:`CREEP_THROTTLE`, which pushes nothing but keeps the gear in and the
        engine braking -- different in every gear -- off.  Beyond what the engine or the
        brakes can do, the pedal saturates.
        """
        speed = max(speed_mps, 0.0)
        force = acceleration * self.mass_kg + self.drag_n_per_mps2 * speed * speed
        if force <= 0.0:
            return CREEP_THROTTLE, min(-force / self._full_brake_force(), 1.0)
        ratio = self._ratio(gear)
        full = self._drive_force(1.0, speed, ratio)
        if full <= force:
            return 1.0, 0.0
        # Below the grip limit the drive force goes with the throttle's square.
        ungripped = (
            self.engine_torque(self._engine_rpm(speed, ratio))
            * ratio
            * self.transmission_efficiency
        )
        throttle = math.sqrt(force * self.wheel_radius_m / ungripped)
        return max(min(throttle, 1.0), CREEP_THROTTLE), 0.0

    # -- tables ----------------------------------------------------------------------

    def accel_map(
        self, speeds_mps: Sequence[float], throttles: Sequence[float]
    ) -> np.ndarray:
        """Autoware's ``accel_map.csv`` body: acceleration per (throttle row, speed column).

        Each speed is taken in the gear the automatic box reaches pulling away
        (:meth:`gear_for`).
        """
        return np.array(
            [
                [self.acceleration(v, t, 0.0, self.gear_for(v)) for v in speeds_mps]
                for t in throttles
            ]
        )

    def brake_map(
        self, speeds_mps: Sequence[float], brakes: Sequence[float]
    ) -> np.ndarray:
        """Autoware's ``brake_map.csv`` body: acceleration per (brake row, speed column).

        Taken with the throttle at zero, engine braking and all, as Autoware's
        vehicle interface would send it.
        """
        return np.array(
            [
                [self.acceleration(v, 0.0, b, self.gear_for(v)) for v in speeds_mps]
                for b in brakes
            ]
        )
