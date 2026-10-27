"""The ego's state as named signals, the values a device's message fields are filled from.

:class:`EgoState` is what the bridge reads of the ego on each tick, in CARLA's
terms (Unreal's left-handed world: x forward/east, y right/south, z up;
metres, degrees).  :func:`ego_signals` turns it into :data:`SIGNALS`: speeds
and accelerations in the vehicle frame (x forward, y left, z up, as ROS and
vehicle buses have them), the geodetic position and NED velocities a GNSS
receiver reports, and the driver's controls.  Nothing here imports CARLA, so
a configuration can be checked against the signals where it is not installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["SIGNALS", "EgoState", "ego_signals"]

Vector = Tuple[float, float, float]

#: Every signal :func:`ego_signals` gives, with its unit and meaning.
SIGNALS: Dict[str, str] = {
    "time": "s; simulation time (the world's elapsed seconds)",
    "frame": "the world's frame number",
    "x": "m; world x (CARLA)",
    "y": "m; world y (CARLA: right-handed y negated)",
    "z": "m; world z",
    "roll": "deg; CARLA rotation",
    "pitch": "deg; CARLA rotation",
    "yaw": "deg; CARLA rotation (clockwise from world x, seen from above)",
    "heading": "deg; clockwise from north, [0, 360)",
    "latitude": "deg; WGS84, from the map's geo-reference",
    "longitude": "deg",
    "altitude": "m",
    "latitude_rad": "rad",
    "longitude_rad": "rad",
    "velocity_north": "m/s",
    "velocity_east": "m/s",
    "velocity_up": "m/s",
    "course": "deg; course over ground, clockwise from north, [0, 360)",
    "speed": "m/s; magnitude of the velocity",
    "speed_kmh": "km/h",
    "longitudinal_velocity": "m/s; vehicle frame, forward",
    "lateral_velocity": "m/s; vehicle frame, left",
    "vertical_velocity": "m/s; vehicle frame, up",
    "longitudinal_acceleration": "m/s^2; vehicle frame, forward",
    "lateral_acceleration": "m/s^2; vehicle frame, left",
    "vertical_acceleration": "m/s^2; vehicle frame, up",
    "roll_rate": "deg/s; about the vehicle's x, right-handed",
    "pitch_rate": "deg/s; about the vehicle's y (left), right-handed",
    "yaw_rate": "deg/s; about the vehicle's z, counter-clockwise positive",
    "yaw_rate_rad": "rad/s",
    "throttle": "[0, 1]",
    "brake": "[0, 1]",
    "steer": "[-1, 1]; positive to the right (CARLA)",
    "hand_brake": "0 or 1",
    "reverse": "0 or 1",
    "gear": "CARLA's gear number",
}


@dataclass(frozen=True)
class EgoState:
    """What the bridge reads of the ego on one tick, in CARLA's terms.

    Attributes:
        time: The world's elapsed seconds.
        frame: The world's frame.
        location: World location, metres.
        rotation: ``(pitch, yaw, roll)``, degrees.
        velocity: World velocity, m/s.
        acceleration: World acceleration, m/s^2.
        angular_velocity: World angular velocity, deg/s (CARLA's,
            left-handed: positive yaw turns right).
        geolocation: ``(latitude, longitude, altitude)``: degrees and metres.
        throttle, brake, steer, hand_brake, reverse, gear: The vehicle's
            control as last applied.
    """

    time: float
    frame: int
    location: Vector
    rotation: Vector
    velocity: Vector = (0.0, 0.0, 0.0)
    acceleration: Vector = (0.0, 0.0, 0.0)
    angular_velocity: Vector = (0.0, 0.0, 0.0)
    geolocation: Vector = (0.0, 0.0, 0.0)
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    gear: int = 0


def _axes(pitch: float, yaw: float, roll: float) -> Tuple[Vector, Vector, Vector]:
    """Unreal's ``FRotationMatrix`` rows: the body's forward, right and up axes
    in world coordinates (as carla_canvas's ``geometry.rotation_matrix``)."""
    p, y, r = math.radians(pitch), math.radians(yaw), math.radians(roll)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)
    forward = (cp * cy, cp * sy, sp)
    right = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    return forward, right, up


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _to_body(v: Vector, axes: Tuple[Vector, Vector, Vector]) -> Vector:
    """*v* (world, left-handed) in the body frame x forward, y left, z up."""
    forward, right, up = axes
    return (_dot(v, forward), -_dot(v, right), _dot(v, up))


def _bearing(east: float, north: float) -> float:
    """Clockwise from north, in ``[0, 360)``."""
    return math.degrees(math.atan2(east, north)) % 360.0


def ego_signals(state: EgoState) -> Dict[str, float]:
    """Every signal of :data:`SIGNALS` for *state*."""
    pitch, yaw, roll = state.rotation
    axes = _axes(pitch, yaw, roll)
    lon_v, lat_v, vert_v = _to_body(state.velocity, axes)
    lon_a, lat_a, vert_a = _to_body(state.acceleration, axes)
    # An angular velocity is a pseudo-vector: the reflection into the
    # right-handed body frame (y right -> left) also negates it.
    wx, wy, wz = _to_body(state.angular_velocity, axes)
    roll_rate, pitch_rate, yaw_rate = -wx, -wy, -wz
    vx, vy, vz = state.velocity
    # CARLA's geo-reference has world x east and y south.
    north, east = -vy, vx
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    latitude, longitude, altitude = state.geolocation
    return {
        "time": state.time,
        "frame": float(state.frame),
        "x": state.location[0],
        "y": state.location[1],
        "z": state.location[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "heading": (yaw + 90.0) % 360.0,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "latitude_rad": math.radians(latitude),
        "longitude_rad": math.radians(longitude),
        "velocity_north": north,
        "velocity_east": east,
        "velocity_up": vz,
        "course": _bearing(east, north),
        "speed": speed,
        "speed_kmh": speed * 3.6,
        "longitudinal_velocity": lon_v,
        "lateral_velocity": lat_v,
        "vertical_velocity": vert_v,
        "longitudinal_acceleration": lon_a,
        "lateral_acceleration": lat_a,
        "vertical_acceleration": vert_a,
        "roll_rate": roll_rate,
        "pitch_rate": pitch_rate,
        "yaw_rate": yaw_rate,
        "yaw_rate_rad": math.radians(yaw_rate),
        "throttle": state.throttle,
        "brake": state.brake,
        "steer": state.steer,
        "hand_brake": 1.0 if state.hand_brake else 0.0,
        "reverse": 1.0 if state.reverse else 0.0,
        "gear": float(state.gear),
    }
