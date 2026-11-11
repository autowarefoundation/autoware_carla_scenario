"""CARLA measurements as the messages a hilspy channel sends.

* A ray-cast LiDAR's buffer (``{x, y, z, intensity}`` float32 per point, in
  the sensor's left-handed frame) becomes hilspy's ``scan`` (x forward, y
  left, z up; intensity 0-255).  hilspy's ``lidar:MODEL`` codec encodes one
  *revolution* into a sensor's packets, while CARLA measures the arc swept in
  one tick, so :class:`LidarRevolution` (:mod:`.lidar_revolution`, re-exported
  here) gathers the arcs into revolutions.
* A camera's BGRA image becomes one frame, as the device expects it
  (:data:`CAMERA_ENCODINGS`).

Nothing here imports CARLA or hilspy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .lidar_revolution import LidarRevolution

__all__ = [
    "CAMERA_ENCODINGS",
    "LidarRevolution",
    "camera_frame",
    "lidar_points",
    "lidar_scan_fields",
    "spin_lidar_points",
    "spin_timestamp",
]

FloatPoints = np.ndarray[tuple[int, int], np.dtype[np.float32]]

#: How a camera device takes its frames:
#:
#: * ``"bgra"`` -- CARLA's raw image as is (``v4l2://...?format=BGRA``).
#: * ``"bgr"``, ``"rgb"`` -- three bytes a pixel (``format=BGR`` / ``RGB``).
#: * ``"jpeg"``, ``"png"`` -- compressed (``format=jpeg`` / ``png``, or a
#:   byte transport such as ``udp://``).
CAMERA_ENCODINGS: Tuple[str, ...] = ("bgra", "bgr", "rgb", "jpeg", "png")


def lidar_points(raw: bytes | memoryview | np.ndarray) -> FloatPoints:
    """A ray-cast LiDAR buffer as an ``[N, 4]`` float32 array (sensor frame)."""
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)


def spin_lidar_points(
    points: np.ndarray, timestamp: float, period: float
) -> np.ndarray:
    """Re-observe a revolution's points as a spinning sensor would have seen them.

    *points* is ``[N, 11]``: CARLA's ``x, y, z, intensity`` (the sensor frame,
    x forward, y right, z up), then per point the time its arc was measured
    (CARLA casts an arc at once, from where the sensor was then) and the
    sensor's motion then in its own frame, ``vx, vy, vz`` (m/s) and the
    angular velocity ``wx, wy, wz`` (rad/s, CARLA's: ``wz`` turns x toward y).
    A sensor that turns fires the ray of azimuth ``atan2(y, x)`` (clockwise
    from forward) ``azimuth / 360`` of *period* after *timestamp*, the time
    hilspy dates it with; this moves each point, with the sensor's motion held
    constant, to where the sensor would have seen it then, so a driver
    correcting the motion over each point's time gets back a cloud of one
    instant.  Returns ``[N, 4]``.
    """
    # Work right-handed (y left): CARLA's frame mirrored in y; its angular
    # velocity, a pseudo-vector, mirrors to (-wx, wy, -wz).
    p = np.stack([points[:, 0], -points[:, 1], points[:, 2]], axis=1).astype(np.float64)
    measured = points[:, 4]
    v = np.stack([points[:, 5], -points[:, 6], points[:, 7]], axis=1)
    w = np.stack([-points[:, 8], points[:, 9], -points[:, 10]], axis=1)
    azimuth = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
    dt = (timestamp + azimuth / 360.0 * period - measured)[:, None]
    # The world seen from a sensor that moved by v*dt and turned by w*dt
    # (small angles: a revolution's dt is a fraction of a second).
    shifted = p - v * dt
    seen = shifted - np.cross(w * dt, shifted)
    out = np.empty((len(points), 4), np.float64)
    out[:, 0] = seen[:, 0]
    out[:, 1] = -seen[:, 1]
    out[:, 2] = seen[:, 2]
    out[:, 3] = points[:, 3]
    return out


def spin_timestamp(points: np.ndarray, period: float) -> float:
    """The revolution start that dates *points* (``[N, >=5]``, column 4 their
    measuring time) closest to when they were measured on average."""
    azimuth = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
    return float(np.mean(points[:, 4] - azimuth / 360.0 * period))


def lidar_scan_fields(
    points: FloatPoints, intensity_scale: float = 255.0
) -> Dict[str, np.ndarray]:
    """hilspy's ``scan`` fields for *points* (``[N, 4]``, CARLA's sensor frame).

    CARLA's intensity (0 to 1) is scaled by *intensity_scale* and clipped to a
    byte, as a sensor reports reflectivity.
    """
    return {
        "x": np.ascontiguousarray(points[:, 0]),
        "y": np.ascontiguousarray(-points[:, 1]),
        "z": np.ascontiguousarray(points[:, 2]),
        "intensity": np.clip(np.rint(points[:, 3] * intensity_scale), 0, 255).astype(
            np.uint8
        ),
    }


def camera_frame(
    raw: bytes | memoryview | np.ndarray, width: int, height: int, encoding: str
) -> np.ndarray | bytes:
    """A camera's BGRA image (CARLA's raw data) as *encoding* takes it."""
    if encoding not in CAMERA_ENCODINGS:
        raise ValueError(
            f"unknown camera encoding {encoding!r}; one of {', '.join(CAMERA_ENCODINGS)}"
        )
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    if encoding == "bgra":
        return bgra
    if encoding == "bgr":
        return np.ascontiguousarray(bgra[:, :, :3])
    if encoding == "rgb":
        return np.ascontiguousarray(bgra[:, :, 2::-1])
    import cv2  # noqa: PLC0415 - only the compressed encodings need it

    ok, buffer = cv2.imencode(
        ".jpg" if encoding == "jpeg" else ".png", np.ascontiguousarray(bgra[:, :, :3])
    )
    if not ok:
        raise ValueError(f"failed to encode a {width}x{height} frame as {encoding}")
    return buffer.tobytes()
