"""CARLA measurements as the messages a hilspy channel sends.

* A ray-cast LiDAR's buffer (``{x, y, z, intensity}`` float32 per point, in
  the sensor's left-handed frame) becomes hilspy's ``scan`` (x forward, y
  left, z up; intensity 0-255).  hilspy's ``lidar:MODEL`` codec encodes one
  *revolution* into a sensor's packets, while CARLA measures the arc swept in
  one tick, so :class:`LidarRevolution` gathers the arcs into revolutions.
* A camera's BGRA image becomes one frame, as the device expects it
  (:data:`CAMERA_ENCODINGS`).

Nothing here imports CARLA or hilspy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CAMERA_ENCODINGS",
    "LidarRevolution",
    "camera_frame",
    "lidar_points",
    "lidar_scan_fields",
]

FloatPoints = np.ndarray[tuple[int, int], np.dtype[np.float32]]

#: How a camera device takes its frames:
#:
#: * ``"bgra"`` -- CARLA's raw image as is (``v4l2://...?format=BGRA``).
#: * ``"bgr"``, ``"rgb"`` -- three bytes a pixel (``format=BGR`` / ``RGB``).
#: * ``"jpeg"``, ``"png"`` -- compressed (``format=jpeg`` / ``png``, or a
#:   byte transport such as ``udp://``).
CAMERA_ENCODINGS: Tuple[str, ...] = ("bgra", "bgr", "rgb", "jpeg", "png")

#: An arc this close to a whole revolution completes it (float sums of tick
#: lengths fall a hair short).
_REVOLUTION_TOLERANCE_DEG = 1e-3


def lidar_points(raw: bytes | memoryview | np.ndarray) -> FloatPoints:
    """A ray-cast LiDAR buffer as an ``[N, 4]`` float32 array (sensor frame)."""
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)


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


class LidarRevolution:
    """Gathers a LiDAR's per-tick arcs into whole revolutions.

    A measurement sweeps ``rotation_frequency * 360 * dt`` degrees, ``dt``
    being the simulated time since the previous one; :meth:`add` returns the
    points of a revolution once the arcs add up to one, and keeps the rest
    for the next.  A LiDAR that turns once a tick (or once a ``sensor_tick``)
    yields each measurement as it comes.

    Args:
        rotation_frequency: The blueprint's ``rotation_frequency``, Hz.
        first_dt: The simulated time the first measurement covers (the
            world's tick length).
    """

    _per_second: float
    _first_dt: float
    _time: Optional[float]
    _swept: float
    _parts: List[FloatPoints]

    def __init__(self, rotation_frequency: float, first_dt: float) -> None:
        if rotation_frequency <= 0.0 or first_dt <= 0.0:
            raise ValueError("rotation_frequency and first_dt must be positive")
        self._per_second = 360.0 * rotation_frequency
        self._first_dt = first_dt
        self._time = None
        self._swept = 0.0
        self._parts = []

    def add(self, time: float, points: FloatPoints) -> Optional[FloatPoints]:
        """Add the measurement taken at *time*; a whole revolution, or ``None``."""
        dt = self._first_dt if self._time is None else time - self._time
        if dt <= 0.0:  # a new episode, its clock restarted: start over
            self._parts.clear()
            self._swept = 0.0
            dt = self._first_dt
        self._time = time
        self._parts.append(points)
        self._swept += self._per_second * dt
        if self._swept + _REVOLUTION_TOLERANCE_DEG < 360.0:
            return None
        self._swept = max(0.0, self._swept - 360.0)
        if self._swept < _REVOLUTION_TOLERANCE_DEG:
            self._swept = 0.0
        revolution = (
            self._parts[0] if len(self._parts) == 1 else np.concatenate(self._parts)
        )
        self._parts = []
        return revolution


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
