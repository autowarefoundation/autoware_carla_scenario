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
