"""The frame math of a T4Scene: CARLA's left-handed world frame to Autoware's
right-handed ``map`` frame (docs/trajectory_log.md,
``coordinate.transform.to_map_frame``), and the rotations and quaternions of
map-frame roll, pitch and yaw.

``y`` flips and the map's offsets are added; ``roll`` keeps its sign,
``pitch`` and ``yaw`` flip.  t4_scene.convert re-exports these under its own
protected names (``_FLIP_Y``, ``_rotations``, ``_quaternions``) and its
``_MapFrame`` calls :func:`map_positions` and :func:`map_angles`.
:func:`checked_rear_axle_offset`, the check of where the ego's ``base_link``
is, lives here too (#1697), and t4_scene.convert re-exports it.  This module
is pure numpy and ``math``, so Codon checks it (#1665).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "FLIP_Y",
    "checked_rear_axle_offset",
    "map_angles",
    "map_positions",
    "quaternions",
    "rotations",
]

#: CARLA's left-handed frame to the map's right-handed one: ``y`` flips.
FLIP_Y = np.array([1.0, -1.0, 1.0])


def map_positions(
    xyz: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    offset: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """(N, 3) CARLA locations as map-frame positions, *offset* (the map's
    ``(x, y, z)`` offsets) added.  A single (3,) location broadcasts too."""
    return xyz * FLIP_Y + offset


def map_angles(
    rpy_deg: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """(N, 3) CARLA rotations (roll, pitch, yaw; degrees) as map-frame
    Euler angles (radians)."""
    return np.radians(rpy_deg) * np.array([1.0, -1.0, -1.0])


def rotations(
    angles: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """(N, 3, 3) body-to-map rotations of (N, 3) roll/pitch/yaw (ZYX)."""
    # Codon cannot unpack an ndarray, so the columns are taken one by one.
    c = np.cos(angles)
    s = np.sin(angles)
    cr, cp, cw = c[:, 0], c[:, 1], c[:, 2]
    sr, sp, sw = s[:, 0], s[:, 1], s[:, 2]
    return np.stack(
        [
            np.stack(
                [cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr], axis=-1
            ),
            np.stack(
                [sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr], axis=-1
            ),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=-2,
    )


def quaternions(
    angles: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """(N, 4) unit quaternions ``(x, y, z, w)`` of (N, 3) roll/pitch/yaw (ZYX)."""
    c = np.cos(angles / 2)
    s = np.sin(angles / 2)
    cr, cp, cw = c[:, 0], c[:, 1], c[:, 2]
    sr, sp, sw = s[:, 0], s[:, 1], s[:, 2]
    return np.stack(
        [
            sr * cp * cw - cr * sp * sw,
            cr * sp * cw + sr * cp * sw,
            cr * cp * sw - sr * sp * cw,
            cr * cp * cw + sr * sp * sw,
        ],
        axis=-1,
    )


def checked_rear_axle_offset(
    rear_axle_offset_m: float, key: str = "rear_axle_offset_m"
) -> float:
    """*rear_axle_offset_m* as a float, which must be finite.

    The offset is where the ego's rear axle (its ``base_link``) is from its
    actor origin along its x axis: negative behind the origin -- the usual
    case, and the default's ``-0.5 * extent.x`` -- 0 at it, positive ahead of
    it, so any finite value is a place.  NaN or infinity would put the ego's
    every pose and ``base_link_to_front`` at NaN, so either is refused before
    a scene is written.  A bool is refused too: ``true`` is not an offset,
    though ``float()`` would read it as 1 m.

    Args:
        rear_axle_offset_m: The offset, metres.
        key: The setting's name, which the error names (``--rear-axle-offset``).

    Raises:
        ValueError: If *rear_axle_offset_m* is a bool, or not finite.
    """
    value = float(rear_axle_offset_m)
    if isinstance(rear_axle_offset_m, bool) or not math.isfinite(value):
        raise ValueError(f"{key} must be finite, not {rear_axle_offset_m}")
    return value
