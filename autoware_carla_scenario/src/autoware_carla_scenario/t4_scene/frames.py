"""The frame math of a T4Scene: CARLA's left-handed world frame to Autoware's
right-handed ``map`` frame (docs/trajectory_log.md,
``coordinate.transform.to_map_frame``), and the rotations and quaternions of
map-frame roll, pitch and yaw.

``y`` flips and the map's offsets are added; ``roll`` keeps its sign,
``pitch`` and ``yaw`` flip.  t4_scene.convert re-exports these under its own
protected names (``_FLIP_Y``, ``_rotations``, ``_quaternions``) and its
``_MapFrame`` calls :func:`map_positions` and :func:`map_angles`; this module
is pure numpy, so Codon checks it (#1665).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FLIP_Y",
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
