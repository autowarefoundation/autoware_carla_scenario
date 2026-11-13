"""A LiDAR revolution's points dated and re-observed as a spinning sensor's.

CARLA casts each tick's arc at once; a turning sensor fires each ray at a
time of its own.  :func:`ray_times` says when the sensor fires each point's
ray, :func:`steady_revolution_start` when a revolution starts, kept on the
sensor's period, and :func:`spin_lidar_points` moves each point to where the
sensor would have seen it then.  It needs only numpy: :mod:`.measurements`
re-exports it beside the conversions that do not compile.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ray_times", "spin_lidar_points", "steady_revolution_start"]


def ray_times(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]], period: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Seconds after a revolution's start a turning sensor fires each point's ray.

    hilspy dates the ray of azimuth ``atan2(y, x)`` (CARLA's sensor frame,
    clockwise from forward) ``azimuth / 360`` of *period* after the scan's
    timestamp, as the sensor fires it.
    """
    y, x = points[:, 1].astype(np.float64), points[:, 0].astype(np.float64)
    return np.mod(np.arctan2(y, x), 2.0 * np.pi) * (period / (2.0 * np.pi))


def steady_revolution_start(
    measured: float, previous: float | None, period: float, gain: float = 0.2
) -> float:
    """When a revolution measured to start at *measured* is dated, *previous*
    being the last one's start.

    A turning sensor starts each revolution one *period* after the last, and
    its packets' times never step back.  CARLA's ticks come a few ms early or
    late on the wall clock, so the measured starts jitter; dated as they are,
    a revolution can start before the last one's packets end, which a viewer
    such as PandarView shows as a missing wedge.  The start therefore follows
    the period and only drifts later toward the measured one (*gain* of the
    difference): CARLA ticks no faster than the wall clock, so the measured
    starts do not run ahead for long.  Where one is a whole period off (a
    revolution missed, a stall) the stream restarts on it.  The points are
    re-observed for the start chosen (:func:`spin_lidar_points`), so the
    cloud stays right.
    """
    if previous is None:
        return measured
    expected = previous + period
    error = measured - expected
    if abs(error) >= period:
        return measured
    return expected + max(0.0, gain * error)


def spin_lidar_points(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    measured: np.ndarray[tuple[int], np.dtype[np.float64]],
    motion: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    timestamp: float,
    rays: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Re-observe a revolution's points as a spinning sensor would have seen them.

    CARLA casts each tick's arc at once, from where the sensor was then
    (*measured*, per point); a turning sensor fires each ray at
    ``timestamp + rays`` (:func:`ray_times`).  Each point moves, with the
    sensor's *motion* then held constant (per point ``vx, vy, vz`` in m/s and
    ``wx, wy, wz`` in rad/s, CARLA's sensor frame, whose angular velocity
    follows the left-hand rule: ``wz`` turns x toward y), to where the sensor
    would have seen it at its ray's time -- small angles, first order -- so a
    driver correcting the motion over each point's time gets back a cloud of
    one instant.  *points* holds CARLA's ``x, y, z, intensity`` in its first
    four columns, ``[N, 4]`` or wider (the bridge passes each point's arc as a
    fifth); the result is ``[N, 4]``, those four.
    """
    # Differences first: epoch seconds do not fit a float32 (CARLA's points are).
    dt = (timestamp - np.asarray(measured, np.float64)) + np.asarray(rays, np.float64)
    # One column at a time, in order (Codon unpacks no generator).
    vx = motion[:, 0] * dt
    vy = motion[:, 1] * dt
    vz = motion[:, 2] * dt
    wx = motion[:, 3] * dt
    wy = motion[:, 4] * dt
    wz = motion[:, 5] * dt
    a = points[:, 0] - vx
    b = points[:, 1] - vy
    c = points[:, 2] - vz
    out = np.empty((len(points), 4), np.float32)
    out[:, 0] = a - wy * c + wz * b
    out[:, 1] = b - wz * a + wx * c
    out[:, 2] = c - wx * b + wy * a
    out[:, 3] = points[:, 3]
    return out
