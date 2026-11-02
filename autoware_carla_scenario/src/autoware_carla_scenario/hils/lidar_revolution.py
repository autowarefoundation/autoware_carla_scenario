"""A LiDAR's per-tick arcs, gathered into the whole revolutions a sensor sends.

hilspy's ``lidar:MODEL`` codec encodes one *revolution* into a sensor's
packets, while CARLA measures the arc a LiDAR sweeps in one tick;
:class:`LidarRevolution` adds the arcs up and hands back the points of each
revolution.  It needs only numpy: :mod:`.measurements` re-exports it beside
the conversions that do not compile.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["LidarRevolution"]

#: An arc this close to a whole revolution completes it (float sums of tick
#: lengths fall a hair short).
_REVOLUTION_TOLERANCE_DEG = 1e-3

#: CARLA's ray-cast LiDAR shoots its rays at yaw ``h - horizontal_fov / 2``,
#: ``h`` being its horizontal angle (0 when spawned, growing by the arc of
#: each tick): a point's azimuth (``atan2(y, x)`` in CARLA's sensor frame,
#: y right) plus this is the angle ``h`` it was measured at.
_AZIMUTH_TO_ANGLE_DEG = 180.0


class LidarRevolution:
    """Gathers a LiDAR's per-tick arcs into whole revolutions.

    A measurement sweeps ``rotation_frequency * 360 * dt`` degrees, ``dt``
    being the simulated time since the previous one; :meth:`add` returns the
    points of a revolution once the arcs add up to one, and keeps the rest
    for the next: the arc that completes a revolution is split at the
    azimuth where the revolution started, so each one covers 360 degrees
    once.  A LiDAR that turns once a tick (or once a ``sensor_tick``) yields
    each measurement as it comes, and so does an arc of a whole revolution
    or more (a slow tick): it covers every azimuth already, and the partial
    revolution gathered before it, measured earlier at the same azimuths, is
    dropped.

    Revolutions start where the sensor did (CARLA turns a LiDAR from
    horizontal angle 0, its rays pointing backwards, when it is spawned),
    and after an arc of a whole revolution or more, where that arc ended.

    Args:
        rotation_frequency: The blueprint's ``rotation_frequency``, Hz.
        first_dt: The simulated time the first measurement covers (the
            world's tick length).
    """

    _per_second: float
    _first_dt: float
    _time: Optional[float]
    #: Degrees swept since the current revolution started.
    _swept: float
    #: The sensor's horizontal angle at which the current revolution
    #: started, in degrees, modulo 360.
    _start: float
    _parts: List[np.ndarray[tuple[int, int], np.dtype[np.float32]]]

    def __init__(self, rotation_frequency: float, first_dt: float) -> None:
        if rotation_frequency <= 0.0 or first_dt <= 0.0:
            raise ValueError("rotation_frequency and first_dt must be positive")
        self._per_second = 360.0 * rotation_frequency
        self._first_dt = first_dt
        self._time = None
        self._swept = 0.0
        self._start = 0.0
        self._parts = []

    def add(
        self, time: float, points: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    ) -> Optional[np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """Add the measurement taken at *time*; a whole revolution, or ``None``."""
        previous = self._time
        if previous is None:
            dt = self._first_dt
        else:
            dt = time - previous
        if dt <= 0.0:  # a new episode, its clock restarted: start over
            self._parts.clear()
            self._swept = 0.0
            self._start = 0.0
            dt = self._first_dt
        self._time = time
        arc = self._per_second * dt
        before = self._swept
        swept = before + arc
        if swept + _REVOLUTION_TOLERANCE_DEG < 360.0:
            self._parts.append(points)
            self._swept = swept
            return None
        if arc + _REVOLUTION_TOLERANCE_DEG >= 360.0:
            # The arc alone covers every azimuth: it is the revolution, and
            # the next one starts where it ended.
            self._parts = []
            self._swept = 0.0
            self._start = (self._start + swept) % 360.0
            return points
        leftover = swept - 360.0
        if leftover < _REVOLUTION_TOLERANCE_DEG:
            # The arc ends on the revolution's end (a sweep dividing 360).
            self._parts.append(points)
            self._swept = 0.0
            if len(self._parts) == 1:
                revolution = self._parts[0]
            else:
                revolution = np.concatenate(self._parts)
            self._parts = []
            return revolution
        # The arc runs past the revolution's end: split it by each point's
        # azimuth, not by its place in the buffer.  CARLA lays the points out
        # channel by channel, each channel's in the order it turned, so the
        # first part of the buffer is the first channels' whole arc, not the
        # first part of the arc.  Relative to the revolution's start, the arc
        # covers [before, 360) then [0, leftover); half way through the angle
        # it did not sweep, [leftover, before), is the cut furthest from both.
        azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        relative = (azimuth + (_AZIMUTH_TO_ANGLE_DEG - self._start)) % 360.0
        ends = relative >= 0.5 * (leftover + before)
        self._parts.append(points[ends])
        self._swept = leftover
        revolution = np.concatenate(self._parts)
        self._parts = [points[~ends]]
        return revolution
