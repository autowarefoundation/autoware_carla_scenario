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
    _parts: List[np.ndarray[tuple[int, int], np.dtype[np.float32]]]

    def __init__(self, rotation_frequency: float, first_dt: float) -> None:
        if rotation_frequency <= 0.0 or first_dt <= 0.0:
            raise ValueError("rotation_frequency and first_dt must be positive")
        self._per_second = 360.0 * rotation_frequency
        self._first_dt = first_dt
        self._time = None
        self._swept = 0.0
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
            dt = self._first_dt
        self._time = time
        self._parts.append(points)
        self._swept += self._per_second * dt
        if self._swept + _REVOLUTION_TOLERANCE_DEG < 360.0:
            return None
        self._swept = max(0.0, self._swept - 360.0)
        if self._swept < _REVOLUTION_TOLERANCE_DEG:
            self._swept = 0.0
        if len(self._parts) == 1:
            revolution = self._parts[0]
        else:
            revolution = np.concatenate(self._parts)
        self._parts = []
        return revolution
