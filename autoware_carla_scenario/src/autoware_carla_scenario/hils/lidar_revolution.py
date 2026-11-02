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

#: A point this far short of a revolution's start, by its azimuth, is one at
#: the start: float rounding in CARLA's angle and in ``atan2`` must not put
#: the ray that starts a revolution at the end of the one before (a ray is a
#: hundredth of a degree or more from the next).
_START_TOLERANCE_DEG = 1e-3


def _angles(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """The horizontal angle each of *points* was measured at, in [0, 360)."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return (np.degrees(np.arctan2(y, x)) + _AZIMUTH_TO_ANGLE_DEG) % 360.0


def _arc_start(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]], arc: float
) -> float:
    """Where an arc of *arc* degrees (less than 360) started, from its points.

    The arc's points leave out the angle it did not sweep, ``360 - arc``
    degrees or more: the point after the widest gap between them is the
    first of the arc.  -1 when the arc has no points, or no gap that wide
    (missing hits can only widen gaps).
    """
    if points.shape[0] == 0:
        return -1.0
    angles = np.sort(_angles(points))
    gaps = np.diff(np.concatenate((angles, angles[:1] + 360.0)))
    widest = int(np.argmax(gaps))
    if float(gaps[widest]) + _REVOLUTION_TOLERANCE_DEG < 360.0 - arc:
        return -1.0
    return float(angles[(widest + 1) % angles.shape[0]])


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

    Revolutions start where the first arc with points starts, read off its
    points, and again after a new episode or an arc of a whole revolution
    or more; the angles swept in between are the simulated time's.  The
    first measurements a sensor takes need not arrive (CARLA turns a LiDAR
    from the tick it is spawned on, listened to or not), so where it
    started, at horizontal angle 0, is not where the first arc that arrives
    does.

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
    #: Whether :attr:`_start` was read off the points since the revolutions
    #: last started over.
    _anchored: bool
    _parts: List[np.ndarray[tuple[int, int], np.dtype[np.float32]]]

    def __init__(self, rotation_frequency: float, first_dt: float) -> None:
        if rotation_frequency <= 0.0 or first_dt <= 0.0:
            raise ValueError("rotation_frequency and first_dt must be positive")
        self._per_second = 360.0 * rotation_frequency
        self._first_dt = first_dt
        self._time = None
        self._swept = 0.0
        self._start = 0.0
        self._anchored = False
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
            self._anchored = False
            dt = self._first_dt
        self._time = time
        arc = self._per_second * dt
        before = self._swept
        swept = before + arc
        if not self._anchored and arc + _REVOLUTION_TOLERANCE_DEG < 360.0:
            start = _arc_start(points, arc)
            if start >= 0.0:
                self._start = (start - before) % 360.0
                self._anchored = True
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
            # Not every renderer sweeps the whole arc (carla_canvas sweeps
            # one revolution at most): the next arc's points tell.
            self._anchored = False
            return points
        leftover = swept - 360.0
        if leftover < _REVOLUTION_TOLERANCE_DEG:
            # The arc ends on the revolution's end (a sweep dividing 360).
            # The next one starts where it did end, a hair either side of it
            # (float sums of tick lengths), so that hairs do not add up.
            self._parts.append(points)
            self._swept = 0.0
            self._start = (self._start + leftover) % 360.0
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
        relative = (
            _angles(points) + (_START_TOLERANCE_DEG - self._start)
        ) % 360.0 - _START_TOLERANCE_DEG
        ends = relative >= 0.5 * (leftover + before)
        self._parts.append(points[ends])
        self._swept = leftover
        revolution = np.concatenate(self._parts)
        self._parts = [points[~ends]]
        return revolution
